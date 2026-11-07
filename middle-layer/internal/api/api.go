// Package api is the HTTP surface the frontend talks to.
//
//	GET  /health                  backend reachable? worker running? sandbox?
//	GET  /policies                what the backend serves
//	GET  /sessions                every session, without transcripts
//	POST /sessions                {policy} -> a new session
//	GET  /sessions/{id}           one session with its episodes
//	POST /sessions/{id}/messages  {content} -> the episode it caused
//
// The last one returns the finished episode as JSON, or with
// `Accept: text/event-stream` streams each step as it happens. That stream is
// what the live tool view is built on. Disconnecting cancels the episode.
//
// The server has no authentication, and a request to it can end with generated
// code running in the sandbox. It therefore only ever listens on loopback, and
// guard stops the two ways a web page could reach it from the user's browser.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/session"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/toolworker"
)

// Far above any real request. The conversation limit is what bounds a prompt, and
// this only bounds what gets read before that check can run.
const maxBodyBytes = 1 << 20

// Backend is what the API needs beyond the loop itself.
type Backend interface {
	Policies(ctx context.Context) (json.RawMessage, []string, error)
	Health(ctx context.Context) error
}

// Worker is what /health reports on.
type Worker interface {
	Info() toolworker.Info
	Alive() bool
}

// Runner is agent.Runner, as an interface so tests can hold an episode open.
type Runner interface {
	Run(ctx context.Context, id, policy string, history []agent.Message, prompt string, emit func(agent.Event)) agent.Episode
}

type Server struct {
	Backend Backend
	Worker  Worker
	Runner  Runner
	Store   *session.Store
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /policies", s.policies)
	mux.HandleFunc("GET /sessions", s.listSessions)
	mux.HandleFunc("POST /sessions", s.createSession)
	mux.HandleFunc("GET /sessions/{id}", s.getSession)
	mux.HandleFunc("POST /sessions/{id}/messages", s.postMessage)
	return guard(mux)
}

// guard stops a web page in the user's browser from driving this server.
//
// A cross-site POST only avoids a CORS preflight with a "simple" content type, so
// requiring JSON forces the preflight, which fails because no CORS headers are
// ever sent. DNS rebinding gets around that by making the page same-origin, and
// the Host check stops it, since the Host is then the attacker's domain.
func guard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !loopbackHost(r.Host) {
			writeError(w, http.StatusForbidden, "this server only answers to localhost")
			return
		}
		if r.Method == http.MethodPost {
			media, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
			if err != nil || media != "application/json" {
				writeError(w, http.StatusUnsupportedMediaType, "send application/json")
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
		}
		next.ServeHTTP(w, r)
	})
}

func loopbackHost(hostport string) bool {
	host, _, err := net.SplitHostPort(hostport)
	if err != nil {
		host = strings.Trim(hostport, "[]")
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

// readJSON decodes a request body strictly, writing the error itself on failure.
func readJSON(w http.ResponseWriter, r *http.Request, into any) bool {
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(into); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
		} else {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("bad JSON: %v", err))
		}
		return false
	}
	return true
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	status := http.StatusOK
	backend := "ok"
	if err := s.Backend.Health(ctx); err != nil {
		status = http.StatusServiceUnavailable
		backend = err.Error()
	}
	info := s.Worker.Info()
	writeJSON(w, status, map[string]any{
		"backend": backend,
		"tools": map[string]any{
			// False is not fatal: the next episode starts a new worker.
			"worker_running": s.Worker.Alive(),
			"tools":          info.Tools,
			"sandbox":        info.Sandbox,
		},
	})
}

func (s *Server) policies(w http.ResponseWriter, r *http.Request) {
	raw, _, err := s.Backend.Policies(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, raw)
}

func (s *Server) listSessions(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"sessions": s.Store.List()})
}

func (s *Server) createSession(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Policy string `json:"policy"`
	}
	if !readJSON(w, r, &request) {
		return
	}
	// Asked each time rather than cached: the backend can restart with other
	// adapters while this process keeps running.
	_, names, err := s.Backend.Policies(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	for _, name := range names {
		if name == request.Policy {
			writeJSON(w, http.StatusCreated, s.Store.Create(request.Policy))
			return
		}
	}
	writeError(w, http.StatusUnprocessableEntity,
		fmt.Sprintf("no policy named %q; serving %s", request.Policy, strings.Join(names, ", ")))
}

func (s *Server) getSession(w http.ResponseWriter, r *http.Request) {
	found, err := s.Store.Get(r.PathValue("id"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, found)
}

func (s *Server) postMessage(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Content string `json:"content"`
	}
	if !readJSON(w, r, &request) {
		return
	}
	if strings.TrimSpace(request.Content) == "" {
		writeError(w, http.StatusUnprocessableEntity, "content is empty")
		return
	}

	id := r.PathValue("id")
	begun, err := s.Store.Begin(id, request.Content)
	switch {
	case errors.Is(err, session.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
		return
	case errors.Is(err, session.ErrBusy):
		writeError(w, http.StatusConflict, err.Error())
		return
	case errors.Is(err, session.ErrTooLong):
		writeError(w, http.StatusRequestEntityTooLarge, err.Error())
		return
	case err != nil:
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var emit func(agent.Event)
	stream := wantsStream(r)
	if stream {
		emit = startStream(w)
	}
	episode := s.Runner.Run(r.Context(), begun.EpisodeID, begun.Policy, begun.History, request.Content, emit)
	s.Store.End(id, episode)

	if stream {
		return
	}
	status := http.StatusOK
	if episode.Failed() {
		status = http.StatusBadGateway
	}
	writeJSON(w, status, episode)
}

func wantsStream(r *http.Request) bool {
	return strings.Contains(r.Header.Get("Accept"), "text/event-stream")
}

// startStream sends the event-stream headers and returns the function that writes
// each event. A write can fail once the client has gone, and that is ignored: the
// request context is cancelled too, which is what stops the episode.
func startStream(w http.ResponseWriter) func(agent.Event) {
	header := w.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	controller := http.NewResponseController(w)
	_ = controller.Flush()

	return func(event agent.Event) {
		data, err := json.Marshal(event)
		if err != nil {
			return
		}
		// JSON has no raw newlines, so one data line always holds the whole event.
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, data); err != nil {
			return
		}
		_ = controller.Flush()
	}
}
