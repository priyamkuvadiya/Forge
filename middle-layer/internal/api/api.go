// Package api is the HTTP surface the frontend talks to.
//
//	GET  /health                  backend reachable? worker running? sandbox?
//	GET  /policies                what the backend serves
//	GET  /sessions                every session, without transcripts
//	POST /sessions                {policy} -> a new session
//	GET  /sessions/{id}           one session with its episodes
//	POST /sessions/{id}/messages  {content} -> the episode it caused
//	GET  /sessions/{id}/live      the running episode's events, from the first
//	POST /sessions/{id}/cancel    stop the running episode
//	GET  /runs                    every imported training run, without records
//	GET  /runs/{name}             one run: its config and every step record
//
// A message returns the finished episode as JSON, or with
// `Accept: text/event-stream` streams each step as it happens. That stream is
// what the live tool view is built on. The episode belongs to the server, not to
// the request: disconnecting leaves it running, /live picks it up again with
// everything it has done so far, and only /cancel or shutdown stops it.
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
	"io/fs"
	"mime"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/metrics"
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
	Metrics *metrics.Store
	// Context is the server's lifetime. Episodes run on it, so cancelling it
	// stops them all; then Wait for their records. Nil means never cancelled.
	Context context.Context
	// Frontend is the built page (frontend/dist), served at /. Nil serves none.
	Frontend fs.FS

	liveMu   sync.Mutex
	live     map[string]*live // Session ID -> its running episode.
	episodes sync.WaitGroup
}

// Handler is called once per Server.
func (s *Server) Handler() http.Handler {
	s.live = map[string]*live{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /policies", s.policies)
	mux.HandleFunc("GET /sessions", s.listSessions)
	mux.HandleFunc("POST /sessions", s.createSession)
	mux.HandleFunc("GET /sessions/{id}", s.getSession)
	mux.HandleFunc("POST /sessions/{id}/messages", s.postMessage)
	mux.HandleFunc("GET /sessions/{id}/live", s.followLive)
	mux.HandleFunc("POST /sessions/{id}/cancel", s.cancelEpisode)
	mux.HandleFunc("GET /runs", s.listRuns)
	mux.HandleFunc("GET /runs/{name}", s.getRun)

	// The API at the root, as before, and again under /api/, which is where the
	// page calls it. That one prefix is all the page needs to work the same
	// whether this server serves it or Vite's dev server forwards to it.
	root := http.NewServeMux()
	root.Handle("/", mux)
	root.Handle("/api/", http.StripPrefix("/api", mux))
	if s.Frontend != nil {
		// Less specific than every API route, so it only gets what they do not.
		static := staticFiles(s.Frontend)
		root.Handle("GET /{$}", static)
		root.Handle("GET /index.html", static)
		root.Handle("GET /assets/", static)
		root.Handle("GET /favicon.ico", static)
	}
	return guard(root)
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
	summaries, err := s.Store.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"sessions": summaries})
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
			created, err := s.Store.Create(request.Policy)
			if err != nil {
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}
			writeJSON(w, http.StatusCreated, created)
			return
		}
	}
	writeError(w, http.StatusUnprocessableEntity,
		fmt.Sprintf("no policy named %q; serving %s", request.Policy, strings.Join(names, ", ")))
}

func (s *Server) getSession(w http.ResponseWriter, r *http.Request) {
	found, err := s.Store.Get(r.PathValue("id"))
	switch {
	case errors.Is(err, session.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
		return
	case err != nil:
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, found)
}

func (s *Server) listRuns(w http.ResponseWriter, _ *http.Request) {
	summaries, err := s.Metrics.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"runs": summaries})
}

func (s *Server) getRun(w http.ResponseWriter, r *http.Request) {
	run, err := s.Metrics.Get(r.PathValue("name"))
	switch {
	case errors.Is(err, metrics.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
		return
	case err != nil:
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, run)
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

	running := s.start(id, begun, request.Content)
	if wantsStream(r) {
		running.follow(r.Context(), startStream(w))
		return
	}
	episode, ok := running.wait(r.Context())
	if !ok {
		return // The client left; the episode carries on without it.
	}
	status := http.StatusOK
	if episode.Failed() {
		status = http.StatusBadGateway
	}
	writeJSON(w, status, episode)
}

// followLive streams the running episode from its first event, for a page that
// was not there when it started. 204 means nothing is running: reload the
// session, where a just-finished episode already is.
func (s *Server) followLive(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	running := s.running(id)
	if running == nil {
		s.nothingRunning(w, id, http.StatusNoContent)
		return
	}
	running.follow(r.Context(), startStream(w))
}

// cancelEpisode stops the running episode. It returns at once; the episode then
// ends as cancelled, which its stream and the session show as usual.
func (s *Server) cancelEpisode(w http.ResponseWriter, r *http.Request) {
	// The body is {} and carries nothing, but it is read anyway. A client that
	// sends Connection: close gets its socket closed after the reply, and on
	// Windows closing one with unread bytes in it sends a reset, which can arrive
	// before the reply does. Measured: 8 of 25 cancels lost their 202 without
	// this, 0 of 25 with it.
	var request struct{}
	if !readJSON(w, r, &request) {
		return
	}
	id := r.PathValue("id")
	running := s.running(id)
	if running == nil {
		s.nothingRunning(w, id, http.StatusConflict)
		return
	}
	running.cancel()
	writeJSON(w, http.StatusAccepted, map[string]string{"cancelling": running.episodeID})
}

// nothingRunning answers for a session with no live episode: 404 if there is no
// such session, otherwise the given status.
func (s *Server) nothingRunning(w http.ResponseWriter, id string, status int) {
	_, err := s.Store.Get(id)
	switch {
	case errors.Is(err, session.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	case err != nil:
		writeError(w, http.StatusInternalServerError, err.Error())
	case status == http.StatusNoContent:
		w.WriteHeader(status)
	default:
		writeError(w, status, "nothing is running in this session")
	}
}

func wantsStream(r *http.Request) bool {
	return strings.Contains(r.Header.Get("Accept"), "text/event-stream")
}

// startStream sends the event-stream headers and returns the function that writes
// each event. A write can fail once the client has gone, and that is ignored: the
// request context is cancelled too, which is what stops the following (and only
// the following; the episode is not the request's).
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
