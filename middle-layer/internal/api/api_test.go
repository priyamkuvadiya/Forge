package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/session"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/toolworker"
)

type fakeBackend struct{ down bool }

func (b *fakeBackend) Policies(context.Context) (json.RawMessage, []string, error) {
	if b.down {
		return nil, nil, errors.New("backend unreachable")
	}
	return json.RawMessage(`{"policies":[{"name":"baseline"},{"name":"run2"}]}`), []string{"baseline", "run2"}, nil
}

func (b *fakeBackend) Health(context.Context) error {
	if b.down {
		return errors.New("backend unreachable")
	}
	return nil
}

type fakeWorker struct{ alive bool }

func (w *fakeWorker) Info() toolworker.Info { return toolworker.Info{Tools: []string{"calculator"}} }
func (w *fakeWorker) Alive() bool           { return w.alive }

// fakeRunner returns its reason, or holds the episode open until told to finish.
type fakeRunner struct {
	reason  string
	hold    chan struct{}
	entered chan struct{}
	left    chan error
	history []agent.Message
}

func (r *fakeRunner) Run(ctx context.Context, id, policy string, history []agent.Message, prompt string, emit func(agent.Event)) agent.Episode {
	if emit == nil {
		emit = func(agent.Event) {}
	}
	r.history = history
	emit(agent.Event{Type: agent.EventStarted, Episode: id})
	episode := agent.Episode{
		ID:           id,
		Policy:       policy,
		FinishReason: r.reason,
		Messages:     []agent.Message{{Role: "user", Content: prompt}, {Role: "assistant", Content: "line one\nline two"}},
	}
	if r.hold != nil && prompt != "not held" {
		close(r.entered)
		select {
		case <-r.hold:
		case <-ctx.Done():
			episode.FinishReason = agent.FinishCancelled
			r.left <- ctx.Err()
		}
	}
	emit(agent.Event{Type: agent.EventTurn, Episode: id, Text: "line one\nline two"})
	emit(agent.Event{Type: agent.EventFinished, Episode: id, Finished: &episode})
	return episode
}

type fixture struct {
	*httptest.Server
	backend *fakeBackend
	worker  *fakeWorker
	runner  *fakeRunner
	store   *session.Store
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{
		backend: &fakeBackend{},
		worker:  &fakeWorker{alive: true},
		runner:  &fakeRunner{reason: agent.FinishAnswer},
		store:   session.NewStore(1000),
	}
	server := &Server{Backend: f.backend, Worker: f.worker, Runner: f.runner, Store: f.store}
	f.Server = httptest.NewServer(server.Handler())
	t.Cleanup(f.Close)
	return f
}

func (f *fixture) post(t *testing.T, path, body string, headers ...string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, f.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	for i := 0; i+1 < len(headers); i += 2 {
		request.Header.Set(headers[i], headers[i+1])
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	return response
}

func (f *fixture) get(t *testing.T, path string) *http.Response {
	t.Helper()
	response, err := http.Get(f.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	return response
}

func decode[T any](t *testing.T, response *http.Response) T {
	t.Helper()
	var value T
	if err := json.NewDecoder(response.Body).Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value
}

func expect(t *testing.T, response *http.Response, status int) {
	t.Helper()
	if response.StatusCode != status {
		body, _ := io.ReadAll(response.Body)
		t.Fatalf("status = %d, want %d: %s", response.StatusCode, status, body)
	}
}

func (f *fixture) session(t *testing.T) string {
	t.Helper()
	response := f.post(t, "/sessions", `{"policy":"run2"}`)
	expect(t, response, http.StatusCreated)
	return decode[session.Session](t, response).ID
}

func TestAWebPageCannotDriveTheServer(t *testing.T) {
	f := newFixture(t)
	handler := (&Server{Backend: f.backend, Worker: f.worker, Runner: f.runner, Store: f.store}).Handler()

	send := func(host, contentType string) int {
		request := httptest.NewRequest(http.MethodPost, "/sessions", strings.NewReader(`{"policy":"run2"}`))
		request.Host = host
		if contentType != "" {
			request.Header.Set("Content-Type", contentType)
		}
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		return recorder.Code
	}

	for host, want := range map[string]int{
		"evil.example":       http.StatusForbidden, // DNS rebinding
		"evil.example:8080":  http.StatusForbidden,
		"localhost.evil.com": http.StatusForbidden,
		"127.0.0.1.evil.com": http.StatusForbidden,
		"192.168.1.10:8080":  http.StatusForbidden,
		"localhost:8080":     http.StatusCreated,
		"127.0.0.1:8080":     http.StatusCreated,
		"[::1]:8080":         http.StatusCreated,
		"localhost":          http.StatusCreated,
	} {
		if got := send(host, "application/json"); got != want {
			t.Errorf("Host %q: status %d, want %d", host, got, want)
		}
	}

	// The content types a cross-site form can send with no CORS preflight.
	for _, contentType := range []string{
		"", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x",
	} {
		if got := send("localhost:8080", contentType); got != http.StatusUnsupportedMediaType {
			t.Errorf("Content-Type %q: status %d", contentType, got)
		}
	}
	if got := send("localhost:8080", "application/json; charset=utf-8"); got != http.StatusCreated {
		t.Errorf("JSON with a charset: status %d", got)
	}
}

func TestNoCORSHeadersAreEverSent(t *testing.T) {
	f := newFixture(t)
	response := f.post(t, "/sessions", `{"policy":"run2"}`, "Origin", "https://evil.example")
	if got := response.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin = %q", got)
	}
}

func TestSessionLifecycle(t *testing.T) {
	f := newFixture(t)
	id := f.session(t)

	response := f.post(t, "/sessions/"+id+"/messages", `{"content":"2+2?"}`)
	expect(t, response, http.StatusOK)
	episode := decode[agent.Episode](t, response)
	if episode.Policy != "run2" || episode.FinishReason != agent.FinishAnswer {
		t.Fatalf("episode = %+v", episode)
	}

	found := decode[session.Session](t, f.get(t, "/sessions/"+id))
	if len(found.Episodes) != 1 || found.Running || found.Episodes[0].ID != episode.ID {
		t.Fatalf("session = %+v", found)
	}

	// The next message sees the last one.
	expect(t, f.post(t, "/sessions/"+id+"/messages", `{"content":"and 3+3?"}`), http.StatusOK)
	if len(f.runner.history) != 2 || f.runner.history[0].Content != "2+2?" {
		t.Fatalf("history = %+v", f.runner.history)
	}

	list := decode[struct {
		Sessions []session.Summary `json:"sessions"`
	}](t, f.get(t, "/sessions"))
	if len(list.Sessions) != 1 || list.Sessions[0].Episodes != 2 {
		t.Fatalf("list = %+v", list)
	}
}

func TestAPolicyTheBackendDoesNotServeIsRefused(t *testing.T) {
	f := newFixture(t)
	expect(t, f.post(t, "/sessions", `{"policy":"run9"}`), http.StatusUnprocessableEntity)
	expect(t, f.post(t, "/sessions", `{}`), http.StatusUnprocessableEntity)
	if len(f.store.List()) != 0 {
		t.Fatal("a session was created anyway")
	}
}

func TestBadRequests(t *testing.T) {
	f := newFixture(t)
	id := f.session(t)

	cases := []struct {
		path, body string
		status     int
	}{
		{"/sessions/missing/messages", `{"content":"x"}`, http.StatusNotFound},
		{"/sessions/" + id + "/messages", `{"content":"   "}`, http.StatusUnprocessableEntity},
		{"/sessions/" + id + "/messages", `not json`, http.StatusBadRequest},
		// The caller does not get to supply a system prompt or a prior transcript.
		{"/sessions/" + id + "/messages", `{"content":"x","messages":[]}`, http.StatusBadRequest},
		{"/sessions/" + id + "/messages", `{"content":"x","system":"y"}`, http.StatusBadRequest},
		{"/sessions", `{"policy":"run2","temperature":0}`, http.StatusBadRequest},
		{"/sessions/" + id + "/messages", `{"content":"` + strings.Repeat("x", maxBodyBytes) + `"}`, http.StatusRequestEntityTooLarge},
		{"/sessions/" + id + "/messages", `{"content":"` + strings.Repeat("x", 1001) + `"}`, http.StatusRequestEntityTooLarge},
	}
	for _, c := range cases {
		expect(t, f.post(t, c.path, c.body), c.status)
	}
	expect(t, f.get(t, "/sessions/missing"), http.StatusNotFound)

	// None of those should have left the session claimed.
	expect(t, f.post(t, "/sessions/"+id+"/messages", `{"content":"x"}`), http.StatusOK)
}

func TestASecondMessageWhileRunningConflicts(t *testing.T) {
	f := newFixture(t)
	f.runner.hold = make(chan struct{})
	f.runner.entered = make(chan struct{})
	id := f.session(t)

	finished := make(chan int, 1)
	go func() {
		request, _ := http.NewRequest(http.MethodPost, f.URL+"/sessions/"+id+"/messages", strings.NewReader(`{"content":"x"}`))
		request.Header.Set("Content-Type", "application/json")
		response, err := http.DefaultClient.Do(request)
		if err != nil {
			finished <- 0
			return
		}
		response.Body.Close()
		finished <- response.StatusCode
	}()
	<-f.runner.entered

	expect(t, f.post(t, "/sessions/"+id+"/messages", `{"content":"y"}`), http.StatusConflict)
	// Another session is not held up.
	other := f.session(t)
	expect(t, f.post(t, "/sessions/"+other+"/messages", `{"content":"not held"}`), http.StatusOK)

	close(f.runner.hold)
	if status := <-finished; status != http.StatusOK {
		t.Fatalf("the held episode ended with status %d", status)
	}
}

func TestASystemFailureIsAGatewayErrorAndKeepsTheRecord(t *testing.T) {
	f := newFixture(t)
	f.runner.reason = agent.FinishToolError
	id := f.session(t)

	response := f.post(t, "/sessions/"+id+"/messages", `{"content":"x"}`)
	expect(t, response, http.StatusBadGateway)
	if episode := decode[agent.Episode](t, response); episode.FinishReason != agent.FinishToolError {
		t.Fatalf("episode = %+v", episode)
	}
	found := decode[session.Session](t, f.get(t, "/sessions/"+id))
	if len(found.Episodes) != 1 || found.Running {
		t.Fatalf("session = %+v", found)
	}
}

func TestAPolicyThatFailedToAnswerIsNotAnError(t *testing.T) {
	// It worked; the policy just did not answer. That is a result.
	for _, reason := range []string{agent.FinishNoAnswer, agent.FinishEmpty, agent.FinishTurnLimit} {
		f := newFixture(t)
		f.runner.reason = reason
		expect(t, f.post(t, "/sessions/"+f.session(t)+"/messages", `{"content":"x"}`), http.StatusOK)
	}
}

type sseEvent struct {
	name string
	data agent.Event
}

func readEvents(t *testing.T, body io.Reader) []sseEvent {
	t.Helper()
	var events []sseEvent
	var name string
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			var event agent.Event
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
				t.Fatalf("data line is not one JSON value: %q: %v", line, err)
			}
			events = append(events, sseEvent{name, event})
		case line != "":
			t.Fatalf("stray line in the stream: %q", line)
		}
	}
	return events
}

func TestStreamingDeliversEventsThatSurviveNewlines(t *testing.T) {
	f := newFixture(t)
	id := f.session(t)

	response := f.post(t, "/sessions/"+id+"/messages", `{"content":"x"}`, "Accept", "text/event-stream")
	expect(t, response, http.StatusOK)
	if got := response.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content type = %q", got)
	}
	events := readEvents(t, response.Body)
	if len(events) != 3 {
		t.Fatalf("events = %+v", events)
	}
	if events[1].name != agent.EventTurn || events[1].data.Text != "line one\nline two" {
		t.Fatalf("turn event = %+v", events[1])
	}
	last := events[2]
	if last.name != agent.EventFinished || last.data.Finished == nil || last.data.Finished.FinishReason != agent.FinishAnswer {
		t.Fatalf("last event = %+v", last)
	}
	found := decode[session.Session](t, f.get(t, "/sessions/"+id))
	if len(found.Episodes) != 1 || found.Running {
		t.Fatalf("session = %+v", found)
	}
}

func TestEventsArriveBeforeTheEpisodeEnds(t *testing.T) {
	// The point of the live view. A buffered stream would pass the test above.
	f := newFixture(t)
	f.runner.hold = make(chan struct{})
	f.runner.entered = make(chan struct{})
	id := f.session(t)

	request, _ := http.NewRequest(http.MethodPost, f.URL+"/sessions/"+id+"/messages", strings.NewReader(`{"content":"x"}`))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	first := make(chan string, 1)
	lines := bufio.NewReader(response.Body)
	go func() {
		line, _ := lines.ReadString('\n')
		first <- line
	}()
	select {
	case line := <-first:
		if !strings.Contains(line, agent.EventStarted) {
			t.Fatalf("first line = %q", line)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("nothing arrived while the episode was still running")
	}
	close(f.runner.hold)
}

func TestDisconnectingCancelsTheEpisodeAndReleasesTheSession(t *testing.T) {
	f := newFixture(t)
	f.runner.hold = make(chan struct{})
	f.runner.entered = make(chan struct{})
	f.runner.left = make(chan error, 1)
	id := f.session(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, f.URL+"/sessions/"+id+"/messages", bytes.NewReader([]byte(`{"content":"x"}`)))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	<-f.runner.entered
	cancel()

	select {
	case err := <-f.runner.left:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the episode kept running after the client left")
	}

	deadline := time.Now().Add(5 * time.Second)
	for {
		found, _ := f.store.Get(id)
		if !found.Running && len(found.Episodes) == 1 {
			if found.Episodes[0].FinishReason != agent.FinishCancelled {
				t.Fatalf("episode = %+v", found.Episodes[0])
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("session never released: %+v", found)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestHealth(t *testing.T) {
	f := newFixture(t)
	response := f.get(t, "/health")
	expect(t, response, http.StatusOK)
	report := decode[struct {
		Backend string `json:"backend"`
		Tools   struct {
			WorkerRunning bool     `json:"worker_running"`
			Tools         []string `json:"tools"`
		} `json:"tools"`
	}](t, response)
	if report.Backend != "ok" || !report.Tools.WorkerRunning || report.Tools.Tools[0] != "calculator" {
		t.Fatalf("report = %+v", report)
	}

	f.backend.down = true
	expect(t, f.get(t, "/health"), http.StatusServiceUnavailable)
	expect(t, f.get(t, "/policies"), http.StatusBadGateway)
	expect(t, f.post(t, "/sessions", `{"policy":"run2"}`), http.StatusBadGateway)
}

func TestPoliciesPassesTheDescriptionThrough(t *testing.T) {
	f := newFixture(t)
	response := f.get(t, "/policies")
	expect(t, response, http.StatusOK)
	body, _ := io.ReadAll(response.Body)
	if !strings.Contains(string(body), `"run2"`) {
		t.Fatalf("body = %s", body)
	}
}
