package api

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/database"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/metrics"
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
	// afterRun runs as the episode returns, before the server records it.
	afterRun func()
}

func (r *fakeRunner) Run(ctx context.Context, id, policy string, history []agent.Message, prompt string, emit func(agent.Event)) agent.Episode {
	if emit == nil {
		emit = func(agent.Event) {}
	}
	r.history = history
	emit(agent.Event{Type: agent.EventStarted, Episode: id, Prompt: prompt})
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
			if r.left != nil {
				r.left <- ctx.Err()
			}
		}
	}
	emit(agent.Event{Type: agent.EventTurn, Episode: id, Text: "line one\nline two"})
	emit(agent.Event{Type: agent.EventFinished, Episode: id, Finished: &episode})
	if r.afterRun != nil {
		r.afterRun()
	}
	return episode
}

type fixture struct {
	*httptest.Server
	server   *Server
	shutdown context.CancelFunc
	backend  *fakeBackend
	worker   *fakeWorker
	runner   *fakeRunner
	store    *session.Store
	metrics  *metrics.Store
	db       *sql.DB
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	db, err := database.Open(filepath.Join(t.TempDir(), "forge.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	f := &fixture{
		backend: &fakeBackend{},
		worker:  &fakeWorker{alive: true},
		runner:  &fakeRunner{reason: agent.FinishAnswer},
		store:   session.NewStore(db, 1000),
		metrics: metrics.NewStore(db),
		db:      db,
	}
	ctx, shutdown := context.WithCancel(context.Background())
	f.shutdown = shutdown
	f.server = &Server{
		Backend: f.backend, Worker: f.worker, Runner: f.runner, Store: f.store, Metrics: f.metrics, Context: ctx,
	}
	f.Server = httptest.NewServer(f.server.Handler())
	t.Cleanup(f.Close)
	// Before Close, which waits for streams that follow a held episode.
	t.Cleanup(func() {
		shutdown()
		f.server.Wait()
	})
	return f
}

// stream opens an event stream that the returned function closes.
func (f *fixture) stream(t *testing.T, method, path, body string) (*bufio.Reader, func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	request, _ := http.NewRequestWithContext(ctx, method, f.URL+path, strings.NewReader(body))
	if method == http.MethodPost {
		request.Header.Set("Content-Type", "application/json")
	}
	request.Header.Set("Accept", "text/event-stream")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		cancel()
		t.Fatalf("status = %d", response.StatusCode)
	}
	return bufio.NewReader(response.Body), func() {
		cancel()
		response.Body.Close()
	}
}

// nextEvent reads one event, failing if none arrives in time.
func nextEvent(t *testing.T, reader *bufio.Reader) agent.Event {
	t.Helper()
	got := make(chan agent.Event, 1)
	failed := make(chan error, 1)
	go func() {
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				failed <- err
				return
			}
			if data, ok := strings.CutPrefix(line, "data: "); ok {
				var event agent.Event
				if err := json.Unmarshal([]byte(data), &event); err != nil {
					failed <- err
					return
				}
				got <- event
				return
			}
		}
	}()
	select {
	case event := <-got:
		return event
	case err := <-failed:
		t.Fatalf("reading the stream: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("no event arrived")
	}
	return agent.Event{}
}

func (f *fixture) hold() {
	f.runner.hold = make(chan struct{})
	f.runner.entered = make(chan struct{})
	f.runner.left = make(chan error, 1)
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
	if summaries, _ := f.store.List(); len(summaries) != 0 {
		t.Fatal("a session was created anyway")
	}
}

func TestADatabaseFailureIsAServerErrorNotAMissingSession(t *testing.T) {
	f := newFixture(t)
	id := f.session(t)
	f.db.Close()

	expect(t, f.get(t, "/sessions/"+id), http.StatusInternalServerError)
	expect(t, f.get(t, "/sessions"), http.StatusInternalServerError)
	expect(t, f.post(t, "/sessions", `{"policy":"baseline"}`), http.StatusInternalServerError)
	expect(t, f.post(t, "/sessions/"+id+"/messages", `{"content":"a"}`), http.StatusInternalServerError)
	expect(t, f.get(t, "/runs"), http.StatusInternalServerError)
	expect(t, f.get(t, "/runs/run1"), http.StatusInternalServerError)
}

func TestAnEpisodeThatCannotBeSavedStillReachesTheClient(t *testing.T) {
	f := newFixture(t)
	id := f.session(t)
	f.runner.afterRun = func() { f.db.Close() }

	response := f.post(t, "/sessions/"+id+"/messages", `{"content":"a"}`)
	expect(t, response, http.StatusOK)
	if episode := decode[agent.Episode](t, response); episode.FinishReason != agent.FinishAnswer {
		t.Fatalf("episode = %+v", episode)
	}
}

func TestRunsAreServedAsImported(t *testing.T) {
	f := newFixture(t)
	root := t.TempDir()
	directory := filepath.Join(root, "run1")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	log := `{"record": "config", "lr": 1e-05}` + "\n" +
		`{"record": "step", "step": 1, "mean_reward": 0.2446}` + "\n" +
		`{"record": "summary", "wall_clock_seconds": 12.5}` + "\n"
	if err := os.WriteFile(filepath.Join(directory, "steps.jsonl"), []byte(log), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := f.metrics.Import(root); err != nil {
		t.Fatal(err)
	}

	listed := decode[struct {
		Runs []metrics.Summary `json:"runs"`
	}](t, f.get(t, "/runs"))
	if len(listed.Runs) != 1 || listed.Runs[0].Name != "run1" || listed.Runs[0].Steps != 1 || !listed.Runs[0].Finished {
		t.Fatalf("runs = %+v", listed.Runs)
	}

	run := decode[map[string]any](t, f.get(t, "/runs/run1"))
	records := run["records"].([]any)
	if len(records) != 1 || records[0].(map[string]any)["mean_reward"] != 0.2446 {
		t.Fatalf("run = %+v", run)
	}
	if run["config"].(map[string]any)["lr"] != 1e-05 || run["wall_clock_seconds"] != 12.5 {
		t.Fatalf("run = %+v", run)
	}

	expect(t, f.get(t, "/runs/run9"), http.StatusNotFound)
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

// The refresh case: the page that started an episode goes away mid-way, and a new
// one has to find it still running and see everything it missed.
func TestDisconnectingLeavesTheEpisodeRunningAndAReconnectSeesItAll(t *testing.T) {
	for _, mode := range []string{"stream", "json"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t)
			f.hold()
			id := f.session(t)

			ctx, leave := context.WithCancel(context.Background())
			request, _ := http.NewRequestWithContext(ctx, http.MethodPost, f.URL+"/sessions/"+id+"/messages", bytes.NewReader([]byte(`{"content":"x"}`)))
			request.Header.Set("Content-Type", "application/json")
			if mode == "stream" {
				request.Header.Set("Accept", "text/event-stream")
			}
			go func() {
				if response, err := http.DefaultClient.Do(request); err == nil {
					response.Body.Close()
				}
			}()
			<-f.runner.entered
			leave()

			select {
			case err := <-f.runner.left:
				t.Fatalf("the episode stopped when the client left: %v", err)
			case <-time.After(200 * time.Millisecond):
			}
			if found, _ := f.store.Get(id); !found.Running {
				t.Fatal("the session no longer reads as running")
			}

			reader, closeStream := f.stream(t, http.MethodGet, "/sessions/"+id+"/live", "")
			defer closeStream()
			// Everything so far is replayed first.
			if first := nextEvent(t, reader); first.Type != agent.EventStarted || first.Prompt != "x" {
				t.Fatalf("first event = %+v", first)
			}
			close(f.runner.hold)
			if turn := nextEvent(t, reader); turn.Type != agent.EventTurn {
				t.Fatalf("event = %+v", turn)
			}
			last := nextEvent(t, reader)
			if last.Type != agent.EventFinished || last.Finished == nil || last.Finished.FinishReason != agent.FinishAnswer {
				t.Fatalf("last event = %+v", last)
			}
			found, _ := f.store.Get(id)
			if found.Running || len(found.Episodes) != 1 || found.Episodes[0].FinishReason != agent.FinishAnswer {
				t.Fatalf("session = %+v", found)
			}
		})
	}
}

func TestTwoPagesFollowingOneEpisodeSeeTheSameEvents(t *testing.T) {
	f := newFixture(t)
	f.hold()
	id := f.session(t)
	first, closeFirst := f.stream(t, http.MethodPost, "/sessions/"+id+"/messages", `{"content":"x"}`)
	defer closeFirst()
	<-f.runner.entered
	second, closeSecond := f.stream(t, http.MethodGet, "/sessions/"+id+"/live", "")
	defer closeSecond()
	close(f.runner.hold)

	for range 3 {
		a, b := nextEvent(t, first), nextEvent(t, second)
		if a.Type != b.Type || a.Text != b.Text {
			t.Fatalf("%+v != %+v", a, b)
		}
	}
}

func TestLiveWithNothingRunning(t *testing.T) {
	f := newFixture(t)
	id := f.session(t)
	expect(t, f.get(t, "/sessions/"+id+"/live"), http.StatusNoContent)
	expect(t, f.get(t, "/sessions/missing/live"), http.StatusNotFound)
}

func TestCancelStopsTheEpisodeAndReleasesTheSession(t *testing.T) {
	f := newFixture(t)
	f.hold()
	id := f.session(t)
	reader, closeStream := f.stream(t, http.MethodPost, "/sessions/"+id+"/messages", `{"content":"x"}`)
	defer closeStream()
	<-f.runner.entered

	// A cross-site form must not be able to cancel either.
	request, _ := http.NewRequest(http.MethodPost, f.URL+"/sessions/"+id+"/cancel", strings.NewReader(""))
	request.Header.Set("Content-Type", "text/plain")
	refused, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	refused.Body.Close()
	expect(t, refused, http.StatusUnsupportedMediaType)

	expect(t, f.post(t, "/sessions/"+id+"/cancel", `{}`), http.StatusAccepted)
	select {
	case err := <-f.runner.left:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the episode was not cancelled")
	}

	var last agent.Event
	for last.Type != agent.EventFinished {
		last = nextEvent(t, reader)
	}
	if last.Finished.FinishReason != agent.FinishCancelled {
		t.Fatalf("last = %+v", last)
	}
	found, _ := f.store.Get(id)
	if found.Running || found.Episodes[0].FinishReason != agent.FinishCancelled {
		t.Fatalf("session = %+v", found)
	}
	expect(t, f.post(t, "/sessions/"+id+"/cancel", `{}`), http.StatusConflict)
	expect(t, f.post(t, "/sessions/missing/cancel", `{}`), http.StatusNotFound)
}

func TestTheFinalEventIsSentOnlyOnceTheEpisodeIsSaved(t *testing.T) {
	// A client that reloads the session on seeing the end must find it there.
	// The runner emits its final event, then dawdles before returning.
	f := newFixture(t)
	f.runner.afterRun = func() { time.Sleep(150 * time.Millisecond) }
	id := f.session(t)
	reader, closeStream := f.stream(t, http.MethodPost, "/sessions/"+id+"/messages", `{"content":"x"}`)
	defer closeStream()

	for event := nextEvent(t, reader); event.Type != agent.EventFinished; event = nextEvent(t, reader) {
	}
	found := decode[session.Session](t, f.get(t, "/sessions/"+id))
	if found.Running || len(found.Episodes) != 1 {
		t.Fatalf("session = %+v", found)
	}
}

func TestEachEventIsCheckpointedAsItHappens(t *testing.T) {
	f := newFixture(t)
	f.hold()
	id := f.session(t)
	_, closeStream := f.stream(t, http.MethodPost, "/sessions/"+id+"/messages", `{"content":"x"}`)
	defer closeStream()
	<-f.runner.entered

	var events sql.NullString
	if err := f.db.QueryRow("SELECT events FROM episodes").Scan(&events); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(events.String, agent.EventStarted) {
		t.Fatalf("checkpoint = %v", events)
	}
	close(f.runner.hold)
}

func TestShutdownCancelsRunningEpisodesAndRecordsThem(t *testing.T) {
	f := newFixture(t)
	f.hold()
	id := f.session(t)
	go func() {
		if response, err := http.DefaultClient.Do(mustRequest(f.URL+"/sessions/"+id+"/messages", `{"content":"x"}`)); err == nil {
			response.Body.Close()
		}
	}()
	<-f.runner.entered

	f.shutdown()
	waited := make(chan struct{})
	go func() {
		f.server.Wait()
		close(waited)
	}()
	select {
	case <-waited:
	case <-time.After(5 * time.Second):
		t.Fatal("Wait did not return after shutdown")
	}
	found, _ := f.store.Get(id)
	if found.Running || len(found.Episodes) != 1 || found.Episodes[0].FinishReason != agent.FinishCancelled {
		t.Fatalf("session = %+v", found)
	}
}

func mustRequest(url, body string) *http.Request {
	request, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		panic(err)
	}
	request.Header.Set("Content-Type", "application/json")
	return request
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
