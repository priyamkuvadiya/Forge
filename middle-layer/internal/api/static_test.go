package api

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

var page = fstest.MapFS{
	"index.html":           {Data: []byte(`<!doctype html><script type="module" src="/assets/index-abc.js"></script>`)},
	"assets/index-abc.js":  {Data: []byte(`console.log("page")`)},
	"assets/index-abc.css": {Data: []byte(`body{}`)},
}

func pageServer(t *testing.T, files fstest.MapFS) *fixture {
	t.Helper()
	f := newFixture(t)
	server := &Server{
		Backend: f.backend, Worker: f.worker, Runner: f.runner, Store: f.store, Metrics: f.metrics,
		Frontend: files,
	}
	if files == nil {
		server.Frontend = nil
	}
	f.Server.Close()
	f.Server = httptest.NewServer(server.Handler())
	t.Cleanup(f.Server.Close)
	return f
}

func body(t *testing.T, response *http.Response) string {
	t.Helper()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestThePageIsServedWithAPolicyThatKeepsItToThisServer(t *testing.T) {
	f := pageServer(t, page)

	response := f.get(t, "/")
	expect(t, response, http.StatusOK)
	if !strings.Contains(body(t, response), "index-abc.js") {
		t.Fatal("not index.html")
	}
	if got := response.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/html") {
		t.Fatalf("content type %q", got)
	}
	policy := response.Header.Get("Content-Security-Policy")
	for _, want := range []string{"default-src 'self'", "script-src 'self'", "connect-src 'self'", "frame-ancestors 'none'"} {
		if !strings.Contains(policy, want) {
			t.Fatalf("policy %q lacks %q", policy, want)
		}
	}
	if strings.Contains(strings.SplitN(strings.SplitN(policy, "script-src", 2)[1], ";", 2)[0], "unsafe") {
		t.Fatalf("scripts are allowed inline: %q", policy)
	}
	if response.Header.Get("Cache-Control") != "no-cache" {
		t.Fatal("index.html must not be cached: it names the current assets")
	}

	script := f.get(t, "/assets/index-abc.js")
	expect(t, script, http.StatusOK)
	if got := script.Header.Get("Content-Type"); !strings.Contains(got, "javascript") {
		t.Fatalf("content type %q", got)
	}
	if !strings.Contains(script.Header.Get("Cache-Control"), "immutable") {
		t.Fatal("hashed assets should be cacheable")
	}
	if script.Header.Get("X-Content-Type-Options") != "nosniff" {
		t.Fatal("no nosniff")
	}
}

func TestOnlyThePagesFilesAreServed(t *testing.T) {
	f := pageServer(t, page)
	for _, path := range []string{"/missing.js", "/assets/", "/assets/../index.html/x", "/assets/nope.js", "/main.go"} {
		if response := f.get(t, path); response.StatusCode != http.StatusNotFound {
			t.Errorf("%s: status %d", path, response.StatusCode)
		}
	}
}

func TestTheAPIIsTheSameUnderAPI(t *testing.T) {
	f := pageServer(t, page)
	created := f.post(t, "/api/sessions", `{"policy":"run2"}`)
	expect(t, created, http.StatusCreated)
	// The same server, the same store: visible at both paths.
	for _, path := range []string{"/sessions", "/api/sessions"} {
		if !strings.Contains(body(t, f.get(t, path)), `"policy":"run2"`) {
			t.Fatalf("%s does not list the session", path)
		}
	}
	expect(t, f.get(t, "/api/health"), http.StatusOK)
	expect(t, f.get(t, "/api/runs/missing"), http.StatusNotFound)
	// The guard covers /api/ too.
	request, _ := http.NewRequest(http.MethodPost, f.URL+"/api/sessions", strings.NewReader(`{"policy":"run2"}`))
	request.Header.Set("Content-Type", "text/plain")
	refused, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	refused.Body.Close()
	expect(t, refused, http.StatusUnsupportedMediaType)
}

func TestThePageIsGuardedLikeTheAPI(t *testing.T) {
	// A rebinding attack could otherwise load the page from the attacker's
	// name and have it call the API same-origin.
	f := pageServer(t, page)
	handler := (&Server{Store: f.store, Frontend: page}).Handler()
	request := httptest.NewRequest(http.MethodGet, "/", nil)
	request.Host = "evil.example"
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status %d", recorder.Code)
	}
}

func TestWithoutAPageTheRootIsNotFound(t *testing.T) {
	f := pageServer(t, nil)
	expect(t, f.get(t, "/"), http.StatusNotFound)
	expect(t, f.get(t, "/api/health"), http.StatusOK)
}
