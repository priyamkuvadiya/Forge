package backend

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
)

func serve(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return New(server.URL+"/", time.Minute)
}

func reply(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, body) }
}

var ask = []agent.Message{{Role: "user", Content: "2+2?"}}

func TestTurnSendsThePolicyAndConversationAndReadsTheReply(t *testing.T) {
	var got struct {
		Policy   string          `json:"policy"`
		Messages []agent.Message `json:"messages"`
	}
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/turn" {
			t.Errorf("%s %s", r.Method, r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = io.WriteString(w, `{"policy":"run2","kind":"tool_call","text":"<tool name=\"search\">q</tool>",
			"tool_call":{"name":"search","arguments":"q"},"answer":null,"raw":"x","latency_ms":12.5}`)
	})

	turn, err := client.Turn(context.Background(), "run2", ask)
	if err != nil {
		t.Fatal(err)
	}
	if got.Policy != "run2" || !reflect.DeepEqual(got.Messages, ask) {
		t.Fatalf("sent %+v", got)
	}
	if turn.Kind != agent.KindToolCall || turn.ToolCall.Name != "search" || turn.LatencyMS != 12.5 {
		t.Fatalf("turn = %+v", turn)
	}
}

func TestAMalformedTurnIsAnErrorNotAPolicyDecision(t *testing.T) {
	cases := map[string]string{
		"unknown kind":      `{"kind":"maybe","text":""}`,
		"call with no call": `{"kind":"tool_call","text":"x","tool_call":null}`,
		"not JSON":          `<html>bad gateway</html>`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := serve(t, reply(body)).Turn(context.Background(), "p", ask); err == nil {
				t.Fatal("no error")
			}
		})
	}
}

func TestAnOverlongReplyIsRefused(t *testing.T) {
	client := serve(t, reply(`{"kind":"answer","text":"`+strings.Repeat("x", maxReplyBytes+1)+`"}`))
	_, err := client.Turn(context.Background(), "p", ask)
	if err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Fatalf("err = %v", err)
	}
}

func TestBackendErrorsCarryTheirDetailAndAreNotRetried(t *testing.T) {
	var calls atomic.Int32
	client := serve(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"detail":"no policy named 'x'"}`)
	})
	_, err := client.Turn(context.Background(), "x", ask)
	if err == nil || !strings.Contains(err.Error(), "no policy named 'x'") {
		t.Fatalf("err = %v", err)
	}
	// A retry would draw a different completion and hide that it had happened.
	if calls.Load() != 1 {
		t.Fatalf("called %d times", calls.Load())
	}
}

func TestATurnTimeoutIsAnErrorAndNotACancellation(t *testing.T) {
	release := make(chan struct{})
	client := serve(t, func(_ http.ResponseWriter, _ *http.Request) { <-release })
	defer close(release)
	client.TurnTimeout = 50 * time.Millisecond

	_, err := client.Turn(context.Background(), "p", ask)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v", err)
	}
	// The loop reports cancellation only for Canceled, so a generation that hit
	// its limit is reported as an outage rather than as the user leaving.
	if errors.Is(err, context.Canceled) {
		t.Fatalf("a timeout looked like a cancellation: %v", err)
	}
}

func TestCancellingTheRequestReachesTheBackendCall(t *testing.T) {
	release := make(chan struct{})
	client := serve(t, func(_ http.ResponseWriter, _ *http.Request) { <-release })
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)
	if _, err := client.Turn(ctx, "p", ask); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}

func TestPoliciesReturnsTheDescriptionAndTheNames(t *testing.T) {
	body := `{"model":"m","policies":[{"name":"baseline","adapter":null},{"name":"run2","adapter":{"sha256":"ab"}}]}`
	raw, names, err := serve(t, reply(body)).Policies(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(names, []string{"baseline", "run2"}) {
		t.Fatalf("names = %v", names)
	}
	if string(raw) != body {
		t.Fatalf("raw = %s", raw)
	}
}

func TestContractDecodesWhatTheBackendPublishes(t *testing.T) {
	body := `{"tools":["calculator"],"max_calls":8,"max_nudges":1,"nudge_message":"n","max_tool_output_chars":4000,"stop":[]}`
	contract, err := serve(t, reply(body)).Contract(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := Contract{Tools: []string{"calculator"}, MaxCalls: 8, MaxNudges: 1, NudgeMessage: "n", MaxToolOutputChars: 4000}
	if !reflect.DeepEqual(contract, want) {
		t.Fatalf("contract = %+v", contract)
	}
}

func TestWaitReadyPollsUntilTheBackendAnswers(t *testing.T) {
	var calls atomic.Int32
	client := serve(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(w, `{"status":"ok"}`)
	})
	if err := client.WaitReady(context.Background(), 30*time.Second); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("called %d times", calls.Load())
	}
}

func TestWaitReadyGivesUpAndSaysWhy(t *testing.T) {
	client := serve(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	err := client.WaitReady(context.Background(), 50*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "not ready") {
		t.Fatalf("err = %v", err)
	}
}
