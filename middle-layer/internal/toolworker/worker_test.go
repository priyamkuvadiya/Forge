package toolworker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// The worker's failure modes are about processes, so these use real ones. The
// test binary re-executes itself as a fake worker that misbehaves on request,
// which is how crashes and hangs can be produced at exact moments without Python.
// The real worker is exercised by tests/test_middle_layer_parity.py.

const modeVariable = "FORGE_FAKE_WORKER"

func TestMain(m *testing.M) {
	if mode := os.Getenv(modeVariable); mode != "" {
		fakeWorker(mode)
		return
	}
	os.Exit(m.Run())
}

// fakeWorker speaks the protocol. The tool name picks the behaviour.
func fakeWorker(mode string) {
	out := bufio.NewWriter(os.Stdout)
	var writeMu sync.Mutex
	send := func(value any) {
		data, _ := json.Marshal(value)
		writeMu.Lock()
		defer writeMu.Unlock()
		_, _ = out.Write(append(data, '\n'))
		_ = out.Flush()
	}

	switch mode {
	case "silent":
		time.Sleep(time.Minute)
		return
	case "exit-early":
		fmt.Fprintln(os.Stderr, "refusing")
		os.Exit(1)
	case "bad-handshake":
		_, _ = out.WriteString("this is not a handshake\n")
		_ = out.Flush()
		time.Sleep(time.Minute)
		return
	}

	tools := []string{"echo"}
	// Reports different tools from its second life onwards.
	if counter := os.Getenv("FORGE_FAKE_COUNTER"); counter != "" {
		if _, err := os.Stat(counter); err == nil {
			tools = []string{"echo", "surprise"}
		}
		_ = os.WriteFile(counter, []byte("x"), 0o600)
	}
	send(map[string]any{"ready": true, "tools": tools, "max_calls": 8, "max_tool_output_chars": 4000, "sandbox": map[string]any{"allow_unconfined": false}})

	open := map[string]bool{}
	var openMu sync.Mutex
	input := bufio.NewReader(os.Stdin)
	for {
		line, err := input.ReadBytes('\n')
		if err != nil {
			os.Exit(0)
		}
		var request struct {
			ID        int64  `json:"id"`
			Op        string `json:"op"`
			Episode   string `json:"episode"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			Response  string `json:"response"`
		}
		_ = json.Unmarshal(line, &request)

		openMu.Lock()
		known := open[request.Episode]
		if request.Op == "open" {
			open[request.Episode] = true
		}
		if request.Op == "finish" {
			delete(open, request.Episode)
		}
		openMu.Unlock()

		switch {
		case request.Op == "open":
			send(map[string]any{"id": request.ID, "ok": true})
		case !known:
			send(map[string]any{"id": request.ID, "ok": false, "error": "unknown episode"})
		case request.Op == "finish":
			send(map[string]any{"id": request.ID, "ok": true, "answer": request.Response, "trace": []any{}})
		case request.Name == "crash":
			os.Exit(3)
		case request.Name == "hang":
			time.Sleep(time.Minute)
		case request.Name == "garbage":
			_, _ = out.WriteString("{corrupt\n")
			_ = out.Flush()
		case request.Name == "reject":
			send(map[string]any{"id": nil, "ok": false, "error": "malformed request"})
		case request.Name == "late":
			go func(id int64) {
				time.Sleep(200 * time.Millisecond)
				send(map[string]any{"id": id, "ok": true, "tool": "late", "tool_ok": true, "message": "late", "output": "late"})
			}(request.ID)
		default:
			send(map[string]any{
				"id": request.ID, "ok": true, "tool": request.Name, "tool_ok": true,
				"output": request.Arguments, "message": "<" + request.Arguments + ">",
				"trace": map[string]any{"index": 0, "tool": request.Name, "arguments": request.Arguments, "ok": true, "output": request.Arguments, "duration_ms": 1.5},
			})
		}
	}
}

func config(t *testing.T, mode string, env ...string) Config {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return Config{
		Command:      []string{executable},
		Env:          append(append(os.Environ(), modeVariable+"="+mode), env...),
		StartTimeout: 20 * time.Second,
		CallTimeout:  20 * time.Second,
	}
}

func start(t *testing.T, c Config) *Worker {
	t.Helper()
	worker, err := Start(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { worker.Stop(time.Second) })
	return worker
}

func open(t *testing.T, worker *Worker, episode string) {
	t.Helper()
	if err := worker.Open(context.Background(), episode, 8); err != nil {
		t.Fatal(err)
	}
}

func TestRoundTrip(t *testing.T) {
	worker := start(t, config(t, "normal"))
	ctx := context.Background()
	if info := worker.Info(); info.MaxCalls != 8 || info.Tools[0] != "echo" {
		t.Fatalf("info = %+v", info)
	}
	open(t, worker, "e")

	result, err := worker.Invoke(ctx, "e", "echo", "héllo ✓")
	if err != nil {
		t.Fatal(err)
	}
	if result.Message != "<héllo ✓>" || !result.OK || result.Trace == nil || result.Trace.DurationMS != 1.5 {
		t.Fatalf("result = %+v", result)
	}

	finished, err := worker.Finish(ctx, "e", "the answer")
	if err != nil {
		t.Fatal(err)
	}
	if finished.Answer == nil || *finished.Answer != "the answer" {
		t.Fatalf("finished = %+v", finished)
	}
}

func TestARefusedRequestIsAnErrorAndTheWorkerSurvives(t *testing.T) {
	worker := start(t, config(t, "normal"))
	_, err := worker.Invoke(context.Background(), "never-opened", "echo", "x")
	if err == nil || !strings.Contains(err.Error(), "unknown episode") {
		t.Fatalf("err = %v", err)
	}
	if errors.Is(err, ErrDied) || !worker.Alive() {
		t.Fatalf("a refused request was treated as a death: %v", err)
	}
}

func TestACrashFailsTheCallAndTheEpisodeCannotContinueOnANewProcess(t *testing.T) {
	worker := start(t, config(t, "normal"))
	ctx := context.Background()
	open(t, worker, "e")

	_, err := worker.Invoke(ctx, "e", "crash", "")
	if !errors.Is(err, ErrDied) {
		t.Fatalf("err = %v", err)
	}
	if worker.Alive() {
		t.Fatal("still reported alive")
	}

	// Before any restart, the episode has nothing to call.
	if _, err := worker.Invoke(ctx, "e", "echo", "x"); !errors.Is(err, ErrDied) {
		t.Fatalf("err = %v", err)
	}

	// The next episode gets a new process.
	open(t, worker, "next")
	if !worker.Alive() {
		t.Fatal("not restarted")
	}

	// The old episode's budget died with the old process. It must be refused
	// here, or its budget would have been silently refilled.
	_, err = worker.Invoke(ctx, "e", "echo", "x")
	if err == nil || !strings.Contains(err.Error(), "unknown episode") {
		t.Fatalf("the old episode continued on the new process: %v", err)
	}
	if _, err := worker.Invoke(ctx, "next", "echo", "x"); err != nil {
		t.Fatal(err)
	}
}

func TestAHungWorkerIsKilledAndTheCallFails(t *testing.T) {
	c := config(t, "normal")
	c.CallTimeout = 300 * time.Millisecond
	worker := start(t, c)
	open(t, worker, "e")

	began := time.Now()
	_, err := worker.Invoke(context.Background(), "e", "hang", "")
	if !errors.Is(err, ErrDied) || !strings.Contains(err.Error(), "killed") {
		t.Fatalf("err = %v", err)
	}
	if elapsed := time.Since(began); elapsed > 10*time.Second {
		t.Fatalf("took %s", elapsed)
	}
	if worker.Alive() {
		t.Fatal("a hung worker was left running")
	}
}

func TestCorruptOutputKillsTheWorker(t *testing.T) {
	// The reason matters: without it, the call timeout would also kill the worker
	// and this would pass with the check removed, just 20 seconds later.
	for tool, reason := range map[string]string{"garbage": "unreadable output", "reject": "rejected a request"} {
		t.Run(tool, func(t *testing.T) {
			worker := start(t, config(t, "normal"))
			open(t, worker, "e")
			_, err := worker.Invoke(context.Background(), "e", tool, "")
			if !errors.Is(err, ErrDied) || !strings.Contains(err.Error(), reason) {
				t.Fatalf("err = %v", err)
			}
			if worker.Alive() {
				t.Fatal("still running after corrupt output")
			}
		})
	}
}

func TestLeavingACallDoesNotKillTheWorker(t *testing.T) {
	worker := start(t, config(t, "normal"))
	open(t, worker, "e")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := worker.Invoke(ctx, "e", "late", ""); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v", err)
	}

	// The abandoned reply arrives with no one waiting, and must not break the loop
	// or be delivered to whoever calls next.
	time.Sleep(300 * time.Millisecond)
	if !worker.Alive() {
		t.Fatal("abandoning a call killed the worker")
	}
	result, err := worker.Invoke(context.Background(), "e", "echo", "mine")
	if err != nil || result.Output != "mine" {
		t.Fatalf("result = %+v, err = %v", result, err)
	}
}

func TestRepliesGoToTheirOwnCallers(t *testing.T) {
	worker := start(t, config(t, "normal"))
	open(t, worker, "e")

	var group sync.WaitGroup
	errs := make(chan error, 64)
	for i := range 64 {
		group.Add(1)
		go func() {
			defer group.Done()
			want := fmt.Sprintf("value-%d", i)
			result, err := worker.Invoke(context.Background(), "e", "echo", want)
			if err != nil {
				errs <- err
			} else if result.Output != want {
				errs <- fmt.Errorf("asked for %s, got %s", want, result.Output)
			}
		}()
	}
	group.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestStartFailures(t *testing.T) {
	t.Run("the worker refuses to start", func(t *testing.T) {
		_, err := Start(context.Background(), config(t, "exit-early"))
		if !errors.Is(err, ErrDied) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("a bad handshake", func(t *testing.T) {
		_, err := Start(context.Background(), config(t, "bad-handshake"))
		if err == nil || !strings.Contains(err.Error(), "handshake") {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("no handshake in time", func(t *testing.T) {
		c := config(t, "silent")
		c.StartTimeout = 100 * time.Millisecond
		_, err := Start(context.Background(), c)
		if err == nil || !strings.Contains(err.Error(), "no handshake") {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("no such program", func(t *testing.T) {
		c := config(t, "normal")
		c.Command = []string{filepath.Join(t.TempDir(), "missing")}
		if _, err := Start(context.Background(), c); err == nil {
			t.Fatal("no error")
		}
	})
}

func TestARestartThatServesDifferentToolsIsRefused(t *testing.T) {
	counter := filepath.Join(t.TempDir(), "counter")
	worker := start(t, config(t, "normal", "FORGE_FAKE_COUNTER="+counter))
	open(t, worker, "e")
	if _, err := worker.Invoke(context.Background(), "e", "crash", ""); !errors.Is(err, ErrDied) {
		t.Fatalf("err = %v", err)
	}

	// The startup check against the backend was about the old process.
	err := worker.Open(context.Background(), "next", 8)
	if err == nil || !strings.Contains(err.Error(), "differs") {
		t.Fatalf("err = %v", err)
	}
}

func TestStopEndsTheWorkerAndNothingRestartsIt(t *testing.T) {
	worker, err := Start(context.Background(), config(t, "normal"))
	if err != nil {
		t.Fatal(err)
	}
	worker.Stop(10 * time.Second)
	if worker.Alive() {
		t.Fatal("still running")
	}
	if err := worker.Open(context.Background(), "e", 8); !errors.Is(err, ErrDied) {
		t.Fatalf("err = %v", err)
	}
}
