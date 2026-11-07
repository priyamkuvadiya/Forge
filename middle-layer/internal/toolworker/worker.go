// Package toolworker supervises `python -m tool_worker`, the process that runs
// the tools, and speaks its line-delimited JSON protocol. See tool_worker.py for
// why the tools stay in Python.
//
// The one rule here: the worker failing must never look like a tool failing. A
// tool that fails is ordinary policy behaviour and goes back to the policy as a
// result. A worker that dies is the system's failure, and telling the policy
// "the tool failed" would make it behave differently for a reason it had nothing
// to do with. So every failure in this package is a Go error, which ends the
// episode with `tool_error`.
package toolworker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
)

// Sandbox says what the code tool's confinement is, as reported by the worker.
type Sandbox struct {
	Backend            *string `json:"backend"`
	Confinement        *string `json:"confinement"`
	FilesystemConfined *bool   `json:"filesystem_confined"`
	AllowUnconfined    bool    `json:"allow_unconfined"`
}

// Info is the worker's handshake.
type Info struct {
	Tools              []string `json:"tools"`
	MaxCalls           int      `json:"max_calls"`
	MaxToolOutputChars int      `json:"max_tool_output_chars"`
	Sandbox            Sandbox  `json:"sandbox"`
}

type Config struct {
	Command []string
	Dir     string
	Env     []string
	Stderr  io.Writer
	// StartTimeout includes the sandbox preflight, and on a fresh Windows machine
	// the sandbox's ~90 second first-time setup.
	StartTimeout time.Duration
	// CallTimeout bounds one call. The sandbox stops a program after seconds on
	// its own, so reaching this means the worker itself is stuck, and it is killed.
	CallTimeout time.Duration
}

// ErrDied is wrapped by every error caused by the process going away.
var ErrDied = errors.New("tool worker died")

type Worker struct {
	config Config
	nextID atomic.Int64

	mu      sync.Mutex
	current *process
	info    *Info
	stopped bool
}

type reply struct {
	data json.RawMessage
}

type process struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser

	writeMu sync.Mutex

	pendingMu sync.Mutex
	pending   map[int64]chan reply

	dead     chan struct{}
	dieOnce  sync.Once
	deathErr error
}

// Start launches the worker and waits for its handshake.
func Start(ctx context.Context, config Config) (*Worker, error) {
	worker := &Worker{config: config}
	if _, err := worker.ensure(ctx); err != nil {
		return nil, err
	}
	return worker, nil
}

// Info returns the handshake of the first process started.
func (w *Worker) Info() Info {
	w.mu.Lock()
	defer w.mu.Unlock()
	return *w.info
}

// Alive reports whether a process is running right now. A dead worker is not
// fatal, since the next episode starts another.
func (w *Worker) Alive() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.current != nil && !w.current.isDead()
}

// ensure returns the running process, starting one if needed. Only opening an
// episode gets to do this. Mid-episode the budget lived in the process that
// died, so those calls have to fail rather than continue on a fresh one.
func (w *Worker) ensure(ctx context.Context) (*process, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped {
		return nil, fmt.Errorf("%w: stopped", ErrDied)
	}
	if w.current != nil && !w.current.isDead() {
		return w.current, nil
	}

	started, info, err := launch(ctx, w.config)
	if err != nil {
		return nil, err
	}
	if w.info != nil && !reflect.DeepEqual(*w.info, info) {
		// Everything agreed at startup was agreed about the old process.
		started.die(errors.New("replaced"))
		return nil, fmt.Errorf("restarted tool worker differs: was %+v, now %+v", *w.info, info)
	}
	w.info = &info
	w.current = started
	return started, nil
}

func launch(ctx context.Context, config Config) (*process, Info, error) {
	if len(config.Command) == 0 {
		return nil, Info{}, errors.New("no tool worker command")
	}
	// Not CommandContext: this process outlives the request that started it.
	cmd := exec.Command(config.Command[0], config.Command[1:]...)
	cmd.Dir = config.Dir
	cmd.Env = config.Env
	cmd.Stderr = config.Stderr
	cmd.WaitDelay = 2 * time.Second

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, Info{}, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, Info{}, err
	}
	if err := cmd.Start(); err != nil {
		return nil, Info{}, fmt.Errorf("start tool worker: %w", err)
	}

	started := &process{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		pending: map[int64]chan reply{},
		dead:    make(chan struct{}),
	}

	lines := bufio.NewReader(stdout)
	type handshake struct {
		info Info
		err  error
	}
	greeted := make(chan handshake, 1)
	go func() {
		line, err := lines.ReadBytes('\n')
		if err != nil {
			greeted <- handshake{err: fmt.Errorf("%w before its handshake: %v", ErrDied, err)}
			return
		}
		var parsed struct {
			Ready bool `json:"ready"`
			Info
		}
		if err := json.Unmarshal(line, &parsed); err != nil || !parsed.Ready {
			greeted <- handshake{err: fmt.Errorf("bad tool worker handshake: %q", line)}
			return
		}
		greeted <- handshake{info: parsed.Info}
	}()

	timer := time.NewTimer(config.StartTimeout)
	defer timer.Stop()
	select {
	case result := <-greeted:
		if result.err != nil {
			started.die(result.err)
			go started.reap()
			return nil, Info{}, result.err
		}
		go started.readLoop(lines)
		go started.reap()
		return started, result.info, nil
	case <-timer.C:
		err := fmt.Errorf("tool worker gave no handshake within %s", config.StartTimeout)
		started.die(err)
		go started.reap()
		return nil, Info{}, err
	case <-ctx.Done():
		started.die(ctx.Err())
		go started.reap()
		return nil, Info{}, ctx.Err()
	}
}

func (p *process) isDead() bool {
	select {
	case <-p.dead:
		return true
	default:
		return false
	}
}

// die marks the process finished and kills it. The stdout pipe is closed too, so
// the read loop cannot be held open by a descendant keeping the handle.
func (p *process) die(reason error) {
	p.dieOnce.Do(func() {
		p.deathErr = reason
		close(p.dead)
		_ = p.cmd.Process.Kill()
		_ = p.stdout.Close()
		_ = p.stdin.Close()
	})
}

func (p *process) reap() {
	<-p.dead
	_ = p.cmd.Wait()
}

func (p *process) readLoop(lines *bufio.Reader) {
	for {
		line, err := lines.ReadBytes('\n')
		if err != nil {
			p.die(fmt.Errorf("%w: %v", ErrDied, err))
			return
		}
		var envelope struct {
			ID *int64 `json:"id"`
		}
		if err := json.Unmarshal(line, &envelope); err != nil {
			// Once one line is unreadable, no later line can be trusted either.
			p.die(fmt.Errorf("%w: unreadable output %q", ErrDied, line))
			return
		}
		if envelope.ID == nil {
			// The worker could not parse something we sent, which is a bug here.
			p.die(fmt.Errorf("%w: it rejected a request: %s", ErrDied, line))
			return
		}
		p.pendingMu.Lock()
		waiter, found := p.pending[*envelope.ID]
		delete(p.pending, *envelope.ID)
		p.pendingMu.Unlock()
		// Not found means the caller gave up, which is normal.
		if found {
			waiter <- reply{data: line}
		}
	}
}

// call sends one request and decodes the reply into out.
func (w *Worker) call(
	ctx context.Context,
	target *process,
	request map[string]any,
	out any,
) error {
	id := w.nextID.Add(1)
	request["id"] = id
	line, err := json.Marshal(request)
	if err != nil {
		return err
	}

	waiter := make(chan reply, 1)
	target.pendingMu.Lock()
	target.pending[id] = waiter
	target.pendingMu.Unlock()
	defer func() {
		target.pendingMu.Lock()
		delete(target.pending, id)
		target.pendingMu.Unlock()
	}()

	// Armed before the write, which can itself block on a stuck worker. Dying
	// closes the pipe, and that releases the write.
	timeout := w.config.CallTimeout
	stuck := time.AfterFunc(timeout, func() {
		target.die(fmt.Errorf("%w: killed after a call took over %s", ErrDied, timeout))
	})
	defer stuck.Stop()

	target.writeMu.Lock()
	_, err = target.stdin.Write(append(line, '\n'))
	target.writeMu.Unlock()
	if err != nil {
		select {
		case <-target.dead:
			return target.deathErr
		default:
		}
		target.die(fmt.Errorf("%w: %v", ErrDied, err))
		return target.deathErr
	}

	select {
	case answer := <-waiter:
		var status struct {
			OK    bool   `json:"ok"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(answer.data, &status); err != nil {
			return fmt.Errorf("tool worker: bad reply: %w", err)
		}
		if !status.OK {
			return fmt.Errorf("tool worker: %s", status.Error)
		}
		if err := json.Unmarshal(answer.data, out); err != nil {
			return fmt.Errorf("tool worker: bad reply: %w", err)
		}
		return nil
	case <-target.dead:
		return target.deathErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (w *Worker) running() (*process, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.current == nil || w.current.isDead() {
		return nil, fmt.Errorf("%w: no running process", ErrDied)
	}
	return w.current, nil
}

// Open starts an episode's budget, restarting the worker if it has died.
func (w *Worker) Open(ctx context.Context, episode string, maxCalls int) error {
	target, err := w.ensure(ctx)
	if err != nil {
		return err
	}
	var ignored struct{}
	return w.call(ctx, target, map[string]any{
		"op": "open", "episode": episode, "max_calls": maxCalls,
	}, &ignored)
}

// Invoke runs one tool call against the episode's budget.
func (w *Worker) Invoke(ctx context.Context, episode, name, arguments string) (agent.Invocation, error) {
	target, err := w.running()
	if err != nil {
		return agent.Invocation{}, err
	}
	var result agent.Invocation
	err = w.call(ctx, target, map[string]any{
		"op": "invoke", "episode": episode, "name": name, "arguments": arguments,
	}, &result)
	return result, err
}

// Finish closes the episode and returns its answer and trace.
func (w *Worker) Finish(ctx context.Context, episode, response string) (agent.Finished, error) {
	target, err := w.running()
	if err != nil {
		return agent.Finished{}, err
	}
	var result agent.Finished
	err = w.call(ctx, target, map[string]any{
		"op": "finish", "episode": episode, "response": response,
	}, &result)
	return result, err
}

// Stop closes stdin, which is the worker's signal to exit, and kills it if it
// has not done so in time. No new process starts after this.
func (w *Worker) Stop(wait time.Duration) {
	w.mu.Lock()
	w.stopped = true
	current := w.current
	w.mu.Unlock()
	if current == nil {
		return
	}
	current.writeMu.Lock()
	_ = current.stdin.Close()
	current.writeMu.Unlock()

	select {
	case <-current.dead:
	case <-time.After(wait):
		current.die(errors.New("did not stop in time"))
	}
}
