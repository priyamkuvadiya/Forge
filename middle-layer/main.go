// The Forge middle layer: sessions, routing and tool execution between the
// frontend and the policy backend.
//
//	go run . --backend http://127.0.0.1:8000
//
// Run from middle-layer/, or point --repo at the repository root, which is where
// it finds `python -m tool_worker`.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"syscall"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/api"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/backend"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/database"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/metrics"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/session"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/toolworker"
)

// maxConversationChars limits a session's conversation when a new message
// arrives. Chosen, not measured for this purpose: about 8-10k tokens, which is
// the size a single full-budget episode (8 calls x the 4000-character tool cap)
// can already reach under evaluation. The card fits far more than that at batch 1
// (72k tokens at batch 48 stayed on it), but nothing else bounds what a long chat
// would otherwise send, and exceeding VRAM on this machine does not raise an
// error, it only runs up to 11x slower.
const maxConversationChars = 32000

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	var (
		addr            = flag.String("addr", "127.0.0.1:8080", "address to serve on; must be loopback")
		backendURL      = flag.String("backend", "http://127.0.0.1:8000", "the policy backend")
		repo            = flag.String("repo", "..", "the repository root, for tool_worker")
		python          = flag.String("python", "", "Python to run tool_worker with (default: the repo's .venv)")
		allowUnconfined = flag.Bool("allow-unconfined", false, "run generated code on a sandbox that does not confine the filesystem")
		backendWait     = flag.Duration("backend-wait", 5*time.Minute, "how long to wait for the backend at startup")
		turnTimeout     = flag.Duration("turn-timeout", 5*time.Minute, "limit on one generation, including queueing for the GPU")
		toolTimeout     = flag.Duration("tool-timeout", time.Minute, "limit on one tool call before the worker is killed")
		dbPath          = flag.String("db", "", "the SQLite file (default: middle-layer/forge.db in the repo)")
		metricsRoot     = flag.String("metrics", "", "directory of training runs to import (default: artifacts/rl in the repo; \"none\" imports nothing)")
		frontendDir     = flag.String("frontend", "", "the built page to serve at / (default: frontend/dist in the repo; \"none\" serves none)")
	)
	flag.Parse()

	if err := requireLoopback(*addr); err != nil {
		return err
	}
	root, err := filepath.Abs(*repo)
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(root, "tool_worker.py")); err != nil {
		return fmt.Errorf("no tool_worker.py in %s; set --repo", root)
	}

	// First, so that a second copy is refused before it starts anything.
	if *dbPath == "" {
		*dbPath = filepath.Join(root, "middle-layer", "forge.db")
	}
	db, err := database.Open(*dbPath)
	if err != nil {
		return err
	}
	defer db.Close()
	log.Printf("database: %s", *dbPath)

	runs := metrics.NewStore(db)
	if *metricsRoot == "" {
		*metricsRoot = filepath.Join(root, "artifacts", "rl")
	}
	if *metricsRoot != "none" {
		importRuns(runs, *metricsRoot)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	command := []string{pythonPath(root, *python), "-u", "-m", "tool_worker"}
	if *allowUnconfined {
		command = append(command, "--allow-unconfined")
	}
	worker, err := toolworker.Start(ctx, toolworker.Config{
		Command:      command,
		Dir:          root,
		Stderr:       os.Stderr,
		StartTimeout: 3 * time.Minute,
		CallTimeout:  *toolTimeout,
	})
	if err != nil {
		return fmt.Errorf("tool worker: %w", err)
	}
	defer worker.Stop(5 * time.Second)

	client := backend.New(*backendURL, *turnTimeout)
	log.Printf("waiting for the backend at %s", *backendURL)
	if err := client.WaitReady(ctx, *backendWait); err != nil {
		return err
	}
	contract, err := client.Contract(ctx)
	if err != nil {
		return err
	}
	if err := checkAgreement(contract, worker.Info()); err != nil {
		return err
	}

	page, err := frontend(root, *frontendDir)
	if err != nil {
		return err
	}

	server := &api.Server{
		Context:  ctx,
		Frontend: page,
		Backend:  client,
		Worker:   worker,
		Store:    session.NewStore(db, maxConversationChars),
		Metrics:  runs,
		Runner: agent.Runner{
			Policy: client,
			Tools:  worker,
			Limits: agent.Limits{
				MaxCalls:     contract.MaxCalls,
				MaxNudges:    contract.MaxNudges,
				NudgeMessage: contract.NudgeMessage,
			},
		},
	}
	// Runs before the worker stops and the database closes (defers run last
	// first): cancel what is running, then let each episode record how it ended.
	defer func() {
		stop()
		server.Wait()
	}()

	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	httpServer := &http.Server{
		Handler:           server.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: an episode takes as long as its turns take, and
		// --turn-timeout and --tool-timeout are what bound those.
		BaseContext: func(net.Listener) context.Context { return ctx },
	}

	info := worker.Info()
	log.Printf("serving on http://%s, tools %v", listener.Addr(), info.Tools)
	if info.Sandbox.Confinement != nil {
		log.Printf("sandbox: %s", *info.Sandbox.Confinement)
	}

	served := make(chan error, 1)
	go func() { served <- httpServer.Serve(listener) }()

	select {
	case err := <-served:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdown)
	}
	return nil
}

// importRuns reports problems and carries on: a damaged training log should not
// stop anyone from chatting with a policy.
func importRuns(runs *metrics.Store, root string) {
	report, err := runs.Import(root)
	if err != nil {
		log.Printf("training runs: %v", err)
		return
	}
	log.Printf("training runs: %d imported, %d unchanged, %d failed",
		len(report.Imported), len(report.Unchanged), len(report.Failed))
	for name, failure := range report.Failed {
		log.Printf("training run %s: %v", name, failure)
	}
}

// frontend finds the built page. Without one the API still serves: the page
// can come from `npm run dev` instead, or not be wanted at all. A directory
// given explicitly has to be there, since asking for one and silently getting
// none would look like a broken page.
func frontend(root, dir string) (fs.FS, error) {
	switch dir {
	case "none":
		return nil, nil
	case "":
		dir = filepath.Join(root, "frontend", "dist")
		if _, err := os.Stat(filepath.Join(dir, "index.html")); err != nil {
			log.Printf("no built page at %s (npm run build in frontend/); serving the API only", dir)
			return nil, nil
		}
	default:
		if _, err := os.Stat(filepath.Join(dir, "index.html")); err != nil {
			return nil, fmt.Errorf("--frontend %s: no index.html there", dir)
		}
	}
	log.Printf("page: %s", dir)
	return os.DirFS(dir), nil
}

// requireLoopback refuses any other address: this server has no authentication,
// and a request to it can end with generated code running.
func requireLoopback(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return err
	}
	if host == "localhost" {
		return nil
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return nil
	}
	return fmt.Errorf("refusing to listen on %q: this server has no authentication, use a loopback address", addr)
}

func pythonPath(root, explicit string) string {
	if explicit != "" {
		return explicit
	}
	venv := filepath.Join(root, ".venv", "bin", "python")
	if runtime.GOOS == "windows" {
		venv = filepath.Join(root, ".venv", "Scripts", "python.exe")
	}
	if _, err := os.Stat(venv); err == nil {
		return venv
	}
	return "python"
}

// checkAgreement refuses to serve if the backend and the worker disagree. The
// backend's prompt tells the policy which tools exist, and the worker is what
// answers it, so a difference means advertising a tool that always fails.
func checkAgreement(contract backend.Contract, info toolworker.Info) error {
	advertised := sortedCopy(contract.Tools)
	held := sortedCopy(info.Tools)
	if !reflect.DeepEqual(advertised, held) {
		return fmt.Errorf("the backend advertises tools %v but the worker holds %v", advertised, held)
	}
	if contract.MaxCalls != info.MaxCalls {
		return fmt.Errorf("call budget: backend %d, worker %d", contract.MaxCalls, info.MaxCalls)
	}
	if contract.MaxToolOutputChars != info.MaxToolOutputChars {
		return fmt.Errorf("tool output cap: backend %d, worker %d", contract.MaxToolOutputChars, info.MaxToolOutputChars)
	}
	return nil
}

func sortedCopy(values []string) []string {
	copied := append([]string{}, values...)
	sort.Strings(copied)
	return copied
}
