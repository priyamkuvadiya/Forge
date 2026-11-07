package metrics

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/database"
)

const (
	config  = `{"record": "config", "policy": {"model": "m"}}`
	step1   = `{"record": "step", "step": 1, "mean_reward": 0.2446, "loss": -0.03526}`
	step2   = `{"record": "step", "step": 2, "mean_reward": 0.5, "penalized": 9}`
	summary = `{"record": "summary", "wall_clock_seconds": 10998.6}`
)

func newStore(t *testing.T) *Store {
	t.Helper()
	db, err := database.Open(filepath.Join(t.TempDir(), "forge.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return NewStore(db)
}

func writeRun(t *testing.T, root, name string, lines ...string) {
	t.Helper()
	directory := filepath.Join(root, name)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	contents := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(directory, "steps.jsonl"), []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRecordsReadBackAsTheyWereWritten(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	writeRun(t, root, "run1", config, step1, step2, summary)

	report, err := store.Import(root)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(report.Imported, []string{"run1"}) || len(report.Failed) != 0 {
		t.Fatalf("report = %+v", report)
	}
	run, err := store.Get("run1")
	if err != nil {
		t.Fatal(err)
	}
	if !run.Finished || run.WallClockSeconds == nil || *run.WallClockSeconds != 10998.6 || run.Steps != 2 {
		t.Fatalf("run = %+v", run.Summary)
	}
	if string(run.Config) != config || string(run.Records[0]) != step1 || string(run.Records[1]) != step2 {
		t.Fatalf("run = %s %s", run.Config, run.Records)
	}
}

func TestAFieldARunNeverLoggedStaysAbsent(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	writeRun(t, root, "run1", config, step1)
	if _, err := store.Import(root); err != nil {
		t.Fatal(err)
	}
	run, _ := store.Get("run1")
	var record map[string]any
	if err := json.Unmarshal(run.Records[0], &record); err != nil {
		t.Fatal(err)
	}
	if _, present := record["penalized"]; present {
		t.Fatalf("record = %v", record)
	}
}

func TestARunStillGoingIsMarkedAndUpdated(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	writeRun(t, root, "run1", config, step1)
	if _, err := store.Import(root); err != nil {
		t.Fatal(err)
	}
	run, _ := store.Get("run1")
	if run.Finished || run.WallClockSeconds != nil || run.Steps != 1 {
		t.Fatalf("run = %+v", run.Summary)
	}

	writeRun(t, root, "run1", config, step1, step2, summary)
	report, _ := store.Import(root)
	if !reflect.DeepEqual(report.Imported, []string{"run1"}) {
		t.Fatalf("report = %+v", report)
	}
	run, _ = store.Get("run1")
	if !run.Finished || run.Steps != 2 {
		t.Fatalf("run = %+v", run.Summary)
	}
}

func TestAnUnchangedLogIsSkipped(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	writeRun(t, root, "run1", config, step1)
	_, _ = store.Import(root)
	report, err := store.Import(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Imported) != 0 || !reflect.DeepEqual(report.Unchanged, []string{"run1"}) {
		t.Fatalf("report = %+v", report)
	}
}

func TestABadLogIsRefusedWholeAndKeepsTheEarlierImport(t *testing.T) {
	cases := map[string][]string{
		"torn last line":  {config, step1, `{"record": "step", "st`},
		"nan":             {config, `{"record": "step", "step": 1, "loss": NaN}`},
		"no config":       {step1},
		"second config":   {config, config},
		"unknown record":  {config, `{"record": "mystery"}`},
		"no record field": {config, `{"step": 1}`},
		"unnumbered step": {config, `{"record": "step"}`},
		"repeated step":   {config, step1, step1},
		"second summary":  {config, summary, summary},
	}
	for name, lines := range cases {
		t.Run(name, func(t *testing.T) {
			store, root := newStore(t), t.TempDir()
			writeRun(t, root, "good", config, step1, step2)
			writeRun(t, root, "run1", config, step1, step2)
			if _, err := store.Import(root); err != nil {
				t.Fatal(err)
			}

			writeRun(t, root, "run1", lines...)
			report, err := store.Import(root)
			if err != nil {
				t.Fatal(err)
			}
			if report.Failed["run1"] == nil {
				t.Fatalf("report = %+v", report)
			}
			if !reflect.DeepEqual(report.Unchanged, []string{"good"}) {
				t.Fatalf("one bad run stopped the others: %+v", report)
			}
			run, err := store.Get("run1")
			if err != nil || run.Steps != 2 {
				t.Fatalf("run = %+v, err = %v", run.Summary, err)
			}
		})
	}
}

func TestWindowsLineEndingsAreNotPartOfARecord(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	directory := filepath.Join(root, "run1")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	contents := config + "\r\n" + step1 + "\r\n"
	if err := os.WriteFile(filepath.Join(directory, "steps.jsonl"), []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Import(root); err != nil {
		t.Fatal(err)
	}
	run, err := store.Get("run1")
	if err != nil {
		t.Fatal(err)
	}
	if string(run.Config) != config || string(run.Records[0]) != step1 {
		t.Fatalf("run = %q %q", run.Config, run.Records)
	}
}

func TestStepsComeBackInOrder(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	writeRun(t, root, "run1", config, step2, step1)
	if _, err := store.Import(root); err != nil {
		t.Fatal(err)
	}
	run, _ := store.Get("run1")
	if string(run.Records[0]) != step1 {
		t.Fatalf("records = %s", run.Records)
	}
}

func TestOnlyDirectoriesWithALogAreRuns(t *testing.T) {
	store, root := newStore(t), t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "empty"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "stray.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeRun(t, root, "b", config)
	writeRun(t, root, "a", config)
	writeRun(t, root, "smoke_all", config, step1)
	if _, err := store.Import(root); err != nil {
		t.Fatal(err)
	}
	summaries, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 || summaries[0].Name != "a" || summaries[1].Name != "b" {
		t.Fatalf("summaries = %+v", summaries)
	}
}

func TestAMissingRootImportsNothing(t *testing.T) {
	store := newStore(t)
	report, err := store.Import(filepath.Join(t.TempDir(), "nowhere"))
	if err != nil || len(report.Imported) != 0 {
		t.Fatalf("report = %+v, err = %v", report, err)
	}
}

func TestAnUnknownRun(t *testing.T) {
	if _, err := newStore(t).Get("missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v", err)
	}
}
