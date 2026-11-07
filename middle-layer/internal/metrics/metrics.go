// Package metrics copies training step logs into the database for the dashboard.
//
// The steps.jsonl files stay the source of truth, and nothing here computes a
// number: a record reads back as the training loop wrote it, or the run is not
// imported at all.
package metrics

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ErrNotFound means no run has that name.
var ErrNotFound = errors.New("no such run")

// Summary is a run without its records.
type Summary struct {
	Name             string   `json:"name"`
	Finished         bool     `json:"finished"`
	WallClockSeconds *float64 `json:"wall_clock_seconds"`
	Steps            int      `json:"steps"`
}

// Run is a run with its configuration and every step record, in step order.
type Run struct {
	Summary
	Config  json.RawMessage   `json:"config"`
	Records []json.RawMessage `json:"records"`
}

// Report says what one Import did.
type Report struct {
	Imported  []string
	Unchanged []string
	// A run that fails keeps whatever earlier import it had.
	Failed map[string]error
}

type Store struct {
	db *sql.DB
}

func NewStore(db *sql.DB) *Store { return &Store{db: db} }

// Import reads root/<run>/steps.jsonl for every run, skipping those whose file has
// not changed. A run with no log is not a run, and a missing root imports nothing.
func (s *Store) Import(root string) (Report, error) {
	report := Report{Failed: map[string]error{}}
	entries, err := os.ReadDir(root)
	if errors.Is(err, fs.ErrNotExist) {
		return report, nil
	}
	if err != nil {
		return report, err
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		// A smoke run proves the loop executes. Its rewards come from a handful of
		// prompts, and on a chart they would look like a result.
		if strings.HasPrefix(name, "smoke") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, name, "steps.jsonl"))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			report.Failed[name] = err
			continue
		}
		changed, err := s.importRun(name, data)
		switch {
		case err != nil:
			report.Failed[name] = err
		case changed:
			report.Imported = append(report.Imported, name)
		default:
			report.Unchanged = append(report.Unchanged, name)
		}
	}
	return report, nil
}

type parsedStep struct {
	step   int
	record string
}

type parsedRun struct {
	config   string
	finished bool
	seconds  *float64
	steps    []parsedStep
}

// parse is strict: a torn or unfamiliar log should be looked at, not half-charted.
func parse(data []byte) (parsedRun, error) {
	var run parsedRun
	seen := map[int]bool{}
	for index, raw := range bytes.Split(data, []byte("\n")) {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			continue
		}
		var head struct {
			Record  string   `json:"record"`
			Step    *int     `json:"step"`
			Seconds *float64 `json:"wall_clock_seconds"`
		}
		if err := json.Unmarshal(line, &head); err != nil {
			return run, fmt.Errorf("line %d: %w", index+1, err)
		}
		switch head.Record {
		case "config":
			if run.config != "" {
				return run, fmt.Errorf("line %d: a second config", index+1)
			}
			run.config = string(line)
		case "step":
			if head.Step == nil {
				return run, fmt.Errorf("line %d: a step with no number", index+1)
			}
			if seen[*head.Step] {
				return run, fmt.Errorf("line %d: step %d again", index+1, *head.Step)
			}
			seen[*head.Step] = true
			run.steps = append(run.steps, parsedStep{*head.Step, string(line)})
		case "summary":
			if run.finished {
				return run, fmt.Errorf("line %d: a second summary", index+1)
			}
			run.finished = true
			run.seconds = head.Seconds
		default:
			return run, fmt.Errorf("line %d: unknown record %q", index+1, head.Record)
		}
	}
	if run.config == "" {
		return run, errors.New("no config record")
	}
	return run, nil
}

func (s *Store) importRun(name string, data []byte) (bool, error) {
	digest := sha256.Sum256(data)
	source := hex.EncodeToString(digest[:])

	var existing string
	err := s.db.QueryRow("SELECT source_sha256 FROM runs WHERE name = ?", name).Scan(&existing)
	if err == nil && existing == source {
		return false, nil
	}
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return false, err
	}

	run, err := parse(data)
	if err != nil {
		return false, err
	}

	tx, err := s.db.Begin()
	if err != nil {
		return false, err
	}
	defer tx.Rollback()
	if _, err := tx.Exec("DELETE FROM runs WHERE name = ?", name); err != nil {
		return false, err
	}
	_, err = tx.Exec(
		"INSERT INTO runs (name, source_sha256, finished, wall_clock_seconds, config) VALUES (?, ?, ?, ?, ?)",
		name, source, run.finished, run.seconds, run.config,
	)
	if err != nil {
		return false, err
	}
	for _, step := range run.steps {
		_, err := tx.Exec(
			"INSERT INTO run_steps (run, step, record) VALUES (?, ?, ?)", name, step.step, step.record,
		)
		if err != nil {
			return false, err
		}
	}
	return true, tx.Commit()
}

// List returns every imported run, by name.
func (s *Store) List() ([]Summary, error) {
	rows, err := s.db.Query(`
		SELECT r.name, r.finished, r.wall_clock_seconds,
		       (SELECT COUNT(*) FROM run_steps t WHERE t.run = r.name)
		FROM runs r ORDER BY r.name`)
	if err != nil {
		return nil, fmt.Errorf("listing runs: %w", err)
	}
	defer rows.Close()

	summaries := []Summary{}
	for rows.Next() {
		var summary Summary
		if err := rows.Scan(&summary.Name, &summary.Finished, &summary.WallClockSeconds, &summary.Steps); err != nil {
			return nil, fmt.Errorf("listing runs: %w", err)
		}
		summaries = append(summaries, summary)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("listing runs: %w", err)
	}
	return summaries, nil
}

// Get returns one run with its records.
func (s *Store) Get(name string) (Run, error) {
	run := Run{Summary: Summary{Name: name}, Records: []json.RawMessage{}}
	var config string
	err := s.db.QueryRow("SELECT finished, wall_clock_seconds, config FROM runs WHERE name = ?", name).
		Scan(&run.Finished, &run.WallClockSeconds, &config)
	if errors.Is(err, sql.ErrNoRows) {
		return Run{}, ErrNotFound
	}
	if err != nil {
		return Run{}, fmt.Errorf("reading run %s: %w", name, err)
	}
	run.Config = json.RawMessage(config)

	rows, err := s.db.Query("SELECT record FROM run_steps WHERE run = ? ORDER BY step", name)
	if err != nil {
		return Run{}, fmt.Errorf("reading run %s: %w", name, err)
	}
	defer rows.Close()
	for rows.Next() {
		var record string
		if err := rows.Scan(&record); err != nil {
			return Run{}, fmt.Errorf("reading run %s: %w", name, err)
		}
		run.Records = append(run.Records, json.RawMessage(record))
	}
	if err := rows.Err(); err != nil {
		return Run{}, fmt.Errorf("reading run %s: %w", name, err)
	}
	run.Steps = len(run.Records)
	return run, nil
}
