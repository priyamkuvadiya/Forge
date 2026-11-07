package database

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenCreatesTheSchemaAndReopensIt(t *testing.T) {
	// This path goes into a URI, where an unescaped # would end it.
	directory := filepath.Join(t.TempDir(), "with space # ✓")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "forge.db")

	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("INSERT INTO sessions (id, policy, created_at) VALUES ('a', 'b', 'c')"); err != nil {
		t.Fatal(err)
	}
	db.Close()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the file is not where it was asked for: %v", err)
	}

	db, err = Open(path)
	if err != nil {
		t.Fatalf("reopening: %v", err)
	}
	defer db.Close()
	var count int
	if err := db.QueryRow("SELECT COUNT(*) FROM sessions").Scan(&count); err != nil || count != 1 {
		t.Fatalf("count = %d, err = %v", count, err)
	}
}

func TestASecondProcessIsRefusedAndTheLockIsReleased(t *testing.T) {
	path := filepath.Join(t.TempDir(), "forge.db")

	// Both the creating open and one that has nothing to migrate must take the lock.
	for _, name := range []string{"creating", "existing"} {
		first, err := Open(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		_, err = Open(path)
		if err == nil || !strings.Contains(err.Error(), "another middle layer") {
			t.Fatalf("%s: second open: %v", name, err)
		}
		first.Close()
	}
}

func TestDurabilitySettingsAreInForce(t *testing.T) {
	db, err := Open(filepath.Join(t.TempDir(), "forge.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var journal string
	var synchronous int
	if err := db.QueryRow("PRAGMA journal_mode").Scan(&journal); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRow("PRAGMA synchronous").Scan(&synchronous); err != nil {
		t.Fatal(err)
	}
	if journal != "wal" || synchronous != 2 {
		t.Fatalf("journal = %s, synchronous = %d (2 is FULL)", journal, synchronous)
	}
}

func TestForeignKeysAreEnforced(t *testing.T) {
	db, err := Open(filepath.Join(t.TempDir(), "forge.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	_, err = db.Exec(`INSERT INTO episodes (id, session_id, prompt, started_at) VALUES ('e', 'missing', 'p', 't')`)
	if err == nil {
		t.Fatal("an episode without a session was accepted")
	}
}

func TestAnUnfinishedEpisodeCannotHalfFinish(t *testing.T) {
	db, err := Open(filepath.Join(t.TempDir(), "forge.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec("INSERT INTO sessions (id, policy, created_at) VALUES ('s', 'p', 't')"); err != nil {
		t.Fatal(err)
	}
	for name, columns := range map[string]string{
		"no body":        `finish_reason, finished_at) VALUES ('e', 's', 'p', 't', 'answer', 't')`,
		"no finish time": `finish_reason, body) VALUES ('e', 's', 'p', 't', 'answer', '{}')`,
		"no reason":      `finished_at, body) VALUES ('e', 's', 'p', 't', 't', '{}')`,
	} {
		_, err := db.Exec(`INSERT INTO episodes (id, session_id, prompt, started_at, ` + columns)
		if err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}

func TestANewerSchemaIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "forge.db")
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("PRAGMA user_version = 99"); err != nil {
		t.Fatal(err)
	}
	db.Close()

	if _, err := Open(path); err == nil || !strings.Contains(err.Error(), "schema version 99") {
		t.Fatalf("err = %v", err)
	}
}

func TestAFileThatIsNotADatabaseIsRefusedAndKept(t *testing.T) {
	path := filepath.Join(t.TempDir(), "forge.db")
	contents := []byte(strings.Repeat("this is somebody's notes\n", 100))
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil {
		t.Fatal("opened it")
	}
	after, err := os.ReadFile(path)
	if err != nil || string(after) != string(contents) {
		t.Fatalf("the file changed: %v", err)
	}
}
