// Package database opens the middle layer's SQLite file and owns its schema.
//
// The session store and the training metrics share this one file, so the demo
// needs no database server and no outside account.
package database

import (
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"modernc.org/sqlite"
	sqlite3 "modernc.org/sqlite/lib"
)

const schemaVersion = 2

// migrations[v] takes a database at version v to v+1. A fresh database runs the
// schema and then every migration, so the two paths cannot drift apart.
var migrations = map[int]string{
	// Version 2: an episode's events so far, rewritten after each one, so an
	// episode cut off by a crash keeps what it had done. NULL once it finishes,
	// when body is the record.
	1: `ALTER TABLE episodes ADD COLUMN events TEXT;`,
}

const schema = `
CREATE TABLE sessions (
	seq        INTEGER PRIMARY KEY AUTOINCREMENT,
	id         TEXT NOT NULL UNIQUE,
	policy     TEXT NOT NULL,
	created_at TEXT NOT NULL
);

-- A row is written when an episode starts, so a crash cannot lose the message
-- that caused it. finish_reason and body stay NULL until it ends.
CREATE TABLE episodes (
	seq           INTEGER PRIMARY KEY AUTOINCREMENT,
	id            TEXT NOT NULL UNIQUE,
	session_id    TEXT NOT NULL REFERENCES sessions(id),
	prompt        TEXT NOT NULL,
	started_at    TEXT NOT NULL,
	finished_at   TEXT,
	finish_reason TEXT,
	body          TEXT,
	CHECK ((finish_reason IS NULL) = (body IS NULL)),
	CHECK ((finish_reason IS NULL) = (finished_at IS NULL))
);
CREATE INDEX episodes_by_session ON episodes(session_id, seq);

CREATE TABLE runs (
	name               TEXT PRIMARY KEY,
	source_sha256      TEXT NOT NULL,
	finished           INTEGER NOT NULL,
	wall_clock_seconds REAL,
	config             TEXT NOT NULL
);

-- Each step log line, verbatim. The fields differ between runs (run1 has no
-- "penalized"), and keeping the record means a field a run never logged stays
-- absent rather than turning into a zero on a chart.
CREATE TABLE run_steps (
	run    TEXT NOT NULL REFERENCES runs(name) ON DELETE CASCADE,
	step   INTEGER NOT NULL,
	record TEXT NOT NULL,
	PRIMARY KEY (run, step)
);
`

// Open opens or creates the database and takes it for this process alone.
//
// The lock is what makes the session store's rule sound: an unfinished episode
// with no live owner was interrupted. A second process would read the first's
// running episodes as interrupted, so it is refused instead. The operating system
// releases the lock however the process ends, including a hard kill.
func Open(path string) (*sql.DB, error) {
	source, err := dataSource(path)
	if err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", source)
	if err != nil {
		return nil, fmt.Errorf("database %s: %w", path, err)
	}
	// One connection for the life of the process: the lock belongs to it, and
	// nothing here is busy enough to need more.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)
	db.SetConnMaxIdleTime(0)

	if err := prepare(db); err != nil {
		db.Close()
		if isBusy(err) {
			return nil, fmt.Errorf("database %s: another middle layer is using it", path)
		}
		return nil, fmt.Errorf("database %s: %w", path, err)
	}
	return db, nil
}

// dataSource builds the SQLite URI. Synchronous is FULL because this is a laptop
// that has lost power mid-run before, and an episode finishes rarely enough for
// the cost not to matter.
func dataSource(path string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("database %s: %w", path, err)
	}
	slashed := filepath.ToSlash(absolute)
	if !strings.HasPrefix(slashed, "/") {
		slashed = "/" + slashed // C:/x has to become /C:/x in a URI.
	}
	location := url.URL{Scheme: "file", Path: slashed}
	return location.String() + "?" +
		"_pragma=locking_mode(EXCLUSIVE)&" +
		"_pragma=journal_mode(WAL)&" +
		"_pragma=synchronous(FULL)&" +
		"_pragma=foreign_keys(1)", nil
}

func prepare(db *sql.DB) error {
	// Exclusive mode takes its lock at the first write, and a database that needs
	// no migration would otherwise never write one.
	if _, err := db.Exec("BEGIN IMMEDIATE"); err != nil {
		return err
	}
	if _, err := db.Exec("COMMIT"); err != nil {
		return err
	}

	var version int
	if err := db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		return err
	}
	if version > schemaVersion || version < 0 {
		return fmt.Errorf("schema version %d, this build only knows up to %d", version, schemaVersion)
	}
	if version == schemaVersion {
		return nil
	}
	return upgrade(db, version)
}

// upgrade brings the database to schemaVersion in one transaction, so a failure
// leaves it at the version it started at.
func upgrade(db *sql.DB, from int) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	version := from
	if version == 0 {
		if _, err := tx.Exec(schema); err != nil {
			return err
		}
		version = 1
	}
	for ; version < schemaVersion; version++ {
		if _, err := tx.Exec(migrations[version]); err != nil {
			return fmt.Errorf("migrating from version %d: %w", version, err)
		}
	}
	if _, err := tx.Exec(fmt.Sprintf("PRAGMA user_version = %d", schemaVersion)); err != nil {
		return err
	}
	return tx.Commit()
}

func isBusy(err error) bool {
	var failure *sqlite.Error
	if !errors.As(err, &failure) {
		return false
	}
	code := failure.Code() & 0xff // The low byte is the primary result code.
	return code == sqlite3.SQLITE_BUSY || code == sqlite3.SQLITE_LOCKED
}
