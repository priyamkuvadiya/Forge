package session

import (
	"database/sql"
	"errors"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/database"
)

func exchange(reason, prompt, reply string) agent.Episode {
	return agent.Episode{
		FinishReason: reason,
		Messages: []agent.Message{
			{Role: "user", Content: prompt},
			{Role: "assistant", Content: reply},
		},
	}
}

func openAt(t *testing.T, path string) *sql.DB {
	t.Helper()
	db, err := database.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func newStore(t *testing.T, maxChars int) *Store {
	t.Helper()
	return NewStore(openAt(t, filepath.Join(t.TempDir(), "forge.db")), maxChars)
}

func create(t *testing.T, store *Store, policy string) Session {
	t.Helper()
	created, err := store.Create(policy)
	if err != nil {
		t.Fatal(err)
	}
	return created
}

func end(t *testing.T, store *Store, id string, episode agent.Episode) {
	t.Helper()
	if err := store.End(id, episode); err != nil {
		t.Fatal(err)
	}
}

func TestOneEpisodeAtATime(t *testing.T) {
	store := newStore(t, 1000)
	created := create(t, store, "run2")

	if _, err := store.Begin(created.ID, "a"); err != nil {
		t.Fatal(err)
	}
	if found, _ := store.Get(created.ID); !found.Running {
		t.Fatal("not marked running")
	}
	if _, err := store.Begin(created.ID, "b"); !errors.Is(err, ErrBusy) {
		t.Fatalf("err = %v", err)
	}

	end(t, store, created.ID, exchange(agent.FinishAnswer, "a", "x"))
	if _, err := store.Begin(created.ID, "b"); err != nil {
		t.Fatalf("not released: %v", err)
	}
}

func TestUnknownSessions(t *testing.T) {
	store := newStore(t, 1000)
	if _, err := store.Get("missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v", err)
	}
	if _, err := store.Begin("missing", "a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v", err)
	}
	end(t, store, "missing", agent.Episode{}) // Must not panic.
}

func TestHistoryIsEveryCompletedEpisodeInOrder(t *testing.T) {
	store := newStore(t, 1000)
	created := create(t, store, "run2")

	begun, _ := store.Begin(created.ID, "one")
	if len(begun.History) != 0 || begun.Policy != "run2" || begun.EpisodeID == "" {
		t.Fatalf("begun = %+v", begun)
	}
	end(t, store, created.ID, exchange(agent.FinishAnswer, "one", "1"))
	// A policy that fails to answer still said what it said.
	begun, _ = store.Begin(created.ID, "two")
	end(t, store, created.ID, exchange(agent.FinishNoAnswer, "two", "2"))

	begun, _ = store.Begin(created.ID, "three")
	want := []agent.Message{
		{Role: "user", Content: "one"},
		{Role: "assistant", Content: "1"},
		{Role: "user", Content: "two"},
		{Role: "assistant", Content: "2"},
	}
	if !reflect.DeepEqual(begun.History, want) {
		t.Fatalf("history = %+v", begun.History)
	}
}

func TestASystemFailureIsKeptForDisplayButNotShownToThePolicy(t *testing.T) {
	for _, reason := range []string{agent.FinishBackendError, agent.FinishToolError, agent.FinishCancelled} {
		t.Run(reason, func(t *testing.T) {
			store := newStore(t, 1000)
			created := create(t, store, "run2")
			_, _ = store.Begin(created.ID, "one")
			end(t, store, created.ID, exchange(reason, "one", "half a reply"))

			begun, err := store.Begin(created.ID, "two")
			if err != nil {
				t.Fatal(err)
			}
			if len(begun.History) != 0 {
				t.Fatalf("history = %+v", begun.History)
			}
			end(t, store, created.ID, exchange(agent.FinishAnswer, "two", "2"))
			if found, _ := store.Get(created.ID); len(found.Episodes) != 2 {
				t.Fatalf("episodes = %d", len(found.Episodes))
			}
		})
	}
}

func TestTheLimitCountsCharactersAcrossTheConversation(t *testing.T) {
	store := newStore(t, 10)
	created := create(t, store, "run2")

	// Ten multibyte characters are within a ten-character limit.
	if _, err := store.Begin(created.ID, "✓✓✓✓✓✓✓✓✓✓"); err != nil {
		t.Fatalf("counted bytes: %v", err)
	}
	end(t, store, created.ID, exchange(agent.FinishAnswer, "aaaa", "bbbb"))

	if _, err := store.Begin(created.ID, "ccc"); !errors.Is(err, ErrTooLong) {
		t.Fatalf("err = %v", err)
	}
	// Being refused must not leave the session claimed.
	if _, err := store.Begin(created.ID, "cc"); err != nil {
		t.Fatalf("err = %v", err)
	}
}

func TestTheLimitCountsCharactersInTheHistoryToo(t *testing.T) {
	store := newStore(t, 10)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "a")
	end(t, store, created.ID, exchange(agent.FinishAnswer, "✓✓✓✓", "✓✓✓✓"))

	// Eight characters of history and two of prompt, though 24 bytes of history.
	if _, err := store.Begin(created.ID, "bb"); err != nil {
		t.Fatalf("counted bytes: %v", err)
	}
}

func TestAMessageThatCannotBeRecordedDoesNotClaimTheSession(t *testing.T) {
	db := openAt(t, filepath.Join(t.TempDir(), "forge.db"))
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	_, err := db.Exec(`CREATE TRIGGER refuse BEFORE INSERT ON episodes
		BEGIN SELECT RAISE(FAIL, 'disk full, say'); END`)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := store.Begin(created.ID, "a"); err == nil {
		t.Fatal("an episode would have run with no record of its message")
	}
	if _, busy := store.running[created.ID]; busy {
		t.Fatal("the session stayed claimed")
	}
}

func TestARefusedMessageLeavesNoRow(t *testing.T) {
	store := newStore(t, 3)
	created := create(t, store, "run2")
	if _, err := store.Begin(created.ID, "too long"); !errors.Is(err, ErrTooLong) {
		t.Fatalf("err = %v", err)
	}
	found, _ := store.Get(created.ID)
	if len(found.Episodes) != 0 {
		t.Fatalf("episodes = %+v", found.Episodes)
	}
}

func TestListIsOldestFirstWithoutTranscripts(t *testing.T) {
	store := newStore(t, 1000)
	first := create(t, store, "baseline")
	second := create(t, store, "run2")
	_, _ = store.Begin(second.ID, "a")
	end(t, store, second.ID, exchange(agent.FinishAnswer, "a", "b"))

	summaries, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 || summaries[0].ID != first.ID || summaries[1].ID != second.ID {
		t.Fatalf("summaries = %+v", summaries)
	}
	if summaries[1].Episodes != 1 {
		t.Fatalf("summaries[1] = %+v", summaries[1])
	}
}

func TestARunningEpisodeIsNotYetListed(t *testing.T) {
	store := newStore(t, 1000)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "a")

	found, _ := store.Get(created.ID)
	if !found.Running || len(found.Episodes) != 0 {
		t.Fatalf("found = %+v", found)
	}
	summaries, _ := store.List()
	if !summaries[0].Running || summaries[0].Episodes != 0 {
		t.Fatalf("summary = %+v", summaries[0])
	}
}

func TestListOrderSurvivesEqualTimestamps(t *testing.T) {
	store := newStore(t, 1000)
	ids := []string{}
	for range 20 {
		ids = append(ids, create(t, store, "run2").ID)
	}
	summaries, _ := store.List()
	for index, summary := range summaries {
		if summary.ID != ids[index] {
			t.Fatalf("position %d: %s, want %s", index, summary.ID, ids[index])
		}
	}
}

func TestEverythingSurvivesAReopenExactly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "forge.db")
	db := openAt(t, path)
	store := NewStore(db, 1000)
	created := create(t, store, "run2")

	answer := "42"
	toolError := "boom"
	episode := agent.Episode{
		ID:     "replaced by the store",
		Policy: "run2",
		// The characters a transcript really holds, and that an encoder could change.
		Messages: []agent.Message{
			{Role: "user", Content: "<b>✓</b> & \"quotes\"\n\ttabs  "},
			{Role: "assistant", Content: `<tool name="calculator">6 * 7</tool>`},
			{Role: "user", Content: "<tool_result name=\"calculator\">\n6 * 7 = 42\n</tool_result>"},
			{Role: "assistant", Content: "<answer>42</answer>"},
		},
		Nudges:       2,
		FinishReason: agent.FinishAnswer,
		Answer:       &answer,
		Trace: []agent.TraceEntry{
			{Index: 0, Tool: "calculator", Arguments: "6 * 7", OK: true, Output: "6 * 7 = 42", DurationMS: 1.25},
			{Index: 1, Tool: "python", Arguments: "x", OK: false, Error: &toolError},
		},
	}
	begun, err := store.Begin(created.ID, episode.Messages[0].Content)
	if err != nil {
		t.Fatal(err)
	}
	end(t, store, created.ID, episode)
	before, err := store.Get(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	db.Close()

	again := NewStore(openAt(t, path), 1000)
	after, err := again.Get(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("after = %+v\nbefore = %+v", after, before)
	}
	if !after.CreatedAt.Equal(created.CreatedAt) {
		t.Fatalf("created at %v, was %v", after.CreatedAt, created.CreatedAt)
	}
	if after.Episodes[0].ID != begun.EpisodeID {
		t.Fatalf("id = %s, want %s", after.Episodes[0].ID, begun.EpisodeID)
	}
	episode.ID = begun.EpisodeID
	if !reflect.DeepEqual(after.Episodes[0], episode) {
		t.Fatalf("episode = %+v", after.Episodes[0])
	}

	next, err := again.Begin(created.ID, "next")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(next.History, episode.Messages) {
		t.Fatalf("history = %+v", next.History)
	}
}

func TestAnEpisodeCutOffByARestartReadsAsInterrupted(t *testing.T) {
	path := filepath.Join(t.TempDir(), "forge.db")
	db := openAt(t, path)
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "first")
	end(t, store, created.ID, exchange(agent.FinishAnswer, "first", "1"))
	begun, _ := store.Begin(created.ID, "cut off")
	db.Close() // No End, as after a crash.

	again := NewStore(openAt(t, path), 1000)
	found, err := again.Get(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if found.Running {
		t.Fatal("nothing can be running after a restart")
	}
	if len(found.Episodes) != 2 {
		t.Fatalf("episodes = %+v", found.Episodes)
	}
	cut := found.Episodes[1]
	if cut.ID != begun.EpisodeID || cut.FinishReason != agent.FinishInterrupted || !cut.Failed() ||
		cut.Policy != "run2" || cut.Answer != nil {
		t.Fatalf("episode = %+v", cut)
	}
	if want := []agent.Message{{Role: "user", Content: "cut off"}}; !reflect.DeepEqual(cut.Messages, want) {
		t.Fatalf("messages = %+v", cut.Messages)
	}

	// The session is usable, and the policy never sees the cut-off message.
	next, err := again.Begin(created.ID, "second")
	if err != nil {
		t.Fatalf("not released: %v", err)
	}
	want := []agent.Message{{Role: "user", Content: "first"}, {Role: "assistant", Content: "1"}}
	if !reflect.DeepEqual(next.History, want) {
		t.Fatalf("history = %+v", next.History)
	}
}

func TestAnInterruptedEpisodeKeepsWhatItHadDone(t *testing.T) {
	path := filepath.Join(t.TempDir(), "forge.db")
	db := openAt(t, path)
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	begun, _ := store.Begin(created.ID, "6*7?")

	ok, ran := true, true
	events := []agent.Event{
		{Type: agent.EventStarted, Episode: begun.EpisodeID},
		{Type: agent.EventTurn, Kind: agent.KindToolCall, Text: `<tool name="calculator">6*7</tool>`,
			ToolCall: &agent.ToolCall{Name: "calculator", Arguments: "6*7"}},
		{Type: agent.EventToolResult, Tool: "calculator", OK: &ok, Ran: &ran, Message: "6*7 = 42",
			Trace: &agent.TraceEntry{Tool: "calculator", Arguments: "6*7", OK: true, Output: "42"}},
	}
	// Each checkpoint replaces the last, so only the latest counts.
	if err := store.Progress(created.ID, events[:1]); err != nil {
		t.Fatal(err)
	}
	if err := store.Progress(created.ID, events); err != nil {
		t.Fatal(err)
	}
	db.Close() // The process dies here.

	found, err := NewStore(openAt(t, path), 1000).Get(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	cut := found.Episodes[0]
	if cut.ID != begun.EpisodeID || cut.FinishReason != agent.FinishInterrupted || cut.Answer != nil {
		t.Fatalf("episode = %+v", cut)
	}
	want := []agent.Message{
		{Role: "user", Content: "6*7?"},
		{Role: "assistant", Content: `<tool name="calculator">6*7</tool>`},
		{Role: "user", Content: "6*7 = 42"},
	}
	if !reflect.DeepEqual(cut.Messages, want) || len(cut.Trace) != 1 || cut.Trace[0].Output != "42" {
		t.Fatalf("episode = %+v", cut)
	}
}

func TestACheckpointCannotTouchAFinishedEpisode(t *testing.T) {
	db := openAt(t, filepath.Join(t.TempDir(), "forge.db"))
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "a")
	if err := store.Progress(created.ID, []agent.Event{{Type: agent.EventStarted}}); err != nil {
		t.Fatal(err)
	}
	end(t, store, created.ID, exchange(agent.FinishAnswer, "a", "b"))

	// A late checkpoint, after the session was released.
	if err := store.Progress(created.ID, []agent.Event{{Type: agent.EventNudge}}); err != nil {
		t.Fatal(err)
	}
	var events sql.NullString
	if err := db.QueryRow("SELECT events FROM episodes").Scan(&events); err != nil {
		t.Fatal(err)
	}
	if events.Valid {
		t.Fatalf("a finished episode still holds a checkpoint: %s", events.String)
	}
	found, _ := store.Get(created.ID)
	if found.Episodes[0].FinishReason != agent.FinishAnswer || found.Episodes[0].Nudges != 0 {
		t.Fatalf("episode = %+v", found.Episodes[0])
	}
}

func TestAnEpisodeThatFailsToSaveReleasesTheSession(t *testing.T) {
	db := openAt(t, filepath.Join(t.TempDir(), "forge.db"))
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "a")
	db.Close()

	if err := store.End(created.ID, exchange(agent.FinishAnswer, "a", "b")); err == nil {
		t.Fatal("the failed write was not reported")
	}
	if _, busy := store.running[created.ID]; busy {
		t.Fatal("the session stayed claimed")
	}
}

func TestAFailedSaveReadsAsInterruptedInTheSameProcess(t *testing.T) {
	db := openAt(t, filepath.Join(t.TempDir(), "forge.db"))
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "a")
	delete(store.running, created.ID) // What End leaves behind when its write fails.

	found, err := store.Get(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(found.Episodes) != 1 || found.Episodes[0].FinishReason != agent.FinishInterrupted {
		t.Fatalf("episodes = %+v", found.Episodes)
	}
}

func TestACorruptEpisodeIsAnErrorNotAGap(t *testing.T) {
	db := openAt(t, filepath.Join(t.TempDir(), "forge.db"))
	store := NewStore(db, 1000)
	created := create(t, store, "run2")
	_, _ = store.Begin(created.ID, "a")
	end(t, store, created.ID, exchange(agent.FinishAnswer, "a", "b"))
	if _, err := db.Exec("UPDATE episodes SET body = 'not json'"); err != nil {
		t.Fatal(err)
	}

	if _, err := store.Get(created.ID); err == nil || errors.Is(err, ErrNotFound) {
		t.Fatalf("Get: %v", err)
	}
	// Answering with part of a conversation would be worse than refusing.
	if _, err := store.Begin(created.ID, "b"); err == nil {
		t.Fatal("Begin ignored it")
	}
	if _, busy := store.running[created.ID]; busy {
		t.Fatal("the refusal left the session claimed")
	}
}
