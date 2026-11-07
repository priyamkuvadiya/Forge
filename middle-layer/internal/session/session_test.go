package session

import (
	"errors"
	"reflect"
	"testing"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
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

func TestOneEpisodeAtATime(t *testing.T) {
	store := NewStore(1000)
	created := store.Create("run2")

	if _, err := store.Begin(created.ID, "a"); err != nil {
		t.Fatal(err)
	}
	if found, _ := store.Get(created.ID); !found.Running {
		t.Fatal("not marked running")
	}
	if _, err := store.Begin(created.ID, "b"); !errors.Is(err, ErrBusy) {
		t.Fatalf("err = %v", err)
	}

	store.End(created.ID, exchange(agent.FinishAnswer, "a", "x"))
	if _, err := store.Begin(created.ID, "b"); err != nil {
		t.Fatalf("not released: %v", err)
	}
}

func TestUnknownSessions(t *testing.T) {
	store := NewStore(1000)
	if _, err := store.Get("missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v", err)
	}
	if _, err := store.Begin("missing", "a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v", err)
	}
	store.End("missing", agent.Episode{}) // Must not panic.
}

func TestHistoryIsEveryCompletedEpisodeInOrder(t *testing.T) {
	store := NewStore(1000)
	created := store.Create("run2")

	begun, _ := store.Begin(created.ID, "one")
	if len(begun.History) != 0 || begun.Policy != "run2" || begun.EpisodeID == "" {
		t.Fatalf("begun = %+v", begun)
	}
	store.End(created.ID, exchange(agent.FinishAnswer, "one", "1"))
	// A policy that fails to answer still said what it said.
	begun, _ = store.Begin(created.ID, "two")
	store.End(created.ID, exchange(agent.FinishNoAnswer, "two", "2"))

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
			store := NewStore(1000)
			created := store.Create("run2")
			_, _ = store.Begin(created.ID, "one")
			store.End(created.ID, exchange(reason, "one", "half a reply"))

			begun, err := store.Begin(created.ID, "two")
			if err != nil {
				t.Fatal(err)
			}
			if len(begun.History) != 0 {
				t.Fatalf("history = %+v", begun.History)
			}
			store.End(created.ID, exchange(agent.FinishAnswer, "two", "2"))
			if found, _ := store.Get(created.ID); len(found.Episodes) != 2 {
				t.Fatalf("episodes = %d", len(found.Episodes))
			}
		})
	}
}

func TestTheLimitCountsCharactersAcrossTheConversation(t *testing.T) {
	store := NewStore(10)
	created := store.Create("run2")

	// Ten multibyte characters are within a ten-character limit.
	if _, err := store.Begin(created.ID, "✓✓✓✓✓✓✓✓✓✓"); err != nil {
		t.Fatalf("counted bytes: %v", err)
	}
	store.End(created.ID, exchange(agent.FinishAnswer, "aaaa", "bbbb"))

	if _, err := store.Begin(created.ID, "ccc"); !errors.Is(err, ErrTooLong) {
		t.Fatalf("err = %v", err)
	}
	// Being refused must not leave the session claimed.
	if _, err := store.Begin(created.ID, "cc"); err != nil {
		t.Fatalf("err = %v", err)
	}
}

func TestListIsOldestFirstWithoutTranscripts(t *testing.T) {
	store := NewStore(1000)
	first := store.Create("baseline")
	second := store.Create("run2")
	_, _ = store.Begin(second.ID, "a")
	store.End(second.ID, exchange(agent.FinishAnswer, "a", "b"))

	summaries := store.List()
	if len(summaries) != 2 || summaries[0].ID != first.ID || summaries[1].ID != second.ID {
		t.Fatalf("summaries = %+v", summaries)
	}
	if summaries[1].Episodes != 1 {
		t.Fatalf("summaries[1] = %+v", summaries[1])
	}
}

func TestGetReturnsACopy(t *testing.T) {
	store := NewStore(1000)
	created := store.Create("run2")
	_, _ = store.Begin(created.ID, "a")
	store.End(created.ID, exchange(agent.FinishAnswer, "a", "b"))

	found, _ := store.Get(created.ID)
	found.Episodes[0].FinishReason = "tampered"
	again, _ := store.Get(created.ID)
	if again.Episodes[0].FinishReason != agent.FinishAnswer {
		t.Fatal("the caller changed the stored episode")
	}
}
