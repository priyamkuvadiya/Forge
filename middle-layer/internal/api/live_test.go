package api

import (
	"context"
	"testing"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
)

// A page that goes away must stop being written to straight away, not when the
// episode it was following ends, which can be minutes later.
func TestAFollowerWhoseClientLeftStopsBeforeTheEpisodeEnds(t *testing.T) {
	episode := newLive("e-1", func() {})
	episode.publish(agent.Event{Type: agent.EventStarted, Episode: "e-1"})

	ctx, leave := context.WithCancel(context.Background())
	sent := make(chan agent.Event, 16)
	returned := make(chan struct{})
	go func() {
		episode.follow(ctx, func(e agent.Event) { sent <- e })
		close(returned)
	}()

	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("the follower was not sent the events so far")
	}

	leave()
	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("the follower kept running after its client left")
	}

	// The episode is unaffected and carries on; nothing more reaches the follower.
	episode.publish(agent.Event{Type: agent.EventTurn, Episode: "e-1"})
	episode.finish(agent.Episode{ID: "e-1"}, agent.Event{Type: agent.EventFinished, Episode: "e-1"})
	if len(sent) != 0 {
		t.Fatalf("%d events sent after the client left", len(sent))
	}
}

// And a follower still attached gets every event, then returns when it ends.
func TestAFollowerStillAttachedGetsEverythingThenReturns(t *testing.T) {
	episode := newLive("e-1", func() {})
	var got []string
	done := make(chan struct{})
	go func() {
		episode.follow(context.Background(), func(e agent.Event) { got = append(got, e.Type) })
		close(done)
	}()
	episode.publish(agent.Event{Type: agent.EventStarted, Episode: "e-1"})
	episode.publish(agent.Event{Type: agent.EventTurn, Episode: "e-1"})
	episode.finish(agent.Episode{ID: "e-1"}, agent.Event{Type: agent.EventFinished, Episode: "e-1"})
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the follower did not return when the episode ended")
	}
	if len(got) != 3 || got[2] != agent.EventFinished {
		t.Fatalf("got %v", got)
	}
}
