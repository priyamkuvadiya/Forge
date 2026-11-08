package api

import (
	"context"
	"log"
	"slices"
	"sync"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/session"
)

// live is one running episode: every event it has emitted, for any number of
// followers arriving at any point, and the means to stop it.
//
// Episodes run on the server's lifetime, not on the request that started them,
// so a page that refreshes or loses its connection mid-episode does not cancel
// it. A new page reattaches with GET /sessions/{id}/live and is sent every event
// from the start, then the rest as they happen.
type live struct {
	episodeID string
	cancel    context.CancelFunc

	mu     sync.Mutex
	events []agent.Event
	// Closed and replaced on every change, which wakes every follower at once.
	wake    chan struct{}
	done    bool
	outcome agent.Episode
}

func newLive(episodeID string, cancel context.CancelFunc) *live {
	return &live{episodeID: episodeID, cancel: cancel, wake: make(chan struct{})}
}

// publish appends an event and returns every event so far, for the checkpoint.
func (l *live) publish(event agent.Event) []agent.Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, event)
	close(l.wake)
	l.wake = make(chan struct{})
	return slices.Clone(l.events)
}

// finish appends the final event and marks the episode over, in one step, so a
// follower sees either neither or both.
func (l *live) finish(outcome agent.Episode, final agent.Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, final)
	l.done = true
	l.outcome = outcome
	close(l.wake)
}

// follow sends every event from the first, then each new one, until the episode
// ends or ctx does. Leaving early does not affect the episode.
func (l *live) follow(ctx context.Context, send func(agent.Event)) {
	next := 0
	for {
		l.mu.Lock()
		batch := slices.Clone(l.events[next:])
		done, wake := l.done, l.wake
		l.mu.Unlock()

		for _, event := range batch {
			send(event)
		}
		next += len(batch)
		if done {
			return
		}
		select {
		case <-wake:
		case <-ctx.Done():
			return
		}
	}
}

// wait returns the finished episode, or false if ctx ends first.
func (l *live) wait(ctx context.Context) (agent.Episode, bool) {
	for {
		l.mu.Lock()
		if l.done {
			outcome := l.outcome
			l.mu.Unlock()
			return outcome, true
		}
		wake := l.wake
		l.mu.Unlock()
		select {
		case <-wake:
		case <-ctx.Done():
			return agent.Episode{}, false
		}
	}
}

// running returns the live episode in a session, or nil.
func (s *Server) running(id string) *live {
	s.liveMu.Lock()
	defer s.liveMu.Unlock()
	return s.live[id]
}

// start runs an episode the store has begun, detached from any request.
func (s *Server) start(id string, begun session.Begun, prompt string) *live {
	ctx, cancel := context.WithCancel(s.lifetime())
	episode := newLive(begun.EpisodeID, cancel)
	s.liveMu.Lock()
	s.live[id] = episode
	s.liveMu.Unlock()

	s.episodes.Add(1)
	go func() {
		defer s.episodes.Done()
		defer cancel()

		var final *agent.Event
		emit := func(event agent.Event) {
			if event.Type == agent.EventFinished {
				// Held back until the record is saved, so that a client which
				// reloads the session on seeing it always finds the episode there.
				held := event
				final = &held
				return
			}
			if err := s.Store.Progress(id, episode.publish(event)); err != nil {
				// The episode carries on; it would only read back with less in it.
				log.Printf("%v", err)
			}
		}
		outcome := s.Runner.Run(ctx, begun.EpisodeID, begun.Policy, begun.History, prompt, emit)
		// A failed save costs only the later record, which then reads back as
		// interrupted with its last checkpoint. The client still gets the episode.
		if err := s.Store.End(id, outcome); err != nil {
			log.Printf("%v", err)
		}

		s.liveMu.Lock()
		delete(s.live, id)
		s.liveMu.Unlock()

		if final == nil {
			final = &agent.Event{Type: agent.EventFinished, Episode: begun.EpisodeID}
		}
		final.Finished = &outcome
		episode.finish(outcome, *final)
	}()
	return episode
}

func (s *Server) lifetime() context.Context {
	if s.Context != nil {
		return s.Context
	}
	return context.Background()
}

// Wait blocks until every episode has ended and been recorded. Cancel the
// server's Context first, or this waits for them to finish on their own.
func (s *Server) Wait() { s.episodes.Wait() }
