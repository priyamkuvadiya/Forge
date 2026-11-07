// Package session keeps conversations.
//
// In memory for now, so a restart loses them. SQLite persistence replaces this
// store's internals next, and the API above it should not need to change.
package session

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
)

var (
	ErrNotFound = errors.New("no such session")
	// ErrBusy: a session runs one episode at a time, since a second message
	// would be answered against a history that is about to change.
	ErrBusy = errors.New("this session is already running an episode")
	// ErrTooLong: the conversation would be over the limit.
	ErrTooLong = errors.New("this conversation is too long; start a new session")
)

// Session is a conversation with one policy. The policy is fixed for its whole
// life: replies written by one set of weights are not a context any other set was
// ever measured on. Comparing policies means opening one session for each.
type Session struct {
	ID        string          `json:"id"`
	Policy    string          `json:"policy"`
	CreatedAt time.Time       `json:"created_at"`
	Running   bool            `json:"running"`
	Episodes  []agent.Episode `json:"episodes"`
}

// Summary is a session without its transcripts.
type Summary struct {
	ID        string    `json:"id"`
	Policy    string    `json:"policy"`
	CreatedAt time.Time `json:"created_at"`
	Running   bool      `json:"running"`
	Episodes  int       `json:"episodes"`
}

// Begun is what an episode needs to start.
type Begun struct {
	EpisodeID string
	Policy    string
	History   []agent.Message
}

type Store struct {
	maxChars int

	mu       sync.Mutex
	sessions map[string]*Session
	// Creation order. Timestamps cannot give it: two sessions created in the same
	// clock tick, which on Windows is common, would compare equal.
	order []string
}

// NewStore limits a conversation to maxChars at the start of each episode.
func NewStore(maxChars int) *Store {
	return &Store{maxChars: maxChars, sessions: map[string]*Session{}}
}

func newID(prefix string) string {
	var bytes [8]byte
	// crypto/rand.Read does not fail on any supported platform.
	_, _ = rand.Read(bytes[:])
	return prefix + hex.EncodeToString(bytes[:])
}

func (s *Store) Create(policy string) Session {
	created := &Session{
		ID:        newID("s-"),
		Policy:    policy,
		CreatedAt: time.Now().UTC(),
		Episodes:  []agent.Episode{},
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[created.ID] = created
	s.order = append(s.order, created.ID)
	return *created
}

func (s *Store) Get(id string) (Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	found, ok := s.sessions[id]
	if !ok {
		return Session{}, ErrNotFound
	}
	copied := *found
	copied.Episodes = append([]agent.Episode{}, found.Episodes...)
	return copied, nil
}

// List returns every session, oldest first.
func (s *Store) List() []Summary {
	s.mu.Lock()
	defer s.mu.Unlock()
	summaries := make([]Summary, 0, len(s.order))
	for _, id := range s.order {
		found := s.sessions[id]
		summaries = append(summaries, Summary{
			ID:        found.ID,
			Policy:    found.Policy,
			CreatedAt: found.CreatedAt,
			Running:   found.Running,
			Episodes:  len(found.Episodes),
		})
	}
	return summaries
}

// Begin claims the session for one episode and returns what the policy should
// see before the new message. Every Begin must be matched by an End.
//
// An episode the system failed (Episode.Failed) is kept for display but left out
// of the history, so an outage does not become something the policy is shown.
func (s *Store) Begin(id, prompt string) (Begun, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	found, ok := s.sessions[id]
	if !ok {
		return Begun{}, ErrNotFound
	}
	if found.Running {
		return Begun{}, ErrBusy
	}

	history := []agent.Message{}
	chars := utf8.RuneCountInString(prompt)
	for _, episode := range found.Episodes {
		if episode.Failed() {
			continue
		}
		for _, message := range episode.Messages {
			history = append(history, message)
			chars += utf8.RuneCountInString(message.Content)
		}
	}
	if chars > s.maxChars {
		return Begun{}, ErrTooLong
	}

	found.Running = true
	return Begun{EpisodeID: newID("e-"), Policy: found.Policy, History: history}, nil
}

// End records a finished episode and releases the session.
func (s *Store) End(id string, episode agent.Episode) {
	s.mu.Lock()
	defer s.mu.Unlock()
	found, ok := s.sessions[id]
	if !ok {
		return
	}
	found.Running = false
	found.Episodes = append(found.Episodes, episode)
}
