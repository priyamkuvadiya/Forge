// Package session keeps conversations, in the middle layer's SQLite file, so they
// survive a restart.
package session

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
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
	db       *sql.DB
	maxChars int

	mu sync.Mutex
	// Session ID -> the episode running in it. In memory on purpose: an episode
	// cannot outlive this process, so after a restart nothing is running, and an
	// unfinished row means that episode was interrupted. database.Open is what
	// guarantees no other process owns one.
	running map[string]string
}

// NewStore limits a conversation to maxChars at the start of each episode.
func NewStore(db *sql.DB, maxChars int) *Store {
	return &Store{db: db, maxChars: maxChars, running: map[string]string{}}
}

func newID(prefix string) string {
	var bytes [8]byte
	// crypto/rand.Read does not fail on any supported platform.
	_, _ = rand.Read(bytes[:])
	return prefix + hex.EncodeToString(bytes[:])
}

func stamp(at time.Time) string { return at.UTC().Format(time.RFC3339Nano) }

func (s *Store) Create(policy string) (Session, error) {
	created := Session{
		ID:        newID("s-"),
		Policy:    policy,
		CreatedAt: time.Now().UTC(),
		Episodes:  []agent.Episode{},
	}
	_, err := s.db.Exec(
		"INSERT INTO sessions (id, policy, created_at) VALUES (?, ?, ?)",
		created.ID, created.Policy, stamp(created.CreatedAt),
	)
	if err != nil {
		return Session{}, fmt.Errorf("creating a session: %w", err)
	}
	return created, nil
}

func (s *Store) Get(id string) (Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	found := Session{ID: id}
	var created string
	err := s.db.QueryRow("SELECT policy, created_at FROM sessions WHERE id = ?", id).
		Scan(&found.Policy, &created)
	if errors.Is(err, sql.ErrNoRows) {
		return Session{}, ErrNotFound
	}
	if err != nil {
		return Session{}, fmt.Errorf("reading session %s: %w", id, err)
	}
	if found.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
		return Session{}, fmt.Errorf("session %s: %w", id, err)
	}
	if found.Episodes, err = s.episodes(id, found.Policy); err != nil {
		return Session{}, err
	}
	_, found.Running = s.running[id]
	return found, nil
}

// episodes reads a session's episodes in order, without the one running now. The
// caller holds s.mu.
//
// A row with no finish reason and no live owner was interrupted, whether by a
// crash, a hard kill or a final write that failed. It is reported as a system
// failure holding whatever its last checkpoint (Progress) recorded: the message
// that caused it at least, and every turn and tool call up to the cut.
func (s *Store) episodes(id, policy string) ([]agent.Episode, error) {
	rows, err := s.db.Query(
		"SELECT id, prompt, body, events FROM episodes WHERE session_id = ? ORDER BY seq", id,
	)
	if err != nil {
		return nil, fmt.Errorf("reading session %s: %w", id, err)
	}
	defer rows.Close()

	episodes := []agent.Episode{}
	for rows.Next() {
		var episodeID, prompt string
		var body, progress sql.NullString
		if err := rows.Scan(&episodeID, &prompt, &body, &progress); err != nil {
			return nil, fmt.Errorf("reading session %s: %w", id, err)
		}
		if body.Valid {
			var episode agent.Episode
			if err := json.Unmarshal([]byte(body.String), &episode); err != nil {
				return nil, fmt.Errorf("episode %s: %w", episodeID, err)
			}
			episodes = append(episodes, episode)
			continue
		}
		if s.running[id] == episodeID {
			continue
		}
		var events []agent.Event
		if progress.Valid {
			if err := json.Unmarshal([]byte(progress.String), &events); err != nil {
				return nil, fmt.Errorf("episode %s: its progress: %w", episodeID, err)
			}
		}
		cut := agent.Replay(episodeID, policy, prompt, events)
		cut.FinishReason = agent.FinishInterrupted
		cut.Error = "the middle layer stopped before this episode finished"
		episodes = append(episodes, cut)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("reading session %s: %w", id, err)
	}
	return episodes, nil
}

// List returns every session, oldest first.
func (s *Store) List() ([]Summary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Ordered by seq, not by time: two sessions created in the same clock tick,
	// which on Windows is common, would compare equal.
	rows, err := s.db.Query(`
		SELECT s.id, s.policy, s.created_at,
		       (SELECT COUNT(*) FROM episodes e WHERE e.session_id = s.id)
		FROM sessions s ORDER BY s.seq`)
	if err != nil {
		return nil, fmt.Errorf("listing sessions: %w", err)
	}
	defer rows.Close()

	summaries := []Summary{}
	for rows.Next() {
		var summary Summary
		var created string
		if err := rows.Scan(&summary.ID, &summary.Policy, &created, &summary.Episodes); err != nil {
			return nil, fmt.Errorf("listing sessions: %w", err)
		}
		if summary.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
			return nil, fmt.Errorf("session %s: %w", summary.ID, err)
		}
		if _, summary.Running = s.running[summary.ID]; summary.Running {
			summary.Episodes-- // Its row exists, and it is not an episode yet.
		}
		summaries = append(summaries, summary)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("listing sessions: %w", err)
	}
	return summaries, nil
}

// Begin claims the session for one episode, records the message, and returns what
// the policy should see before it. Every Begin must be matched by an End.
//
// An episode the system failed (Episode.Failed) is kept for display but left out
// of the history, so an outage does not become something the policy is shown.
func (s *Store) Begin(id, prompt string) (Begun, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var policy string
	err := s.db.QueryRow("SELECT policy FROM sessions WHERE id = ?", id).Scan(&policy)
	if errors.Is(err, sql.ErrNoRows) {
		return Begun{}, ErrNotFound
	}
	if err != nil {
		return Begun{}, fmt.Errorf("reading session %s: %w", id, err)
	}
	if _, busy := s.running[id]; busy {
		return Begun{}, ErrBusy
	}

	episodes, err := s.episodes(id, policy)
	if err != nil {
		return Begun{}, err
	}
	history := []agent.Message{}
	chars := utf8.RuneCountInString(prompt)
	for _, episode := range episodes {
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

	episodeID := newID("e-")
	_, err = s.db.Exec(
		"INSERT INTO episodes (id, session_id, prompt, started_at) VALUES (?, ?, ?, ?)",
		episodeID, id, prompt, stamp(time.Now()),
	)
	if err != nil {
		return Begun{}, fmt.Errorf("recording a message in session %s: %w", id, err)
	}
	s.running[id] = episodeID
	return Begun{EpisodeID: episodeID, Policy: policy, History: history}, nil
}

// Progress checkpoints a running episode: its events so far, which is what it
// reads back as if the process stops before End. It writes only to the episode
// this session is running now, and only while that episode is unfinished, so a
// late call cannot overwrite a finished record.
func (s *Store) Progress(id string, events []agent.Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	episodeID, ok := s.running[id]
	if !ok {
		return nil
	}
	body, err := json.Marshal(events)
	if err != nil {
		return fmt.Errorf("episode %s: %w", episodeID, err)
	}
	_, err = s.db.Exec(
		"UPDATE episodes SET events = ? WHERE id = ? AND finish_reason IS NULL", string(body), episodeID,
	)
	if err != nil {
		return fmt.Errorf("checkpointing episode %s: %w", episodeID, err)
	}
	return nil
}

// End records a finished episode and releases the session. The session is released
// even if the write fails: the episode then reads back as interrupted, and the
// conversation can carry on without it.
func (s *Store) End(id string, episode agent.Episode) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	episodeID, ok := s.running[id]
	if !ok {
		return nil
	}
	delete(s.running, id)

	episode.ID = episodeID
	body, err := json.Marshal(episode)
	if err != nil {
		return fmt.Errorf("episode %s: %w", episodeID, err)
	}
	_, err = s.db.Exec(
		// The checkpoint goes: body is the record now.
		"UPDATE episodes SET finished_at = ?, finish_reason = ?, body = ?, events = NULL WHERE id = ?",
		stamp(time.Now()), episode.FinishReason, string(body), episodeID,
	)
	if err != nil {
		return fmt.Errorf("recording episode %s: %w", episodeID, err)
	}
	return nil
}
