// Package agent drives one episode: policy turn, tool call, tool result, and so
// on until the policy answers or runs out of room.
//
// This is the serving twin of `run_episodes` and `_step` in baseline_agent.py,
// and it has to stay that. The published numbers describe the policy inside that
// loop, so a difference here (one nudge too many, a tool result appended in a
// different form) would mean serving something other than what was measured.
// The rules that can live elsewhere do: the backend reads each completion and
// the tool worker renders each result, and this package only decides what
// happens next. eval/serving.py and tests/test_middle_layer_parity.py compare
// the two loops transcript by transcript.
package agent

import (
	"context"
	"errors"
	"strings"
	"time"
)

// What one completion amounted to, as read by the backend.
const (
	KindEmpty    = "empty"
	KindToolCall = "tool_call"
	KindAnswer   = "answer"
	KindNoAnswer = "no_answer"
)

// How an episode ended. The first four are the evaluation's own finish reasons.
// The others have no counterpart there: they mean the system failed rather than
// the policy, and an episode ending in one is not part of the conversation.
const (
	FinishAnswer    = "answer"
	FinishNoAnswer  = "no_answer"
	FinishEmpty     = "empty"
	FinishTurnLimit = "turn_limit"

	FinishBackendError = "backend_error"
	FinishToolError    = "tool_error"
	FinishCancelled    = "cancelled"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ToolCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Turn is one reply from the backend.
type Turn struct {
	Kind      string    `json:"kind"`
	Text      string    `json:"text"`
	ToolCall  *ToolCall `json:"tool_call"`
	Answer    *string   `json:"answer"`
	Raw       string    `json:"raw"`
	LatencyMS float64   `json:"latency_ms"`
}

// TraceEntry mirrors tools.toolbox.TraceEntry.
type TraceEntry struct {
	Index      int     `json:"index"`
	Tool       string  `json:"tool"`
	Arguments  string  `json:"arguments"`
	OK         bool    `json:"ok"`
	Output     string  `json:"output"`
	Error      *string `json:"error"`
	DurationMS float64 `json:"duration_ms"`
}

// Invocation is what one tool call came to. Message is the text to append to
// the conversation, already rendered and truncated. Trace is nil when the budget
// refused the call, since a refused call never ran.
type Invocation struct {
	Tool    string      `json:"tool"`
	OK      bool        `json:"tool_ok"`
	Output  string      `json:"output"`
	Error   *string     `json:"tool_error"`
	Message string      `json:"message"`
	Trace   *TraceEntry `json:"trace"`
}

// Finished is what the tool worker reports when an episode closes.
type Finished struct {
	Answer *string      `json:"answer"`
	Trace  []TraceEntry `json:"trace"`
}

type Policy interface {
	Turn(ctx context.Context, policy string, messages []Message) (Turn, error)
}

type Tools interface {
	Open(ctx context.Context, episode string, maxCalls int) error
	Invoke(ctx context.Context, episode, name, arguments string) (Invocation, error)
	Finish(ctx context.Context, episode, response string) (Finished, error)
}

// Limits are the values the loop must match, published by the backend at
// GET /contract so that they are not written down twice.
type Limits struct {
	MaxCalls     int
	MaxNudges    int
	NudgeMessage string
}

// MaxTurns matches the evaluation: every call, the turn that spends the
// budget-exhausted message, the nudges, and the turn that finally answers.
func (l Limits) MaxTurns() int { return l.MaxCalls + l.MaxNudges + 2 }

// Episode is the record of one user message and everything it caused.
type Episode struct {
	ID           string       `json:"id"`
	Policy       string       `json:"policy"`
	Messages     []Message    `json:"messages"`
	Nudges       int          `json:"nudges"`
	FinishReason string       `json:"finish_reason"`
	Answer       *string      `json:"answer"`
	Trace        []TraceEntry `json:"trace"`
	Error        string       `json:"error,omitempty"`
}

// Failed reports whether the system, rather than the policy, ended the episode.
func (e Episode) Failed() bool {
	switch e.FinishReason {
	case FinishBackendError, FinishToolError, FinishCancelled:
		return true
	}
	return false
}

// Event is one step of an episode as it happens, for the live view. Unused
// fields are omitted, so each type carries only what it means.
type Event struct {
	Type      string    `json:"type"`
	Episode   string    `json:"episode"`
	Turn      int       `json:"turn"`
	Kind      string    `json:"kind,omitempty"`
	Text      string    `json:"text,omitempty"`
	ToolCall  *ToolCall `json:"tool_call,omitempty"`
	LatencyMS float64   `json:"latency_ms,omitempty"`
	// Tool result: Message is what the policy was shown. Ran is false when the
	// budget refused the call.
	Tool       string   `json:"tool,omitempty"`
	OK         *bool    `json:"ok,omitempty"`
	Message    string   `json:"message,omitempty"`
	ToolError  *string  `json:"tool_error,omitempty"`
	Ran        *bool    `json:"ran,omitempty"`
	DurationMS float64  `json:"duration_ms,omitempty"`
	Finished   *Episode `json:"finished,omitempty"`
}

const (
	EventStarted    = "episode_started"
	EventTurn       = "turn"
	EventToolResult = "tool_result"
	EventNudge      = "nudge"
	EventFinished   = "episode_finished"
)

// closeTimeout bounds releasing a worker session after the request has gone.
const closeTimeout = 10 * time.Second

type Runner struct {
	Policy Policy
	Tools  Tools
	Limits Limits
}

// Run plays one episode. It always returns a finished record: failures are
// reported in FinishReason rather than as an error, because what happened before
// the failure is still worth showing. emit may be nil.
//
// history is what came before this user message in the session. A policy was
// only ever evaluated on a single task, so anything after the first episode of a
// session is outside what the published numbers cover.
func (r Runner) Run(
	ctx context.Context,
	id, policy string,
	history []Message,
	prompt string,
	emit func(Event),
) Episode {
	if emit == nil {
		emit = func(Event) {}
	}
	episode := Episode{
		ID:       id,
		Policy:   policy,
		Messages: []Message{{Role: "user", Content: prompt}},
		Trace:    []TraceEntry{},
	}
	emit(Event{Type: EventStarted, Episode: id})

	var replies []string
	conversation := func() []Message {
		full := make([]Message, 0, len(history)+len(episode.Messages))
		return append(append(full, history...), episode.Messages...)
	}

	// finish closes the tool session whatever happened, or the worker keeps it.
	finish := func(reason string, failure error) Episode {
		episode.FinishReason = reason
		closing, cancel := context.WithTimeout(context.WithoutCancel(ctx), closeTimeout)
		defer cancel()
		finished, err := r.Tools.Finish(closing, id, strings.Join(replies, "\n"))
		switch {
		case failure != nil:
			// No answer on a failed episode, but the calls that ran did run.
			episode.Error = failure.Error()
		case err != nil:
			// The answer is read by the worker, so without it there is no
			// honest answer to report, whatever the policy wrote.
			episode.FinishReason = FinishToolError
			episode.Error = err.Error()
		default:
			episode.Answer = finished.Answer
			episode.Trace = finished.Trace
		}
		emit(Event{Type: EventFinished, Episode: id, Finished: &episode})
		return episode
	}
	// Failing before the worker has an episode to close.
	abort := func(reason string, failure error) Episode {
		episode.FinishReason = reason
		episode.Error = failure.Error()
		emit(Event{Type: EventFinished, Episode: id, Finished: &episode})
		return episode
	}

	if err := r.Tools.Open(ctx, id, r.Limits.MaxCalls); err != nil {
		return abort(failureReason(err, FinishToolError), err)
	}

	for turn := range r.Limits.MaxTurns() {
		reply, err := r.Policy.Turn(ctx, policy, conversation())
		if err != nil {
			return finish(failureReason(err, FinishBackendError), err)
		}
		emit(Event{
			Type:      EventTurn,
			Episode:   id,
			Turn:      turn,
			Kind:      reply.Kind,
			Text:      reply.Text,
			ToolCall:  reply.ToolCall,
			LatencyMS: reply.LatencyMS,
		})

		if reply.Kind == KindEmpty {
			// An immediate EOS. Nothing to append, nothing to nudge towards.
			return finish(FinishEmpty, nil)
		}

		replies = append(replies, reply.Text)
		episode.Messages = append(episode.Messages, Message{Role: "assistant", Content: reply.Text})

		switch reply.Kind {
		case KindToolCall:
			result, err := r.Tools.Invoke(ctx, id, reply.ToolCall.Name, reply.ToolCall.Arguments)
			if err != nil {
				return finish(failureReason(err, FinishToolError), err)
			}
			episode.Messages = append(episode.Messages, Message{Role: "user", Content: result.Message})
			// Collected as it goes, so it survives the worker dying later on.
			if result.Trace != nil {
				episode.Trace = append(episode.Trace, *result.Trace)
			}
			ran := result.Trace != nil
			event := Event{
				Type:      EventToolResult,
				Episode:   id,
				Turn:      turn,
				Tool:      result.Tool,
				OK:        &result.OK,
				Message:   result.Message,
				ToolError: result.Error,
				Ran:       &ran,
			}
			if result.Trace != nil {
				event.DurationMS = result.Trace.DurationMS
			}
			emit(event)
		case KindAnswer:
			return finish(FinishAnswer, nil)
		default:
			if episode.Nudges >= r.Limits.MaxNudges {
				return finish(FinishNoAnswer, nil)
			}
			episode.Nudges++
			episode.Messages = append(episode.Messages, Message{Role: "user", Content: r.Limits.NudgeMessage})
			emit(Event{Type: EventNudge, Episode: id, Turn: turn, Message: r.Limits.NudgeMessage})
		}
	}
	return finish(FinishTurnLimit, nil)
}

func failureReason(err error, otherwise string) string {
	if errors.Is(err, context.Canceled) {
		return FinishCancelled
	}
	return otherwise
}
