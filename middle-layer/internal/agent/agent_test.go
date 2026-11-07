package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// These use fakes and check the loop's decisions. Whether those decisions match
// the evaluation's is checked against the real Python loop, in
// tests/test_middle_layer_parity.py.

var limits = Limits{MaxCalls: 2, MaxNudges: 1, NudgeMessage: "NUDGE"}

type scriptedPolicy struct {
	turns []Turn
	err   map[int]error
	seen  [][]Message
}

func (p *scriptedPolicy) Turn(_ context.Context, _ string, messages []Message) (Turn, error) {
	index := len(p.seen)
	p.seen = append(p.seen, append([]Message{}, messages...))
	if err := p.err[index]; err != nil {
		return Turn{}, err
	}
	if index >= len(p.turns) {
		return Turn{Kind: KindNoAnswer, Text: "out of script"}, nil
	}
	return p.turns[index], nil
}

type fakeTools struct {
	openErr, invokeErr, finishErr error
	calls                         []string
	refuseAfter                   int
	response                      string
	finishCtxErr                  error
	finishes                      int
}

func (t *fakeTools) Open(context.Context, string, int) error { return t.openErr }

func (t *fakeTools) Invoke(_ context.Context, _, name, arguments string) (Invocation, error) {
	if t.invokeErr != nil {
		return Invocation{}, t.invokeErr
	}
	t.calls = append(t.calls, name)
	result := Invocation{Tool: name, OK: true, Message: "RESULT:" + arguments}
	if t.refuseAfter == 0 || len(t.calls) <= t.refuseAfter {
		result.Trace = &TraceEntry{Index: len(t.calls) - 1, Tool: name, Arguments: arguments, OK: true}
	}
	return result, nil
}

func (t *fakeTools) Finish(ctx context.Context, _, response string) (Finished, error) {
	t.finishes++
	t.response = response
	t.finishCtxErr = ctx.Err()
	if t.finishErr != nil {
		return Finished{}, t.finishErr
	}
	answer := "from-worker"
	return Finished{Answer: &answer, Trace: []TraceEntry{{Tool: "authoritative"}}}, nil
}

func call(name, arguments string) Turn {
	return Turn{
		Kind:     KindToolCall,
		Text:     `<tool name="` + name + `">` + arguments + `</tool>`,
		ToolCall: &ToolCall{Name: name, Arguments: arguments},
	}
}

func answer(text string) Turn {
	return Turn{Kind: KindAnswer, Text: "<answer>" + text + "</answer>", Answer: &text}
}

func prose(text string) Turn { return Turn{Kind: KindNoAnswer, Text: text} }

func play(policy *scriptedPolicy, tools *fakeTools, history ...Message) Episode {
	return Runner{Policy: policy, Tools: tools, Limits: limits}.
		Run(context.Background(), "e", "run2", history, "task", nil)
}

func roles(messages []Message) string {
	parts := make([]string, len(messages))
	for i, message := range messages {
		parts[i] = message.Role[:1]
	}
	return strings.Join(parts, "")
}

func TestToolResultGoesBackAndAnswerEnds(t *testing.T) {
	policy := &scriptedPolicy{turns: []Turn{call("search", "q"), answer("4")}}
	tools := &fakeTools{}
	episode := play(policy, tools)

	if episode.FinishReason != FinishAnswer || episode.Failed() {
		t.Fatalf("finish = %q", episode.FinishReason)
	}
	want := []Message{
		{"user", "task"},
		{"assistant", `<tool name="search">q</tool>`},
		{"user", "RESULT:q"},
		{"assistant", "<answer>4</answer>"},
	}
	if !reflect.DeepEqual(episode.Messages, want) {
		t.Fatalf("messages = %#v", episode.Messages)
	}
	if !reflect.DeepEqual(policy.seen[1], want[:3]) {
		t.Fatalf("second turn saw %#v", policy.seen[1])
	}
}

func TestTheAnswerAndTraceComeFromTheWorker(t *testing.T) {
	// The worker reads the answer across all turns, as the evaluation does. A
	// turn's own answer is not the record.
	tools := &fakeTools{}
	episode := play(&scriptedPolicy{turns: []Turn{call("search", "q"), answer("4")}}, tools)
	if episode.Answer == nil || *episode.Answer != "from-worker" {
		t.Fatalf("answer = %v", episode.Answer)
	}
	if len(episode.Trace) != 1 || episode.Trace[0].Tool != "authoritative" {
		t.Fatalf("trace = %#v", episode.Trace)
	}
	// Every assistant turn joined with one newline, which is `Episode.response`.
	if want := `<tool name="search">q</tool>` + "\n<answer>4</answer>"; tools.response != want {
		t.Fatalf("response = %q", tools.response)
	}
}

func TestOneNudgeThenGiveUp(t *testing.T) {
	policy := &scriptedPolicy{turns: []Turn{prose("hm"), prose("still no")}}
	episode := play(policy, &fakeTools{})
	if episode.FinishReason != FinishNoAnswer || episode.Nudges != 1 {
		t.Fatalf("finish = %q, nudges = %d", episode.FinishReason, episode.Nudges)
	}
	if len(policy.seen) != 2 {
		t.Fatalf("turns = %d, want 2", len(policy.seen))
	}
	if got := episode.Messages[2]; got != (Message{"user", "NUDGE"}) {
		t.Fatalf("nudge message = %#v", got)
	}
}

func TestANudgeCanStillBeAnswered(t *testing.T) {
	episode := play(&scriptedPolicy{turns: []Turn{prose("hm"), answer("4")}}, &fakeTools{})
	if episode.FinishReason != FinishAnswer || episode.Nudges != 1 {
		t.Fatalf("finish = %q, nudges = %d", episode.FinishReason, episode.Nudges)
	}
}

func TestEmptyTurnEndsWithNothingAppended(t *testing.T) {
	tools := &fakeTools{}
	episode := play(&scriptedPolicy{turns: []Turn{{Kind: KindEmpty}}}, tools)
	if episode.FinishReason != FinishEmpty {
		t.Fatalf("finish = %q", episode.FinishReason)
	}
	if roles(episode.Messages) != "u" || tools.response != "" {
		t.Fatalf("messages = %q, response = %q", roles(episode.Messages), tools.response)
	}
}

func TestTurnLimitIsCallsPlusNudgesPlusTwo(t *testing.T) {
	// With limits {2 calls, 1 nudge}: 5 turns, the same formula as run_episodes.
	policy := &scriptedPolicy{}
	for range 20 {
		policy.turns = append(policy.turns, call("search", "again"))
	}
	episode := play(policy, &fakeTools{refuseAfter: 2})
	if episode.FinishReason != FinishTurnLimit {
		t.Fatalf("finish = %q", episode.FinishReason)
	}
	if len(policy.seen) != 5 {
		t.Fatalf("turns = %d, want 5", len(policy.seen))
	}
}

func TestHistoryIsShownButNotRecorded(t *testing.T) {
	history := []Message{{"user", "earlier"}, {"assistant", "<answer>1</answer>"}}
	policy := &scriptedPolicy{turns: []Turn{answer("4")}}
	episode := play(policy, &fakeTools{}, history...)
	if roles(policy.seen[0]) != "uau" || policy.seen[0][0].Content != "earlier" {
		t.Fatalf("policy saw %#v", policy.seen[0])
	}
	if roles(episode.Messages) != "ua" || episode.Messages[0].Content != "task" {
		t.Fatalf("recorded %#v", episode.Messages)
	}
}

func TestSystemFailuresAreNeverShownToThePolicy(t *testing.T) {
	boom := errors.New("boom")
	cases := []struct {
		name   string
		policy *scriptedPolicy
		tools  *fakeTools
		want   string
		turns  int
	}{
		{
			"backend fails on the second turn",
			&scriptedPolicy{turns: []Turn{call("search", "q")}, err: map[int]error{1: boom}},
			&fakeTools{}, FinishBackendError, 2,
		},
		{
			"worker dies during a call",
			&scriptedPolicy{turns: []Turn{call("search", "q")}},
			&fakeTools{invokeErr: boom}, FinishToolError, 1,
		},
		{
			"worker cannot open the episode",
			&scriptedPolicy{},
			&fakeTools{openErr: boom}, FinishToolError, 0,
		},
		{
			"the answer cannot be read",
			&scriptedPolicy{turns: []Turn{answer("4")}},
			&fakeTools{finishErr: boom}, FinishToolError, 1,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			episode := play(c.policy, c.tools)
			if episode.FinishReason != c.want || !episode.Failed() {
				t.Fatalf("finish = %q, failed = %v", episode.FinishReason, episode.Failed())
			}
			if episode.Answer != nil {
				t.Fatalf("a failed episode reported answer %q", *episode.Answer)
			}
			if !strings.Contains(episode.Error, "boom") {
				t.Fatalf("error = %q", episode.Error)
			}
			if len(c.policy.seen) != c.turns {
				t.Fatalf("policy ran %d turns, want %d", len(c.policy.seen), c.turns)
			}
			for _, seen := range c.policy.seen {
				for _, message := range seen {
					if strings.Contains(message.Content, "boom") {
						t.Fatalf("the policy was shown the failure: %q", message.Content)
					}
				}
			}
		})
	}
}

func TestAFailedEpisodeKeepsTheCallsThatRan(t *testing.T) {
	policy := &scriptedPolicy{
		turns: []Turn{call("search", "q")},
		err:   map[int]error{1: errors.New("boom")},
	}
	episode := play(policy, &fakeTools{})
	if len(episode.Trace) != 1 || episode.Trace[0].Arguments != "q" {
		t.Fatalf("trace = %#v", episode.Trace)
	}
}

func TestCancellationIsNotReportedAsAnOutageAndStillClosesTheEpisode(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	policy := &scriptedPolicy{err: map[int]error{0: context.Canceled}}
	tools := &fakeTools{}
	episode := Runner{Policy: policy, Tools: tools, Limits: limits}.Run(ctx, "e", "p", nil, "task", nil)

	if episode.FinishReason != FinishCancelled {
		t.Fatalf("finish = %q", episode.FinishReason)
	}
	if tools.finishes != 1 {
		t.Fatalf("finish called %d times; the worker would keep the session", tools.finishes)
	}
	// With the request's context this call could never reach the worker.
	if tools.finishCtxErr != nil {
		t.Fatalf("closing used a cancelled context: %v", tools.finishCtxErr)
	}
}

func TestEventsFollowTheEpisodeInOrder(t *testing.T) {
	var events []Event
	policy := &scriptedPolicy{turns: []Turn{prose("hm"), call("search", "q"), answer("4")}}
	Runner{Policy: policy, Tools: &fakeTools{}, Limits: limits}.Run(
		context.Background(), "e", "p", nil, "task",
		func(event Event) { events = append(events, event) },
	)

	var types []string
	for _, event := range events {
		types = append(types, event.Type)
	}
	want := []string{
		EventStarted, EventTurn, EventNudge, EventTurn, EventToolResult, EventTurn, EventFinished,
	}
	if !reflect.DeepEqual(types, want) {
		t.Fatalf("events = %v", types)
	}
	if call := events[3].ToolCall; call == nil || call.Name != "search" {
		t.Fatalf("turn event = %#v", events[3])
	}
	if result := events[4]; result.Message != "RESULT:q" || result.OK == nil || !*result.OK {
		t.Fatalf("result event = %#v", result)
	}
	if last := events[6]; last.Finished == nil || last.Finished.FinishReason != FinishAnswer {
		t.Fatalf("last event = %#v", last)
	}
}

func TestARefusedCallIsMarkedAsNotHavingRun(t *testing.T) {
	var refused *Event
	policy := &scriptedPolicy{turns: []Turn{call("a", "1"), answer("4")}}
	tools := &fakeTools{refuseAfter: -1} // No call is ever traced.
	Runner{Policy: policy, Tools: tools, Limits: limits}.Run(
		context.Background(), "e", "p", nil, "task",
		func(event Event) {
			if event.Type == EventToolResult {
				refused = &event
			}
		},
	)
	if refused == nil || refused.Ran == nil || *refused.Ran {
		t.Fatalf("event = %#v", refused)
	}
}
