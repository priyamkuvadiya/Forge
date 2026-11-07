// Package backend is the client for the Python policy server (module 6).
//
// The backend owns the prompt prefix, the reading of a completion and the
// sampling settings. This client sends a conversation and passes back what it
// was told, and checks that the reply makes sense: a malformed turn must fail
// here rather than reach the loop looking like a policy decision.
package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
)

// A turn is at most 384 tokens plus its `raw` copy. Anything far beyond that is
// not a reply from this backend.
const maxReplyBytes = 4 << 20

type Client struct {
	BaseURL string
	// TurnTimeout bounds one generation, including time queued behind other
	// requests on the backend's GPU lock.
	TurnTimeout time.Duration
	HTTP        *http.Client
}

// New has no overall client timeout: each call sets its own deadline, because a
// generation and a health check should not share one.
func New(baseURL string, turnTimeout time.Duration) *Client {
	return &Client{
		BaseURL:     strings.TrimRight(baseURL, "/"),
		TurnTimeout: turnTimeout,
		HTTP:        &http.Client{},
	}
}

// Contract is GET /contract: what this layer's loop has to match.
type Contract struct {
	Tools              []string `json:"tools"`
	MaxCalls           int      `json:"max_calls"`
	MaxNudges          int      `json:"max_nudges"`
	NudgeMessage       string   `json:"nudge_message"`
	MaxToolOutputChars int      `json:"max_tool_output_chars"`
}

func (c *Client) do(ctx context.Context, method, path string, body any, out any) error {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		payload = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, payload)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := c.HTTP.Do(request)
	if err != nil {
		return fmt.Errorf("backend %s %s: %w", method, path, err)
	}
	defer response.Body.Close()

	data, err := io.ReadAll(io.LimitReader(response.Body, maxReplyBytes+1))
	if err != nil {
		return fmt.Errorf("backend %s %s: %w", method, path, err)
	}
	if len(data) > maxReplyBytes {
		return fmt.Errorf("backend %s %s: reply over %d bytes", method, path, maxReplyBytes)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("backend %s %s: status %d: %s", method, path, response.StatusCode, detail(data))
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("backend %s %s: bad reply: %w", method, path, err)
	}
	return nil
}

// detail pulls FastAPI's error message out of a body, if that is what it is.
func detail(data []byte) string {
	var parsed struct {
		Detail any `json:"detail"`
	}
	if json.Unmarshal(data, &parsed) == nil && parsed.Detail != nil {
		return fmt.Sprint(parsed.Detail)
	}
	return string(data)
}

// Turn asks for the policy's next turn. It is not retried: sampling is not
// idempotent, and a retry would silently draw a different completion.
func (c *Client) Turn(ctx context.Context, policy string, messages []agent.Message) (agent.Turn, error) {
	ctx, cancel := context.WithTimeout(ctx, c.TurnTimeout)
	defer cancel()

	var turn agent.Turn
	body := map[string]any{"policy": policy, "messages": messages}
	if err := c.do(ctx, http.MethodPost, "/turn", body, &turn); err != nil {
		return agent.Turn{}, err
	}
	switch turn.Kind {
	case agent.KindEmpty, agent.KindAnswer, agent.KindNoAnswer:
	case agent.KindToolCall:
		if turn.ToolCall == nil {
			return agent.Turn{}, fmt.Errorf("backend POST /turn: a tool_call turn with no call")
		}
	default:
		return agent.Turn{}, fmt.Errorf("backend POST /turn: unknown kind %q", turn.Kind)
	}
	return turn, nil
}

func (c *Client) Contract(ctx context.Context) (Contract, error) {
	var contract Contract
	err := c.do(ctx, http.MethodGet, "/contract", nil, &contract)
	return contract, err
}

// Policies returns the backend's description unchanged, for the frontend, along
// with the names in it.
func (c *Client) Policies(ctx context.Context) (json.RawMessage, []string, error) {
	var raw json.RawMessage
	if err := c.do(ctx, http.MethodGet, "/policies", nil, &raw); err != nil {
		return nil, nil, err
	}
	var parsed struct {
		Policies []struct {
			Name string `json:"name"`
		} `json:"policies"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, nil, fmt.Errorf("backend GET /policies: bad reply: %w", err)
	}
	names := make([]string, len(parsed.Policies))
	for i, policy := range parsed.Policies {
		names[i] = policy.Name
	}
	return raw, names, nil
}

// Health reports whether the backend is answering at all.
func (c *Client) Health(ctx context.Context) error {
	var ignored json.RawMessage
	return c.do(ctx, http.MethodGet, "/health", nil, &ignored)
}

// WaitReady polls until the backend answers. Loading the model takes a while and
// the port only opens once it is done, so being refused at first is normal.
func (c *Client) WaitReady(ctx context.Context, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for {
		attempt, stop := context.WithTimeout(ctx, 5*time.Second)
		err := c.Health(attempt)
		stop()
		if err == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("backend not ready after %s: %w", timeout, err)
		case <-time.After(time.Second):
		}
	}
}
