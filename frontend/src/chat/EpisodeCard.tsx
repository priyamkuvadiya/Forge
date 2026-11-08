import { Fragment } from "react";
import { systemFailed } from "../api/types";
import { describeFinish, type EpisodeView, type Setting, type Step } from "./steps";

interface Props {
  view: EpisodeView;
  setting: Setting;
  live: boolean;
  onCancel?: () => void;
  cancelling?: boolean;
}

export function EpisodeCard({ view, setting, live, onCancel, cancelling }: Props) {
  const failed = view.finish !== null && systemFailed(view.finish);
  return (
    <article className={`episode${failed ? " failed" : ""}`}>
      <div className="episode-head">
        {setting === "evaluated" ? (
          <span className="tag eval" title="A single task with no prior conversation: what the held-out evaluation ran.">
            evaluated setting
          </span>
        ) : (
          <span
            className="tag followup"
            title="The policy sees the earlier conversation here. The evaluation only ever ran single tasks, so no published number covers this."
          >
            follow-up: outside what was evaluated
          </span>
        )}
        {live && <span className="tag running">running</span>}
        {live && onCancel && (
          <button className="small" onClick={onCancel} disabled={cancelling}>
            {cancelling ? "cancelling…" : "cancel"}
          </button>
        )}
      </div>

      <div className="bubble user">{view.prompt}</div>

      {view.steps.map((step, index) => (
        <StepView key={index} step={step} />
      ))}

      {live && <div className="thinking">generating…</div>}

      {view.finish !== null && (
        <footer className="episode-foot">
          {view.finish === "answer" && view.answer !== null && (
            <div className="final-answer">
              <span className="muted">answer</span> <strong>{view.answer}</strong>
            </div>
          )}
          <span className={`finish ${failed ? "bad" : view.finish === "answer" ? "good" : "warn"}`}>
            {describeFinish(view.finish)}
          </span>
          {view.nudges > 0 && <span className="muted"> · nudged {view.nudges}×</span>}
          {failed && (
            <div className="muted small-text">
              {view.error}
              <br />
              {view.finish === "cancelled" ? "Stopped before it finished" : "A system failure, not the policy's"}: it is
              kept here but never shown to the policy in later turns.
            </div>
          )}
        </footer>
      )}
    </article>
  );
}

function StepView({ step }: { step: Step }) {
  switch (step.kind) {
    case "assistant":
      return (
        <div className="bubble assistant">
          <Marked text={step.text} />
          {step.latencyMs !== null && <span className="latency">{(step.latencyMs / 1000).toFixed(1)} s</span>}
        </div>
      );
    case "tool":
      return <ToolStep step={step} />;
    case "nudge":
      return (
        <div className="nudge" title={step.text}>
          nudge: the loop asked for a final answer
        </div>
      );
  }
}

function ToolStep({ step }: { step: Extract<Step, { kind: "tool" }> }) {
  const trace = step.trace;
  const status = !step.ran ? "refused" : trace?.ok ? "ok" : "failed";
  return (
    <div className={`tool-card ${status}`}>
      <div className="tool-head">
        <span className="tool-name">{step.tool}</span>
        <span className={`tool-status ${status}`}>
          {status === "refused" ? "not run: call budget spent" : status === "ok" ? "ok" : "error"}
        </span>
        {trace && (
          <span className="muted">
            {trace.duration_ms < 1 ? "<1" : trace.duration_ms.toFixed(0)} ms
          </span>
        )}
      </div>
      {trace && (
        <div className="tool-args">
          <span className="muted">input</span>
          <pre>{trace.arguments}</pre>
        </div>
      )}
      {trace?.error && <div className="tool-error">{trace.error}</div>}
      <details open={step.shown.length < 600}>
        <summary className="muted">what the policy was shown ({step.shown.length} chars)</summary>
        <pre>{step.shown}</pre>
      </details>
    </div>
  );
}

const MARKS = /(<tool name="[^"]*">[\s\S]*?<\/tool>|<answer>[\s\S]*?<\/answer>)/g;

/** Shows the policy's raw text, with its tool calls and answer picked out. */
function Marked({ text }: { text: string }) {
  const parts = text.split(MARKS);
  return (
    <span className="raw">
      {parts.map((part, index) => {
        if (index % 2 === 0) return <Fragment key={index}>{part}</Fragment>;
        const kind = part.startsWith("<answer>") ? "mark-answer" : "mark-tool";
        return (
          <code key={index} className={kind}>
            {part}
          </code>
        );
      })}
    </span>
  );
}
