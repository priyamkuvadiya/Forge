import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, createSession, getPolicies, getSession, listSessions } from "../api/client";
import { cancelEpisode, followLive, streamMessage, type StreamOutcome } from "../api/stream";
import type { AgentEvent, PolicyInfo, Session, SessionSummary } from "../api/types";
import { labelFor, orderPolicies, TRAINED } from "../policies";
import { EpisodeCard } from "./EpisodeCard";
import { applyEvent, emptyView, fromEpisode, settingAt, type EpisodeView } from "./steps";

interface Props {
  sessionId: string | null;
  onSelect: (id: string | null) => void;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function ChatView({ sessionId, onSelect }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [policies, setPolicies] = useState<PolicyInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshList = useCallback(() => {
    listSessions()
      .then((list) => setSessions([...list].reverse())) // Newest first.
      .catch((e: unknown) => setError(message(e)));
  }, []);

  useEffect(() => {
    refreshList();
    getPolicies()
      .then((p) => setPolicies(orderPolicies(p.policies)))
      .catch((e: unknown) => setError(`policies: ${message(e)}`));
  }, [refreshList]);

  const start = async (policy: string) => {
    try {
      const created = await createSession(policy);
      refreshList();
      onSelect(created.id);
    } catch (e) {
      setError(message(e));
    }
  };

  return (
    <div className="chat">
      <aside className="sidebar">
        <NewSession policies={policies} onStart={start} />
        <h3>Sessions</h3>
        {sessions.length === 0 && <p className="muted">None yet.</p>}
        <ul className="session-list">
          {sessions.map((s) => (
            <li key={s.id}>
              <button className={s.id === sessionId ? "selected" : ""} onClick={() => onSelect(s.id)}>
                <span className="session-policy">{labelFor(s.policy).title}</span>
                <span className="muted">
                  {new Date(s.created_at).toLocaleString()} · {s.episodes} msg
                  {s.running && <span className="dot" title="an episode is running" />}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section className="conversation">
        {error && (
          <div className="error-banner" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        {sessionId ? (
          <Conversation key={sessionId} id={sessionId} onChanged={refreshList} onMissing={() => onSelect(null)} />
        ) : (
          <Intro />
        )}
      </section>
    </div>
  );
}

function Intro() {
  return (
    <div className="intro">
      <h2>Talk to a policy, and watch it use its tools</h2>
      <p>
        Start a session with one policy. Each turn the policy writes, each tool call it makes and exactly what the tool
        returned to it appear as they happen. The policy is fixed for a session: to compare two, open one session each
        and send the same message.
      </p>
      <p className="muted">
        The first message in a session is the setting the held-out evaluation measured: one task, no prior
        conversation. Later messages see the earlier ones, which no published number covers, and are labelled so.
      </p>
    </div>
  );
}

function NewSession({ policies, onStart }: { policies: PolicyInfo[]; onStart: (policy: string) => void }) {
  const [choice, setChoice] = useState<string>(TRAINED);
  const available = policies.some((p) => p.name === choice) ? choice : (policies[0]?.name ?? "");
  return (
    <div className="new-session">
      <h3>New session</h3>
      {policies.length === 0 && <p className="muted">Waiting for the backend's policies…</p>}
      {policies.map((p) => {
        const label = labelFor(p.name);
        return (
          <label key={p.name} className={`policy-option${available === p.name ? " chosen" : ""}`}>
            <input
              type="radio"
              name="policy"
              value={p.name}
              checked={available === p.name}
              onChange={() => setChoice(p.name)}
            />
            <span>
              <strong>{label.title}</strong>
              <span className="muted small-text">{label.note}</span>
            </span>
          </label>
        );
      })}
      <button className="primary" disabled={!available} onClick={() => onStart(available)}>
        Start
      </button>
    </div>
  );
}

interface ConversationProps {
  id: string;
  onChanged: () => void;
  onMissing: () => void;
}

function Conversation({ id, onChanged, onMissing }: ConversationProps) {
  const [session, setSession] = useState<Session | null>(null);
  const [live, setLive] = useState<EpisodeView | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);
  // The parent's callbacks change on every render of the parent. Held in a ref so
  // that nothing below depends on them: an effect that did would rerun, and its
  // cleanup aborts the stream being followed.
  const parent = useRef({ onChanged, onMissing });
  parent.current = { onChanged, onMissing };

  const reload = useCallback(async () => {
    try {
      const found = await getSession(id);
      setSession(found);
      return found;
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) parent.current.onMissing();
      else setError(message(e));
      return null;
    }
  }, [id]);

  const onEvent = useCallback((event: AgentEvent) => {
    setLive((view) => applyEvent(view ?? emptyView(""), event));
  }, []);

  // Follows the running episode until it ends. A dropped connection is retried
  // by reattaching: the episode is the middle layer's, so it never stopped.
  const settle = useCallback(
    async (first: () => Promise<StreamOutcome>) => {
      const controller = new AbortController();
      abort.current?.abort();
      abort.current = controller;
      let outcome: StreamOutcome;
      try {
        outcome = await first();
        for (let attempt = 0; outcome === "dropped" && !controller.signal.aborted && attempt < 5; attempt++) {
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
          if (controller.signal.aborted) return;
          setLive(emptyView("")); // /live replays from the first event.
          outcome = await followLive(id, onEvent, controller.signal);
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(message(e));
      }
      if (controller.signal.aborted) return;
      // The record and the end of the live view in one render, so the episode
      // is never on screen twice or not at all.
      try {
        const found = await getSession(id);
        if (controller.signal.aborted) return;
        setSession(found);
      } catch (e) {
        setError(message(e));
      }
      setLive(null);
      setCancelling(false);
      parent.current.onChanged();
    },
    [id, onEvent],
  );

  // On opening a session, reattach if an episode is running in it: this is what
  // makes a refresh mid-episode lose nothing.
  useEffect(() => {
    let gone = false;
    reload().then((found) => {
      if (!gone && found?.running) {
        setLive(emptyView(""));
        void settle(() => followLive(id, onEvent, abort.current?.signal));
      }
    });
    return () => {
      gone = true;
      abort.current?.abort(); // Stops following only. The episode carries on.
    };
  }, [id, reload, settle, onEvent]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [live, session]);

  const send = () => {
    const content = draft.trim();
    if (!content || live) return;
    setDraft("");
    setError(null);
    setLive(emptyView(content));
    void settle(() => streamMessage(id, content, onEvent, abort.current?.signal));
    onChanged();
  };

  const cancel = async () => {
    setCancelling(true);
    try {
      await cancelEpisode(id);
    } catch (e) {
      setError(message(e));
      setCancelling(false);
    }
  };

  if (!session) {
    return <p className="muted">{error ?? "Loading…"}</p>;
  }

  const episodes = session.episodes;
  const label = labelFor(session.policy);

  return (
    <div className="thread">
      <header className="thread-head">
        <h2>{label.title}</h2>
        <span className="muted small-text">{label.note}</span>
      </header>
      {episodes.map((episode, index) => (
        <EpisodeCard key={episode.id} view={fromEpisode(episode)} setting={settingAt(episodes, index)} live={false} />
      ))}
      {live && (
        <EpisodeCard
          view={live}
          setting={settingAt(episodes, episodes.length)}
          live
          onCancel={cancel}
          cancelling={cancelling}
        />
      )}
      {error && (
        <div className="error-banner" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      <div ref={bottom} />
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          value={draft}
          placeholder={live ? "Wait for this episode to finish…" : "Ask something. Try arithmetic, code, or a question about the corpus."}
          disabled={live !== null}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={3}
        />
        <button className="primary" type="submit" disabled={live !== null || draft.trim() === ""}>
          Send
        </button>
      </form>
    </div>
  );
}
