import { useEffect, useState } from "react";
import { getHealth } from "./api/client";
import type { Health } from "./api/types";

type View = "chat" | "training";

function viewFromHash(): View {
  return window.location.hash.startsWith("#/training") ? "training" : "chat";
}

export function App() {
  const [view, setView] = useState<View>(viewFromHash);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Forge</span>
        <nav>
          <a href="#/chat" className={view === "chat" ? "active" : ""}>
            Chat
          </a>
          <a href="#/training" className={view === "training" ? "active" : ""}>
            Training
          </a>
        </nav>
        <HealthBadge />
      </header>
      <main className="content">
        {view === "chat" ? <p className="muted">Chat goes here.</p> : <p className="muted">Training dashboard goes here.</p>}
      </main>
    </div>
  );
}

/** Polls /health so a backend that went away is visible before a message fails. */
function HealthBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const check = () =>
      getHealth()
        .then((h) => {
          if (alive) {
            setHealth(h);
            setError(null);
          }
        })
        .catch((e: unknown) => {
          if (alive) setError(e instanceof Error ? e.message : String(e));
        });
    check();
    const timer = window.setInterval(check, 10_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  if (error) {
    return <span className="badge bad" title={error}>middle layer unreachable</span>;
  }
  if (!health) {
    return <span className="badge">checking…</span>;
  }
  const backendOk = health.backend === "ok";
  const sandbox = health.tools.sandbox;
  return (
    <span className="health">
      <span className={`badge ${backendOk ? "good" : "bad"}`} title={health.backend}>
        backend {backendOk ? "up" : "down"}
      </span>
      <span className={`badge ${sandbox.filesystem_confined ? "good" : "warn"}`} title={sandbox.confinement ?? ""}>
        sandbox {sandbox.backend ?? "?"}
        {sandbox.filesystem_confined ? "" : " (filesystem unconfined)"}
      </span>
    </span>
  );
}
