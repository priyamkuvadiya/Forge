import type {
  Episode,
  Health,
  PoliciesResponse,
  Run,
  RunSummary,
  Session,
  SessionSummary,
} from "./types";

/** Every request goes through Vite's proxy (vite.config.ts), never cross-origin. */
export const API_BASE = "/api";

/** A non-2xx response. The message is the server's `{"error": ...}` when it sent one. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorFrom(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "error" in body && typeof body.error === "string") {
      message = body.error;
    }
  } catch {
    // Not JSON: a proxy error page, say. The status line is all there is.
  }
  return new ApiError(response.status, message);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, init);
  if (!response.ok) {
    throw await errorFrom(response);
  }
  return (await response.json()) as T;
}

/** POSTs must be JSON: the middle layer refuses anything else (see its guard). */
function post(body: unknown, extra?: RequestInit): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    ...extra,
  };
}

/** /health answers 503 with a body when the backend is down, which is still a report. */
export async function getHealth(): Promise<Health> {
  const response = await fetch(API_BASE + "/health");
  if (response.ok || response.status === 503) {
    return (await response.json()) as Health;
  }
  throw await errorFrom(response);
}

export const getPolicies = () => request<PoliciesResponse>("/policies");

export const listSessions = () =>
  request<{ sessions: SessionSummary[] }>("/sessions").then((r) => r.sessions);

export const createSession = (policy: string) => request<Session>("/sessions", post({ policy }));

export const getSession = (id: string) => request<Session>(`/sessions/${encodeURIComponent(id)}`);

/** Waits for the whole episode. The live view uses streamMessage instead. */
export const sendMessage = (id: string, content: string) =>
  request<Episode>(`/sessions/${encodeURIComponent(id)}/messages`, post({ content }));

export const listRuns = () => request<{ runs: RunSummary[] }>("/runs").then((r) => r.runs);

export const getRun = (name: string) => request<Run>(`/runs/${encodeURIComponent(name)}`);
