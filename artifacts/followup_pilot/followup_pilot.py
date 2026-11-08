"""Pilot: does a held-out task score differently as a follow-up than as a first message?

Through the real middle layer. For each target task t, k fresh sessions where t is
the first message (the evaluated setting), and k sessions where a held-out task of
another category comes first and t follows. Paired by task.
"""
import json
import sys
import urllib.request
from collections import defaultdict

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from task_suite import verify  # noqa: E402
from task_suite.registry import load_suite  # noqa: E402

URL = "http://127.0.0.1:8080/api"
CATS = ["math", "code", "qa", "multi_tool", "no_tool"]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def say(policy, messages):
    sid = post("/sessions", {"policy": policy})["id"]
    return [post(f"/sessions/{sid}/messages", {"content": m}) for m in messages]


from tools import code_exec  # noqa: E402

RUNNER = code_exec.SandboxedCodeRunner()


def score(task, episode):
    # As the evaluation does (Episode.response): every assistant turn of this
    # episode joined, i.e. everything after the task's own prompt.
    msgs = episode["messages"]
    start = max(i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"] == task.prompt)
    turns = [m["content"] for m in msgs[start + 1:] if m["role"] == "assistant"]
    return verify(task, "\n".join(turns), code_runner=RUNNER)


def main():
    policy, per_cat, k, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    held = load_suite()["heldout"]
    by = {c: [t for t in held if t.category == c] for c in CATS}
    rows = []
    for ci, c in enumerate(CATS):
        for i in range(per_cat):
            t = by[c][i * (len(by[c]) // per_cat)]
            u = by[CATS[(ci + 1) % 5]][i]
            for _ in range(k):
                first = say(policy, [t.prompt])[0]
                follow = say(policy, [u.prompt, t.prompt])[1]
                rows.append({"task": t.task_id, "cat": c, "before": u.task_id,
                             "first": score(t, first), "follow": score(t, follow),
                             "first_finish": first["finish_reason"], "follow_finish": follow["finish_reason"],
                             "first_calls": len(first.get("trace") or []), "follow_calls": len(follow.get("trace") or [])})
                print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(out, "w"), indent=1)
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])
    for r in rows:
        for key in (r["cat"], "all"):
            a = agg[key]
            a[0] += r["first"]; a[1] += r["follow"]; a[2] += 1; a[3] += r["first_calls"]; a[4] += r["follow_calls"]
    for key, a in agg.items():
        print(f"{key:10} n={a[2]:3} first {a[0]/a[2]:.3f} follow-up {a[1]/a[2]:.3f}  calls/ep {a[3]/a[2]:.2f} -> {a[4]/a[2]:.2f}")


if __name__ == "__main__":
    main()
