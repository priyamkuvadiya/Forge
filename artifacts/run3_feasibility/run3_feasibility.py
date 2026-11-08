"""run3 feasibility: is multi-hop QA blocked by retrieval, by reading, or by chaining?

Retrieval (CPU): does searching the subject return the first-hop doc, does the
second-hop doc's title appear in the first doc (so a policy could read the
bridge off it), and does searching that title return the second doc?

Oracle (GPU, through the running backend): the conversation is handed a perfect
chain, two real searches with real results (subject, then the bridge title), and
the policy only has to answer. Scored with the real verify(). If this is near 0,
a reward that teaches chaining cannot move QA.
"""

import json
import sys
import urllib.request
from collections import defaultdict

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline_agent import render_for_policy  # noqa: E402
from task_suite import verify  # noqa: E402
from task_suite.registry import load_suite  # noqa: E402
from tools import build_registry  # noqa: E402
from tools.search import load_index, render_hits  # noqa: E402
from tools.toolbox import ToolCall, ToolResult  # noqa: E402

BACKEND = "http://127.0.0.1:8000"
index = load_index()
registry = build_registry(include_code=False)


def docs_of(task):
    gt = task.ground_truth
    return gt.get("supporting_docs") or task.metadata["supporting_docs"]


def search_result(query):
    return render_for_policy(ToolResult(tool="search", ok=True, output=render_hits(index.search(query))))


def turn(policy, messages):
    body = json.dumps({"policy": policy, "messages": messages}).encode()
    req = urllib.request.Request(f"{BACKEND}/turn", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def oracle_episode(policy, task):
    docs = docs_of(task)
    queries = [task.metadata["subject"]] + [index.get(d).title for d in docs[1:]]
    messages = [{"role": "user", "content": task.prompt}]
    for q in queries:
        messages += [
            {"role": "assistant", "content": f'<tool name="search">{q}</tool>'},
            {"role": "user", "content": search_result(q)},
        ]
    calls = 0
    for _ in range(4):
        out = turn(policy, messages)
        messages.append({"role": "assistant", "content": out["text"]})
        if out["kind"] == "tool_call" and calls < 6:
            calls += 1
            result = registry.invoke(ToolCall(name=out["tool_call"]["name"], arguments=out["tool_call"]["arguments"]))
            messages.append({"role": "user", "content": render_for_policy(result)})
            continue
        return verify(task, out["text"]), out["kind"], calls
    return 0.0, "turns", calls


def retrieval(tasks):
    rows = defaultdict(lambda: [0, 0, 0, 0, 0])
    for t in tasks:
        docs = docs_of(t)
        r = rows[(t.split, t.category, len(docs))]
        r[0] += 1
        r[1] += docs[0] in [h.doc_id for h in index.search(t.metadata["subject"])]
        r[2] += docs[0] in [h.doc_id for h in index.search(t.prompt)]
        # every later hop: its title is written in an earlier doc, and searching it finds it
        seen = index.get(docs[0]).text.lower()
        bridge_ok = found_ok = True
        for d in docs[1:]:
            title = index.get(d).title
            bridge_ok &= title.lower() in seen
            found_ok &= d in [h.doc_id for h in index.search(title)]
            seen += " " + index.get(d).text.lower()
        r[3] += bridge_ok
        r[4] += found_ok
    print("split    category  hops  n  d1|subject d1|question bridge-named later|title")
    for (s, c, h), r in sorted(rows.items()):
        print(f"{s:8} {c:10} {h:3} {r[0]:3} {r[1]:10} {r[2]:11} {r[3]:13} {r[4]:11}")


def main():
    suite = load_suite()
    tasks = [t for s in ("train", "heldout") for t in suite[s] if t.category in ("qa", "multi_tool")]
    retrieval(tasks)
    if "--retrieval-only" in sys.argv:
        return
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 2
    policies = sys.argv[sys.argv.index("--policies") + 1].split(",")
    results = {}
    for policy in policies:
        for t in tasks:
            for i in range(k):
                score, kind, calls = oracle_episode(policy, t)
                results.setdefault(policy, []).append(
                    {"task": t.task_id, "split": t.split, "category": t.category, "score": score, "kind": kind, "calls": calls}
                )
            print(policy, t.task_id, [r["score"] for r in results[policy][-k:]], flush=True)
    json.dump(results, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
    for policy, rs in results.items():
        agg = defaultdict(list)
        for r in rs:
            agg[(r["split"], r["category"])].append(r["score"])
        for key, v in sorted(agg.items()):
            per_task = defaultdict(list)
            for r in rs:
                if (r["split"], r["category"]) == key:
                    per_task[r["task"]].append(r["score"])
            mixed = sum(1 for s in per_task.values() if 0 < sum(s) < len(s))
            print(f"{policy:8} {key[0]:8} {key[1]:10} mean {sum(v)/len(v):.3f}  tasks with mixed outcomes {mixed}/{len(per_task)}")


if __name__ == "__main__":
    main()
