"""Print a few oracle conversations' documents and run2's answer, for reading."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run3_feasibility as f
from task_suite import verify

suite = f.load_suite()
ids = sys.argv[1].split(",")
tasks = {t.task_id: t for s in suite.values() for t in s}
for tid in ids:
    t = tasks[tid]
    docs = f.docs_of(t)
    queries = [t.metadata["subject"]] + [f.index.get(d).title for d in docs[1:]]
    msgs = [{"role": "user", "content": t.prompt}]
    for q in queries:
        msgs += [{"role": "assistant", "content": f'<tool name="search">{q}</tool>'},
                 {"role": "user", "content": f.search_result(q)}]
    print("=" * 80)
    print(tid, "|", t.prompt, "| GT", t.ground_truth.get("answers") or t.ground_truth.get("value"), "| docs", docs)
    for q in queries:
        print("  query", repr(q), "->", [h.doc_id for h in f.index.search(q)])
    for d in docs:
        print("  ", d, f.index.get(d).title, "::", f.index.get(d).text[:300])
    out = f.turn("run2", msgs)
    print("  RUN2:", out["kind"], repr(out["text"][:300]), "score", verify(t, out["text"]))
