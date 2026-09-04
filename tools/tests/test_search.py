"""Tests for the search tool.

The safety surface here is nil — the tool reads a fixed local JSON file — so
these tests are about a different risk: that retrieval quietly caps or
trivialises the two categories that depend on it. Both failures are silent.
If the supporting documents can't be found, QA and multi_tool tasks are
unsolvable and the reward curve just looks flat. If a single query returns
every document a multi-hop question needs, the "multi-hop" claim is false and
the policy never has to learn to chain searches.
"""

import pytest

from task_suite.registry import load_suite
from tools import search as search_module
from tools.search import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    Document,
    SearchIndex,
    load_documents,
    load_index,
    render_hits,
    tokenize,
)


@pytest.fixture(scope="module")
def index() -> SearchIndex:
    return load_index()


@pytest.fixture(scope="module")
def tasks():
    splits = load_suite()
    return [task for split_tasks in splits.values() for task in split_tasks]


# --------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------

def test_index_covers_the_whole_corpus(index):
    assert len(index) == 130


def test_documents_have_content(index):
    for doc in index.documents:
        assert doc.doc_id and doc.title and doc.text.strip()


def test_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        SearchIndex([])


def test_rejects_duplicate_doc_ids():
    """A duplicated id would make `get()` ambiguous and silently drop a fact."""
    doc = Document(doc_id="d-00", title="A", text="alpha")
    with pytest.raises(ValueError):
        SearchIndex([doc, Document(doc_id="d-00", title="B", text="beta")])


def test_get_returns_the_document(index):
    doc = index.get("ins-00")
    assert doc.doc_id == "ins-00"
    assert doc.title


def test_get_raises_on_an_unknown_id(index):
    with pytest.raises(KeyError):
        index.get("does-not-exist")


# --------------------------------------------------------------------------
# Ranking behaviour
# --------------------------------------------------------------------------

def test_searching_a_title_returns_that_document_first(index):
    """Every document must be reachable by name.

    This is the retrieval floor: the QA chains work by learning an entity's
    name from one document and searching for it. If a name didn't retrieve its
    own document, the second hop would be impossible.
    """
    misses = [
        doc.doc_id
        for doc in index.documents
        if not index.search(doc.title, k=1) or index.search(doc.title, k=1)[0].doc_id != doc.doc_id
    ]
    assert misses == [], f"documents not retrievable by their own title: {misses}"


def test_empty_query_returns_nothing(index):
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("!!! ???") == []


def test_unmatched_query_returns_nothing_rather_than_filler(index):
    """A failed query has to look failed.

    Padding results out to `k` would hand the policy three irrelevant
    documents and no signal that its query was wrong.
    """
    assert index.search("zzzqqqxxx nonexistentterm") == []


def test_k_is_respected_and_capped(index):
    # "instrument" rather than "expedition": only four documents are titled
    # "... Expedition" (the rest are Surveys and Traverses), so that query
    # cannot fill a larger k and would test the corpus, not the cap.
    assert len(index.search("instrument", k=1)) == 1
    assert len(index.search("instrument", k=3)) == 3
    assert len(index.search("instrument", k=1000)) <= MAX_TOP_K


def test_k_must_be_positive(index):
    with pytest.raises(ValueError):
        index.search("expedition", k=0)


def test_results_are_ordered_by_descending_score(index):
    hits = index.search("expedition station instrument", k=MAX_TOP_K)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_search_is_deterministic(index):
    """Two identical queries must agree, or eval re-runs disagree with themselves."""
    query = "Which instrument was designed by the researcher who led the expedition?"
    assert index.search(query, k=MAX_TOP_K) == index.search(query, k=MAX_TOP_K)


def test_ties_are_broken_stably(index):
    """Documents in this corpus are near-identical in shape, so ties are common."""
    rebuilt = SearchIndex(load_documents())
    query = "field instrument dry mass routine service"
    assert [h.doc_id for h in index.search(query, k=MAX_TOP_K)] == [
        h.doc_id for h in rebuilt.search(query, k=MAX_TOP_K)
    ]


def test_query_is_case_insensitive(index):
    assert index.search("KESTREL ARRAY", k=1) == index.search("kestrel array", k=1)


def test_tokenizer_splits_on_punctuation():
    assert tokenize("Fenwold Station's 12 berths.") == ["fenwold", "station", "s", "12", "berths"]


# --------------------------------------------------------------------------
# Sufficiency for the QA and multi_tool categories
# --------------------------------------------------------------------------

def _first_hop_docs(task) -> list[str]:
    if task.category == "qa":
        return task.ground_truth["supporting_docs"]
    return task.metadata["supporting_docs"]


def test_the_first_hop_is_reachable_from_the_question_text(index, tasks):
    """Searching the raw question must surface at least one supporting document.

    This is the entry point to every retrieval chain. A policy's first query is
    realistically the question itself, or something close to it; if that
    returned nothing useful, the task would be unsolvable regardless of how
    well the policy reasoned.
    """
    retrieval_tasks = [t for t in tasks if t.category in ("qa", "multi_tool")]
    assert len(retrieval_tasks) == 118 + 50, "suite changed; revisit this test's coverage"

    failures = []
    for task in retrieval_tasks:
        hits = {hit.doc_id for hit in index.search(task.prompt, k=5)}
        if not hits & set(_first_hop_docs(task)):
            failures.append(task.task_id)

    assert failures == [], f"{len(failures)} tasks whose first hop is unreachable: {failures[:5]}"


def test_one_query_rarely_retrieves_a_whole_chain(index, tasks):
    """Pin how often a single search hands back a task's full supporting set.

    This is not zero, and pretending otherwise would be the dishonest version
    of this test. Every document in a class is rendered from one template, so
    a question mentioning "vessel" ranks every vessel document about equally
    and the right one can arrive incidentally, buried among its neighbours.

    It was 7 of 118 when the corpus had 68 documents. Enlarging the small
    classes with distractors brought it to 4. What remains is tolerable rather
    than fatal because no single *document* answers the question (asserted in
    `task_suite/tests/test_qa_world.py`), so the policy must still read the
    first document to learn which of the retrieved neighbours is the answer.
    What it saved was the second search *call*, not the reasoning.

    The number is asserted exactly. If it moves, the corpus or the ranking
    changed and someone needs to look at it rather than discover it later in a
    reward curve.
    """
    multi_hop = [
        t for t in tasks
        if t.category == "qa" and len(t.ground_truth["supporting_docs"]) > 1
    ]
    assert len(multi_hop) == 118, "suite changed; re-measure before adjusting the bound"

    shortcut = [
        task.task_id
        for task in multi_hop
        if set(task.ground_truth["supporting_docs"])
        <= {hit.doc_id for hit in index.search(task.prompt, k=MAX_TOP_K)}
    ]

    assert len(shortcut) == 4, (
        f"expected 4 single-query chains at k={MAX_TOP_K}, got {len(shortcut)}: {shortcut}"
    )


def test_raising_k_would_erode_the_multi_hop_property(index, tasks, monkeypatch):
    """The reason `MAX_TOP_K` is 5 and not 10, kept as an executable record.

    Without this, the cap looks arbitrary and the next person who wants more
    recall raises it without knowing what it costs. The cap itself is lifted
    for the comparison — otherwise `search` would clamp the very value being
    measured and the test would compare 5 against 5.
    """
    multi_hop = [
        t for t in tasks
        if t.category == "qa" and len(t.ground_truth["supporting_docs"]) > 1
    ]

    def shortcut_count(k: int) -> int:
        monkeypatch.setattr(search_module, "MAX_TOP_K", k)
        return sum(
            1 for task in multi_hop
            if set(task.ground_truth["supporting_docs"])
            <= {hit.doc_id for hit in index.search(task.prompt, k=k)}
        )

    at_cap = shortcut_count(MAX_TOP_K)
    at_ten = shortcut_count(10)
    assert (at_cap, at_ten) == (4, 14), f"ranking changed: k=5 gave {at_cap}, k=10 gave {at_ten}"


def test_ties_do_not_favour_low_numbered_documents(index, tasks):
    """Ranking must not correlate with a document's position in the corpus.

    These documents come from a few templates, so BM25 often cannot separate a
    class at all - every instrument document contains "instrument" and
    "measures" exactly once, leaving length as the only differentiator. Exact
    score ties therefore decide the top-k cut for most retrieval tasks, which
    makes the tie-break's bias the thing that actually determines what the
    agent reads.

    Ordering ties by doc_id, as this first did, ranks by entity index. Since
    the distractors are appended after the real entities, every tie went to a
    real document and enlarging the corpus bought far less than it appeared
    to. The hash tie-break removes that correlation, and this test is what
    keeps it removed.

    Measured on the tie itself rather than on all results. Counting every
    returned document would be confounded: the genuinely relevant ones are
    real entities, which are the low-numbered ones, so a correct ranking would
    look biased. What matters is only which member of a *tied group* gets the
    last slot.
    """
    retrieval = [t for t in tasks if t.category in ("qa", "multi_tool")]

    contested, smallest_won = 0, 0
    for task in retrieval:
        hits = index.search(task.prompt, k=MAX_TOP_K)
        if len(hits) < MAX_TOP_K:
            continue

        # Documents scoring exactly what the last returned hit scored: the
        # group the tie-break had to choose from. `_score` is private, but the
        # alternative is re-deriving BM25 in the test and asserting against a
        # second implementation rather than against this one.
        terms = tokenize(task.prompt)
        cutoff = hits[-1].score
        tied = [
            doc.doc_id
            for i, doc in enumerate(index.documents)
            if abs(index._score(i, terms) - cutoff) < 1e-12
        ]
        if len(tied) < 2:
            continue

        contested += 1
        if hits[-1].doc_id == min(tied):
            smallest_won += 1

    assert contested > 50, f"only {contested} contested cuts; not enough to judge"

    # Ordering ties by doc_id would make this exactly 1.0 by construction.
    share = smallest_won / contested
    assert share < 0.5, (
        f"the lowest doc_id won {share:.0%} of {contested} contested cuts; "
        "the tie-break is still correlated with corpus position"
    )


def test_every_chain_closes_by_following_named_entities(index, tasks):
    """Simulate the whole chain: search, read what came back, search a name in it.

    Retrieving the first document is not enough — the tool has to make each
    *subsequent* query work too. The simulated policy here is deliberately
    dumb: it only ever searches for the title of a document whose name appears
    verbatim in text it has already retrieved. If even that closes every chain,
    a real policy is not blocked by retrieval.

    The loop runs `hops` rounds rather than one. Some tasks are three-hop
    (expedition to leader to instrument), and a single round of following
    would fail them for reasons that have nothing to do with the search tool —
    which is exactly what an earlier version of this test got wrong.
    """
    multi_hop = [
        t for t in tasks
        if t.category == "qa" and len(t.ground_truth["supporting_docs"]) > 1
    ]
    assert {t.metadata["hops"] for t in multi_hop} == {2, 3}

    failures = []
    for task in multi_hop:
        supporting = set(task.ground_truth["supporting_docs"])
        found = {hit.doc_id for hit in index.search(task.prompt, k=DEFAULT_TOP_K)}

        for _ in range(task.metadata["hops"]):
            for doc_id in list(found):
                text = index.get(doc_id).text
                for candidate in index.documents:
                    if candidate.title in text:
                        found |= {h.doc_id for h in index.search(candidate.title, k=1)}

        if not supporting <= found:
            failures.append(task.task_id)

    assert failures == [], (
        f"{len(failures)} multi-hop chains do not close via search: {failures[:5]}"
    )


# --------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------

def test_render_includes_doc_ids_so_the_agent_can_cite(index):
    rendered = render_hits(index.search("Kestrel Array", k=2))
    assert "[ins-00]" in rendered
    assert "Kestrel Array" in rendered


def test_render_reports_an_empty_result_explicitly(index):
    assert render_hits([]) == "No documents matched that query."


def test_hits_carry_the_full_document_text(index):
    """Snippets would clip the numbers multi_tool arithmetic depends on."""
    hit = index.search("Kestrel Array", k=1)[0]
    assert hit.text == index.get(hit.doc_id).text
