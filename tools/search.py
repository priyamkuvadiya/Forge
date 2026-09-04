"""Keyword search over the fixed local corpus in `task_suite/data/corpus.json`.

A local corpus rather than a real search API, for the reason module 2 already
committed to: the eval numbers this project reports have to be reproducible
months later, and a live index changes underneath you. It also keeps the
retrieval side deterministic, so a change in a reward curve is attributable
to the policy rather than to the internet.

Ranking is BM25, hand-rolled — it is about forty lines, and pulling a
dependency in for it would put the retrieval quality of the QA and multi_tool
categories inside a black box this project should be able to explain.

Two decisions worth stating because they are not the obvious defaults:

**Whole documents are returned, not snippets.** The corpus documents are two
or three sentences each, so a snippet window would either return the whole
document anyway or clip a fact in half. Clipping would be actively harmful
here: `multi_tool` answers are arithmetic over numbers that live in these
documents, and a snippet that drops the number turns a solvable task into an
unsolvable one, capping achievable reward for reasons that have nothing to do
with the policy.

**Multi-hop questions need multi-hop reasoning, and `k` is capped to keep it
that way.** The corpus is built so each fact lives in exactly one document and
the chains are one-directional (see `task_suite/qa_world.py`), so no single
document ever answers a multi-hop question. That property is absolute and
holds at any `k`.

What is *not* absolute is the need for a second search *call*. Every document
in a class is rendered from one template, so a question mentioning "vessel"
ranks every vessel document about equally and a large enough `k` returns the
right one incidentally, buried among its neighbours. The agent still has to
read the first document to know which of them the answer is; it just didn't
have to issue a second query to have it in hand.

Two things push back on that, both measured rather than assumed. The corpus
classes were enlarged with distractors (`task_suite/qa_world.py`), which
halved the effect, and `MAX_TOP_K` is set from the measurement below rather
than picked for convenience. What remains is 4 of 118 multi-hop tasks, and
those 4 are asserted exactly in the tests rather than rounded away.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "task_suite" / "data" / "corpus.json"

# Standard BM25 parameters. b=0.75 applies the usual length normalization;
# these documents are near-uniform in length, so it barely bites, but leaving
# it at the default keeps the implementation recognisable as plain BM25.
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_TOP_K = 3
# The cap is a measured choice, not a round number. Entity classes in this
# corpus are large but templated, so a question containing the word "vessel"
# ranks every vessel document about equally and a large `k` sweeps a slice of
# the whole class in one query. Measured over the suite's 118 multi-hop QA
# tasks, the number whose full supporting set arrives in a single search of
# the raw question:
#
#     k                        2    3    4    5    6    7    8   10
#     68-doc corpus            2    2    4    7   14   16   21   32
#     130-doc corpus           1    2    2    4    6    7    9   14
#
# The second row is after `task_suite/qa_world.py` grew the small classes with
# distractors, which roughly halved the effect at every k but did not remove
# it - the count still climbs with k, so the cap still does work.
#
# The number of retrieval tasks whose first hop is unreachable is 0 at every k
# in that range, so raising the cap cannot improve solvability; there is
# nothing left to fix. Five leaves headroom for the imprecise queries a small
# policy actually writes, at 4/118 rather than 14/118. See
# `tests/test_search.py::test_one_query_rarely_retrieves_a_whole_chain`.
MAX_TOP_K = 5

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs.

    No stemming and no stopword list. Stopwords are handled by BM25's IDF
    term, which already scores a word appearing in every document at close to
    nothing — a hard-coded list would do the same job less precisely, and
    would risk dropping a token that happens to matter in this corpus.
    """
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    text: str
    score: float


class SearchIndex:
    """An in-memory BM25 index over the corpus.

    Built once and reused: the RL loop issues a very large number of queries
    across training, and rebuilding per call would put corpus parsing on the
    hot path for no reason.
    """

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("cannot build a search index over an empty corpus")

        self.documents = list(documents)
        self._by_id = {doc.doc_id: doc for doc in self.documents}
        if len(self._by_id) != len(self.documents):
            raise ValueError("corpus contains duplicate doc_ids")

        # Title and body are indexed together: a question naming an entity is
        # searching for the document titled after it.
        self._term_frequencies: list[dict[str, int]] = []
        self._lengths: list[int] = []
        document_frequency: dict[str, int] = {}

        for doc in self.documents:
            tokens = tokenize(f"{doc.title} {doc.text}")
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            self._term_frequencies.append(frequencies)
            self._lengths.append(len(tokens))
            for term in frequencies:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        count = len(self.documents)
        self._average_length = sum(self._lengths) / count
        # The `1 +` keeps every IDF strictly positive, so a term appearing in
        # every document contributes ~0 rather than pushing a score negative.
        self._idf = {
            term: math.log(1 + (count - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, doc_id: str) -> Document:
        """Fetch a document by id. Used by the eval harness, not by the agent."""
        if doc_id not in self._by_id:
            raise KeyError(f"no document {doc_id!r} in the corpus")
        return self._by_id[doc_id]

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        """Return the top `k` documents for `query`, best first.

        Documents scoring zero are dropped rather than padded in to reach `k`:
        a query matching nothing should come back empty, so the policy sees
        that its query failed instead of reading three unrelated documents and
        having to work out that they are noise.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        k = min(k, MAX_TOP_K)

        terms = tokenize(query)
        if not terms:
            return []

        scored: list[tuple[float, str]] = []
        for index, doc in enumerate(self.documents):
            score = self._score(index, terms)
            if score > 0:
                scored.append((score, doc.doc_id))

        # Sort by descending score, then by a hash of the query and the
        # doc_id.
        #
        # The tie-break is doing real work here, not tidying an edge case.
        # These documents are rendered from a handful of templates, so BM25
        # frequently cannot separate a whole class at all: every instrument
        # document contains "instrument" and "measures" exactly once, and the
        # only thing varying the score is length. Measured over the suite's
        # retrieval tasks, 110 of 168 have their top-k cut decided by an exact
        # score tie rather than by relevance.
        #
        # That makes the tie-break's *bias* the thing that matters. Ordering
        # by doc_id, as this first did, ranks by entity index - and since
        # distractors are appended after the real entities, every tie went to
        # a real document. The corpus grew without retrieval getting much
        # harder, which is the opposite of the intent.
        #
        # Hashing decorrelates the order from the numbering while staying
        # exactly reproducible: the same query against the same corpus always
        # returns the same documents in the same order, so eval re-runs agree,
        # but nothing about a document's position in the corpus helps it rank.
        # `hashlib` rather than `hash()`, which is salted per process and would
        # make results differ between runs.
        scored.sort(key=lambda pair: (-pair[0], self._tie_break(query, pair[1])))

        return [
            SearchHit(
                doc_id=self._by_id[doc_id].doc_id,
                title=self._by_id[doc_id].title,
                text=self._by_id[doc_id].text,
                score=score,
            )
            for score, doc_id in scored[:k]
        ]

    @staticmethod
    def _tie_break(query: str, doc_id: str) -> bytes:
        return hashlib.blake2b(f"{query}\x00{doc_id}".encode(), digest_size=16).digest()

    def _score(self, index: int, terms: list[str]) -> float:
        frequencies = self._term_frequencies[index]
        length = self._lengths[index]
        norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self._average_length)

        total = 0.0
        for term in terms:
            tf = frequencies.get(term)
            if not tf:
                continue
            total += self._idf[term] * (tf * (BM25_K1 + 1)) / (tf + norm)
        return total


def render_hits(hits: list[SearchHit]) -> str:
    """Format results for the agent transcript."""
    if not hits:
        return "No documents matched that query."
    return "\n\n".join(
        f"[{hit.doc_id}] {hit.title}\n{hit.text}" for hit in hits
    )


def load_documents(path: Path | str = DEFAULT_CORPUS_PATH) -> list[Document]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(doc_id=entry["doc_id"], title=entry["title"], text=entry["text"])
        for entry in payload["documents"]
    ]


def load_index(path: Path | str = DEFAULT_CORPUS_PATH) -> SearchIndex:
    return SearchIndex(load_documents(path))
