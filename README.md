# Forge

Training an agentic LLM's tool-use policy with reinforcement learning on
verifiable-reward tasks — arithmetic requiring a calculator tool, coding
problems checkable by running tests, and multi-hop QA answerable via a
search tool. Rewards come from real programmatic verifiers, not a learned
preference model, so nothing here can be gamed the way an eval score can.

This repo makes two separate claims, kept architecturally and narratively
distinct throughout:

1. **A small transformer language model built from scratch** — a standalone
   proof of understanding transformer internals (attention, positional
   encoding, the training loop). It does not feed the agent below.
2. **An agent whose tool-use policy is trained with RL (GRPO)** — the real
   system, backed by a small pretrained open-weight model (Qwen2.5), not by
   the from-scratch model in (1). An 8GB GPU cannot pretrain a model capable
   enough to be a useful agent backbone.

## Status

Work in progress, built incrementally. Sections below are added only once
the corresponding module exists and has been run for real.

Modules 1 (from-scratch transformer), 2 (task suite and verifiers) and 3
(tools, including the sandbox) are complete — 381 tests. Next is the prompted
baseline agent, which produces the first reward numbers this project can
report. Nothing has been scored against the task suite yet, so there are no
results below beyond module 1's training curve.

## Module 1: from-scratch transformer

`transformer_scratch/` — a GPT-style decoder-only transformer (causal
self-attention, learned positional embeddings, pre-LN residual blocks) built
directly on `torch.nn`, trained on the character-level tinyshakespeare
corpus. Exists solely to demonstrate transformer internals are understood
from scratch; it is not used anywhere else in this project.

### Architecture

| | |
| --- | --- |
| Parameters | 10.77M |
| Layers / heads / embedding | 6 / 6 / 384 |
| Context length | 256 characters |
| Vocabulary | 65 characters (built from the corpus, no external tokenizer) |
| Dropout | 0.2 |

Attention, the MLP block, positional embeddings and the training loop are
written directly against `torch.nn` — no `transformers` import anywhere in
this module. Token embedding and output head share weights.

### Training

Trained on tinyshakespeare (1,115,394 characters, 90/10 train/val split) on
a single RTX 4060 Laptop GPU: 5000 iterations, batch 64, AdamW, bf16
autocast, learning rate 1e-3 with 100-iteration linear warmup then cosine
decay to 1e-4, gradient clipping at 1.0. **Wall clock: 15.0 minutes**, peak
3033 MiB VRAM.

### Results

| metric | value |
| --- | --- |
| **Best validation loss** | **1.4652** (iteration 1750) |
| Training loss at that point | 1.1114 |
| Final validation loss (iter 4999) | 1.6703 |
| Final training loss (iter 4999) | 0.6411 |

![loss curve](transformer_scratch/artifacts/loss_curve.png)

**The model overfits after iteration ~1750**, and the curve above shows it
plainly: validation loss bottoms out at 1.4652 and then climbs for the
remaining 3250 iterations while training loss keeps falling to 0.6411. This
is expected — 5000 iterations at batch 64 is roughly 73 epochs over a 1.1M
character corpus, so the ceiling here is memorisation, not data.

Two honest consequences:

- The released checkpoint is the **best-validation** one (iteration 1750),
  not the final one. Keeping the final model would have shipped something
  0.205 nats worse.
- 5000 iterations was more than this setup needed. The validation minimum
  arrives around iteration 1750–2000; the remaining ~9 minutes of compute
  bought nothing but overfitting. Left as-is because it makes the
  overfitting visible rather than hiding it behind an early stop.

### Sample output

Generated from the best-validation checkpoint (temperature 0.8, unconditioned):

```text
AUTOLYCUS:
And the lower of much a dufficious hands
at once to make them; Or, and entreats
guides like a life to the merely territ, my
first soul to fight find their harms. My friends sit their
treacherous enemy is an old sins or as mine eagle
My partner doth not between my sister news with them;

DUKE VINCENTIO:
Heavens bring the read in his noble kind heart;
And being and ancient villany of our course,
Good sooth, as if I cannot be assured
To burn his bed back, in pointy a bones
```

This is a fair representative draw, not a best-of-N pick. The model has
learned the *shape* of the corpus — real speaker headings, blank-verse line
breaks, largely well-formed English words and plausible local syntax — but
it is not semantically coherent, and it invents words ("dufficious",
"territ"). That is the expected ceiling for a 10.77M-parameter character-level
model at a validation loss of ~1.47, and it is the honest limit of what this
module claims.

### A note on precision

The first attempt at this run was fp32 and hard-crashed the machine
(Kernel-Power 41, no crash dump — a power cut, not a driver fault). Measuring
before re-running turned out to matter:

| config | peak power | peak temp | peak VRAM | µs/token |
| --- | --- | --- | --- | --- |
| fp32, batch 32 | 115.8 W | 78 °C | 2287 MiB | 15.5 |
| bf16, batch 32 | 82.6 W | 66 °C | 1731 MiB | 9.0 |
| bf16, batch 64 | 81.4 W | 66 °C | 3033 MiB | 9.6 |

bf16 autocast draws ~30% less power, runs 12 °C cooler and is ~1.6x faster
per token than fp32. Batch size barely moves peak power (81.4 W at batch 64
vs 82.6 W at batch 32) because the GPU is already saturated at batch 32 — so
gradient accumulation, which would have been the usual way to keep effective
batch size up under a power ceiling, was measured to be unnecessary here and
was not used.

### Reproducing

```bash
python transformer_scratch/train.py      # ~15 min on an RTX 4060
python transformer_scratch/generate.py --save
pytest transformer_scratch/tests -q      # 6 tests
```

The corpus downloads automatically on first run. Tests cover attention/MLP/
block output shapes, an actual causal-masking correctness check (perturbing a
future token must not change earlier positions' outputs), and an
overfit-one-batch sanity check.

## Module 2: the verifiable-reward task suite

`task_suite/` — the tasks the agent is trained and evaluated on, and the
`verify()` functions that score them. This is the reward signal for the whole
project, so it is built before any agent, tool or training code exists: if the
verifiers are wrong, every reward curve and pass-rate reported later is wrong
with them, and no amount of care in the training loop fixes that.

| split | math | code | qa | multi-tool | total |
| --- | --- | --- | --- | --- | --- |
| train | 150 | 30 | 85 | 35 | 300 |
| held out | 60 | 20 | 33 | 15 | 128 |

The suite is pinned to `task_suite/data/suite.json` and committed. Everything
that produces it is deterministic, so it could be rebuilt on demand — it is
stored anyway, because an eval number only means something if the tasks behind
it are exactly the tasks that were run, and a file anyone can diff is a
stronger guarantee than "it regenerates the same". A test asserts the
checked-in file still matches a fresh build, so a stale pin surfaces as a
failing test rather than as a quietly wrong number.

### The reward contract

The policy reasons and calls tools however it likes, but has to commit to a
final answer inside `<answer></answer>` for that answer to be scored. Only the
last such block counts, and a response with no parseable commitment scores
0.0 rather than being generously re-parsed. That strictness is deliberate:
format compliance is part of what RL has to learn, and looser parsing — grab
the last number, accept a prefix match — hands out reward for text that never
actually answered the question.

`verify(task, response) -> float` in `[0, 1]` is the single entry point, so
the baseline agent, the GRPO loop and the eval harness cannot drift apart in
how they score.

### How each category is made verifiable

**Math (`tasks_math.py`)** — 10 parameterized templates. The prompt and the
answer are rendered from the same drawn parameters, so a task's answer cannot
disagree with its question; that is the failure mode which quietly corrupts a
hand-written math set. Numbers are drawn to be awkward on purpose — prices to
the cent, non-round rates, multi-step chains — with fewer than 10% of answers
landing on a whole number. If the arithmetic were easy, an improvement in
reward would say nothing about whether the policy learned to use the
calculator. Scoring is a numeric match within a per-task tolerance, which the
template sets (0 for integer answers, half a cent for money).

Their tests assert *properties* rather than recomputing the formula, which
would only prove the code equals itself: compounding grows the principal, two
machines together beat either alone, a weighted average lies between its
inputs, an equal rise and fall ends below the start.

**Code (`tasks_code.py`)** — 50 hand-authored problems, 223 test cases.
Generated instances make no sense here: a coding problem's difficulty lives in
its specification, not its numbers, so thirty instances of one template would
measure one skill thirty times. Held-out coding tasks are therefore different
*problems*, not different instances — the only way a held-out coding score
means anything. There are 20 of them, which is enough for a real effect to
show and still small enough that the eval will have to report it with an
interval rather than as a bare percentage.

Hand-written expected values are exactly where typos hide, so the test file
holds an independently written reference solution for every one of the 50
problems, written from the prompt text rather than from the expectations, and
asserts each scores 1.0 through the real verifier. Two independent derivations
agreeing is what makes the expectations trustworthy. Score is the fraction of
test cases passed, not all-or-nothing — coding is the category where a small
policy most often gets the shape right and one edge case wrong, and a dense
signal there is worth more to GRPO than a cliff.

**Multi-hop QA (`qa_world.py`)** — a 68-document corpus about an invented
research institute, and 118 questions over it. The corpus is fictional on
purpose. A question about the real world can be answered from the base model's
weights without ever calling the search tool, so the reward would be measuring
memorization with no way to tell the two apart. Every entity here is invented,
so retrieval is the only route to a correct answer.

Questions are generated from the entity graph rather than written by hand,
which guarantees every answer is actually supported by the corpus — a
hand-written multi-hop question whose second hop is in no document is an
unanswerable task that silently caps the reward the category can earn, and it
is easy to write one by accident. Each fact lives in exactly one document and
the chains run one way: an expedition's document names its leader but not the
instrument that leader designed, and the researcher's document names the
instrument but not the expedition. A test asserts, for all 118 questions
against the rendered prose, that no single document contains both the entity
the question names and the answer it wants — one careless filler sentence
would turn a two-hop question into a lookup while the question text stayed
identical. Questions run at 2 and 3 hops across 8 templates, each with three
surface phrasings: with one phrasing apiece the category would be eight
sentence moulds, and a policy can learn which slot of which mould holds the
answer instead of learning to retrieve it.

Scoring is exact match on a normalized string against authored alias forms,
not token F1. Overlap credit would reward an answer for merely containing a
right-looking word, and a small policy trained against that learns to stuff
answers with plausible entities.

**Cross-tool (`tasks_multi_tool.py`)** — 50 tasks that need search *and* the
calculator. The other three categories map one-to-one onto a tool: math wants
the calculator, QA wants search, code wants the executor. A policy can infer
which tool to reach for from the shape of the question alone, so RL on those
categories teaches an agent to *use* a tool well but never to *choose* one —
a thin reading of "trained tool-use policy". These tasks put the numbers in
the synthetic corpus, so they have to be retrieved, and make the answer
arithmetic over them, so it has to be computed. Six templates over the same
world: elapsed years, durations, summed instrument masses, vessel length
differences, combined berths.

Two properties are enforced when the tasks are generated rather than hoped
for, because a task failing either is not cross-tool at all — just QA or
arithmetic wearing the other one's clothes. The answer never appears verbatim
in any supporting document, so the arithmetic cannot be skipped by spotting
the result already written down; candidates that would violate this are
dropped, not reworded. And differences are only emitted when they come out
positive, so no task turns on guessing which way round a subtraction was
meant.

### Running untrusted code, and not being lied to by it

`verify_code.py` never executes anything itself. It builds a self-checking
program and hands it to an injected runner, which module 3 supplies as a real
sandbox. That inversion keeps the verifier pure and unit-testable with a fake
runner, and leaves exactly one place in the repo where untrusted generated
code actually runs — so exactly one place to harden.

The first version of that harness graded itself inside the sandbox and
reported booleans, and a submission could forge the result outright: poisoning
`sys.modules["json"]` so `dumps` returned `"[true, true, true]"` scored 1.0 on
a function that returned the string `"definitely not the answer"`. Since this
is a reward function an RL policy optimizes against, that is a live hole
rather than a curiosity.

It is fixed at the root rather than patched. **The expected answers never
leave the verifier.** The harness carries only the call arguments; it invokes
the function, serializes whatever came back, and the comparison happens here
in trusted code. A submission that wants a passing report line now has to
print the correct return values and has no way to learn what they are —
printing the correct values *is* solving the task. That exact poisoning
submission is now a regression test, alongside one asserting the harness never
contains the string `expected` at all.

### The held-out split

Held-out means something different in each category, and each is enforced by a
test: math draws independent problem parameters from a separate seed, code
holds back entirely different problems, and QA and the cross-tool tasks
partition the question *subjects* so a held-out question asks about an entity
no training question ever mentions. The corpus itself is necessarily shared —
it is the world both splits search. `build_suite()` refuses to return a suite
in which any prompt appears in both splits.

Math is still the largest category, because it is the cheapest to generate.
That is a real imbalance, and it is why every eval number this project reports
is broken down per category rather than pooled into a single headline
pass-rate.

### Reproducing

```bash
python -m task_suite.registry      # rebuild and pin the suite, print the summary
pytest task_suite/tests -q         # 172 tests
```

No reward numbers appear here yet — nothing has been run against these tasks.
The first real numbers arrive with the prompted baseline agent (module 4).

## Module 3: the tools

Three tools the agent can call. Two are safe by construction; the third runs
code a language model wrote, and is the only part of this repo whose isolation
is a design rather than an implementation detail.

### Calculator

Arithmetic parsed with `ast` and walked against a whitelist, never `eval`. The
whitelist is the obvious half. The less obvious half is refusing expressions
that are *valid* but hostile: `9**9**9` is four characters of legal arithmetic
that pins a core and exhausts memory. GRPO will emit millions of expressions
and is under no obligation to emit sensible ones, so guards on exponent size,
estimated result width, expression length and term count are a correctness
requirement, not defensive padding — an unguarded power hangs a training run
with no error to point at.

The tests check sufficiency as well as safety: every one of the suite's 210
math tasks is solved through the calculator to within that task's own
tolerance. A calculator that was safe but could not express the arithmetic
would cap the achievable reward invisibly.

### Search

Hand-rolled BM25 over the pinned 130-document corpus. Whole documents are
returned rather than snippets, because cross-tool answers are arithmetic over
numbers *inside* those documents and a clipped snippet turns a solvable task
into an unsolvable one.

Two properties were measured rather than assumed, and both turned out worse
than expected before they were fixed. A single query used to return a
multi-hop question's entire supporting set for **32 of 118** questions, because
entity classes were small and templated — a question mentioning "vessel"
ranks every vessel document about equally, so a large `k` swept a slice of the
whole class. And because documents rendered from one template score
*identically*, 110 of the 168 retrieval tasks had their top-k cut decided by an
exact score tie rather than by relevance, which made the tie-break — ordering
by `doc_id`, i.e. by entity index — the thing that actually decided what the
agent read.

Three changes, each measured:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 68 docs, partial tied groups | 2 | 2 | 4 | 7 | 14 | 16 | 21 | 32 |
| 130 docs, partial tied groups | 1 | 2 | 2 | 4 | 6 | 7 | 9 | 14 |
| 130 docs, whole groups only | 0 | 1 | 1 | **2** | 2 | 3 | 3 | 4 |

The corpus gained 62 distractor entities that no question asks about and no
answer appears in; results are cut at a tie boundary rather than sampled from
one; and `MAX_TOP_K` is 5 on the strength of the table rather than by taste.
The number of tasks whose first hop is unreachable is **0 at every k in every
row**, so none of this costs solvability.

A consequence worth stating because it looks like a bug: a query too broad to
discriminate returns *nothing*. Searching "vessel" matches all 32 vessel
documents equally, and there is no honest way to pick five of them — coming
back empty tells the policy its query does not narrow anything down, which is
the thing it needs to learn.

### The code sandbox

An AppContainer, a Job Object, and a suspended start, each closing something
the others do not. There is no Docker and no WSL on the development machine,
and Python on Windows has no `resource` module, so `fork` + `setrlimit` was
never available.

- **AppContainer** confines what the process can *reach*. This is the part
  that matters for reward hacking. The first version did not have it, and a
  probe showed a submission could hardcode the repo path and read
  `task_suite/data/suite.json` — the expected answers for its own task. The
  suite withholds those answers from the harness precisely so a passing line
  requires computing them, and that defence is worth nothing if they are
  readable from disk. Network access is refused too.
- **Job Object** confines what it can *consume*: memory caps so an allocation
  fails as `MemoryError` instead of swapping the machine, `ActiveProcessLimit
  = 1` so a fork bomb fails at its first spawn, kill-on-close, and
  die-on-unhandled-exception so a crash cannot raise a modal dialog and hang
  an unattended run.
- **`CREATE_SUSPENDED`** removes the assignment race: the process is created
  suspended, put in the job, and only then resumed.

A sandboxed run costs **285 ms**, of which **249 ms** is bare CPython startup
on Windows — the isolation machinery is about 40 ms. That matters because the
usual argument for a subprocess over a container is speed, and measured, it
mostly isn't: a container pays the same interpreter cost on top of its own.
The honest case against a container here is the multi-gigabyte install
becoming a dependency for anyone reproducing this project's numbers.

The sandbox tests run real fork bombs, real memory bombs and real infinite
loops rather than asserting the happy path, and a separate integration file
runs all 50 coding problems' reference solutions through the real verifier and
the real sandbox, requiring 1.0 on every one — so that no isolation constraint
silently depresses the category's reward.

Known limits, stated rather than glossed: the child runs as the calling user,
and everything it *is* granted — the interpreter directory, its own scratch
space — it can read. This is a strong boundary against a model that wanders
into the repo and a weak one against an attacker who already has local code
execution.

### Reproducing

```bash
pytest tools/tests -q             # 203 tests, including the adversarial ones
```
