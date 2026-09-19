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

![tests](https://github.com/priyamkuvadiya/Forge/actions/workflows/tests.yml/badge.svg)

Modules 1 (from-scratch transformer), 2 (task suite and verifiers), 3 (tools,
the sandbox, and the tool-call contract) and 4 (the prompted baseline agent)
are complete — 474 tests, run on Linux and Windows on every push.

The control-group numbers now exist: the prompted Qwen2.5-0.5B-Instruct
baseline scores a **macro-average reward of 0.1908** across the five held-out
categories. It calls a tool in under a fifth of episodes in every category —
not once in 160 coding episodes — and when it does call one, it never chains:
of the 72 QA and multi_tool episodes that used a tool, none made a second
call, on the two categories that structurally require more than one hop.
Closing that gap is what module 5's GRPO training has to do. Next is that
training loop; there is no trained policy and no baseline-versus-RL comparison
yet.

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

| split | math | code | qa | multi-tool | no-tool | total |
| --- | --- | --- | --- | --- | --- | --- |
| train | 150 | 30 | 85 | 35 | 40 | 340 |
| held out | 60 | 20 | 33 | 15 | 20 | 148 |

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
calculator. The other single-tool categories map one-to-one onto a tool: math wants
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

**No-tool control (`tasks_no_tool.py`)** — 60 tasks that need no tool at all.
Every other category rewards reaching for one, which makes a policy that
learned "always search" score identically to one that learned *when* to
search. Over-calling is a real trained-policy failure mode and this suite
could not see it. These tasks are single-operation arithmetic on small
integers, counting items listed in the prompt, and questions whose answer is
stated in the prompt — half numeric, half textual, so the control covers
reflexive use of the calculator *and* of search.

Three things are asserted rather than assumed, because a control that
secretly needed a tool would make the measurement built on it meaningless.
The arithmetic stays small enough to do mentally; every comprehension answer
appears verbatim in its own prompt; and none of the invented names appears
anywhere in the search corpus, so a policy that reaches for search gets
nothing back. The splits draw on disjoint pools of operands, names and
vehicles, so held-out means here what it means everywhere else.

**The reward does not penalise a tool call.** Scoring is correctness alone.
What this category enables is a measurement — the call count from the tool
trace — so module 9 can report "the RL policy called a tool on N% of tasks
that needed none" alongside the reward. Folding a penalty into the reward
would be training against a proxy invented here rather than against a
verifiable outcome, which is the thing this whole project avoids.

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
pytest task_suite/tests -q         # 186 tests
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

`code_exec.py` picks a backend and presents one interface, because everything
above it should neither know nor care which it got.

**Windows** — an AppContainer, a Job Object, and a suspended start, each
closing something the others do not. There is no Docker and no WSL on the
development machine, and Python on Windows has no `resource` module, so
`fork` + `setrlimit` was never available here.

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

**POSIX** — `fork` plus `setrlimit`: address-space and CPU caps, `RLIMIT_NPROC
= 1` so a fork bomb fails at its first spawn, a file-size cap, no core dumps,
and the child in its own session so a timeout kills the whole process group.
It exists because a runner that only works on one laptop makes the coding
category unreproducible, which fails this project's own bar.

One honest caveat, also stated at the top of `code_exec_posix.py`: **it does
not confine the filesystem**. Proper confinement needs a mount namespace,
which needs root or user namespaces, neither of which can be assumed — so on
POSIX a submission can still read the repo and therefore the expected answers,
the exact hole the AppContainer closes on Windows. Landlock is the intended
fix and is not done yet.

That difference is data rather than prose. Each backend declares a
`Confinement` — what it bounds and what it does not — the eval harness records
it beside every coding score, and `python -m tools.code_exec` prints it with a
loud warning when the filesystem is unconfined. A coding number therefore
always says which boundary produced it, instead of a Linux result looking
identical to a Windows one. The tests asserting filesystem and network
confinement are marked Windows-only for the same reason: so they cannot skip
quietly and imply a boundary that is not there.

It was written blind — there is no Linux, WSL or container runtime on the
development machine — which is exactly why CI runs the full suite on
`ubuntu-latest` on every push (424 passed, 8 Windows-only tests skipped). That
was not ceremony: the first Linux run found two real bugs, both of which would
have shipped as silent reward bugs rather than as visible failures.
`RLIMIT_NPROC = 1`, copied across from the Windows job object's
`ActiveProcessLimit = 1`, blocks *threads* on Linux because a thread is a
task — so any solution using `threading` would have scored 0. And the same run
on Windows CI showed the harness could not open the null device inside a fresh
AppContainer, which made *every* coding task score 0 there while passing
locally. Both look exactly like a model that cannot code.

Known limits on Windows too, stated rather than glossed: the child runs as the
calling user, and everything it *is* granted — the interpreter directory, its
own scratch space — it can read. This is a strong boundary against a model that wanders
into the repo and a weak one against an attacker who already has local code
execution.

### One contract for calling them

`toolbox.py` is what the rest of the project actually talks to. Without it,
each of modules 4, 5, 7 and 9 would have invented its own way to name a tool,
parse a call and record what happened — slightly differently.

The policy emits `<tool name="calculator">2 + 2</tool>`, deliberately shaped
like the `<answer>` contract so there is one syntax to learn, not two. Every
tool returns the same `ToolResult`, and every failure — bad syntax, unknown
tool, a refused expression, a timeout — comes back as a result rather than an
exception, because one bad tool call is ordinary policy behaviour that should
cost reward, not abort a rollout.

A `ToolSession` wraps one task attempt with a call budget (8 by default;
the longest legitimate chain in the suite is a three-hop QA question) and a
trace of every call, its arguments, its result and its duration. That trace is
not bookkeeping: module 7 stores it in SQLite for the live reasoning view, and
module 9 needs the call counts to answer the question the no-tool category
exists to ask. Exhausting the budget returns a message telling the policy to
answer with what it has, rather than silently dropping the call — a policy
that is ignored learns nothing.

The prompt describing the tools is generated from the registry, so it can
never advertise a tool that is not present — a live risk here, since the
sandbox is platform-specific and a registry built without it must not promise
a `python` tool the policy would waste its whole budget discovering is absent.

### Reproducing

```bash
python -m tools.code_exec --setup   # one-time, ~90s on Windows
python -m tools.code_exec           # report the backend and what it confines
pytest tools/tests -q               # 246 tests, including the adversarial ones
```

The setup step creates the AppContainer profile and grants it read access to
the interpreter directory — an `icacls` pass over roughly 50,000 files. It is
idempotent, marker-guarded on both the container SID and the interpreter (so a
Python upgrade re-does it rather than failing obscurely), and happens once per
machine. It is an explicit command rather than a lazy side effect of the first
sandboxed call, because as a side effect it is indistinguishable from a hang.

`python -m tools.code_exec` with no arguments prints which backend is active
and what it confines, and warns loudly when the filesystem is not confined.
The same value is recorded by the eval harness beside every coding score, so a
number always says which boundary produced it:

```text
backend:     windows
confinement: appcontainer+job (confines: filesystem, network, resources)
```

Runs are independent and the sandbox is thread-safe: measured on this machine,
16 runs take 5.6 s serially and 0.8 s across 8 workers, so an effective 49 ms
per run rather than 285 ms. Module 5 should execute code rollouts in parallel
for that reason — with the caveat that each concurrent run may hold up to the
256 MB memory cap, so the worker count is a RAM budget, not a free lunch.

## Module 4: the prompted baseline agent

`baseline_agent.py` — **Qwen2.5-0.5B-Instruct**, prompted rather than trained,
using the module 3 tools on the module 2 tasks. This is the control group. It
is the reason any later claim about RL can be checked, and the numbers below
are what module 5's trained policy has to beat.

Nothing in this module trains anything. The backbone is an off-the-shelf
open-weight model, loaded in bf16, and the only thing that changes between
this arm and the RL arm is the weights: same system prompt, same worked
examples, same tool contract, same 8-call budget, same verifiers, same
sampling parameters.

### The loop

An episode is a chat transcript. The policy writes a turn; if it ends in
`<tool name="...">...</tool>` the call runs and the result comes back as the
next message; if it contains `<answer>...</answer>` the episode is over and
`task_suite.verify()` scores it. Three details are worth pulling out:

- **Generation stops at `</tool>` or `</answer>`**, and everything the model
  writes after its own tool call is discarded. That is not tidiness. A small
  model routinely writes `<tool name="search">...</tool>` and then invents the
  `<tool_result>` block it is about to be handed, together with an answer
  derived from it. Keeping that text would let an episode collect reward off a
  fabricated retrieval, and would teach module 5's policy to hallucinate tool
  output as a strategy.
- **All live episodes step in lockstep**, so each generation call sees a full
  batch and finished episodes drop out. The tool calls a batch produces are
  then executed together on a thread pool, which is what makes the sandbox's
  measured 49 ms-across-8-workers matter. Run one episode at a time and the
  held-out eval below goes from half an hour to most of a day.
- **The policy interface is `conversations -> completions` and nothing else.**
  Chat templating, padding and sampling live in the backbone, so the loop
  itself is driven by scripted policies in CI, on both platforms, with no
  torch installed. The truncation rule, the answer/tool-call ordering rule,
  the budget and the nudge are all text handling, and a GPU adds nothing to
  testing them.

### What the dry run changed

The suite was frozen only *after* a throwaway probe against the real model, on
the **train** split, because a task-suite change after the baseline is
recorded invalidates the baseline. Four things came out of it, and the first
is the one that matters:

**Prompted from a description alone, the model scored 0.000 in every
category.** Across a ten-task probe it emitted zero `<answer>` blocks and zero
tool calls. It solved "What is 5 plus 4?" correctly in prose and scored 0.0;
asked about a fictional expedition, it apologised for not knowing instead of
searching for it. A baseline of zero everywhere is arithmetically honest and
analytically worthless — it measures whether a 0.5B model can infer an
unfamiliar tag convention from prose, which is not what this project claims RL
improves, and it would hand module 5 a guaranteed win over a control that
never attempted the task.

So the control gets the prompt a competent engineer would actually ship: four
worked examples replayed as real conversation turns — a direct answer, a
coding answer, a calculator call, a search call. They are invented and
deliberately unlike anything in the suite, their tool results are produced by
the real `calculate`, `render_hits` and `ToolResult.render` rather than typed
out, and module 5's policy is prompted with exactly the same turns.
`--no-few-shot` exists so their value can be quoted rather than guessed at,
and it is quoted below: they are worth fifteen times the macro reward, almost
all of it output-format compliance.

One of them was quietly cheating and a test caught it. The first direct-answer
example was "What is 3 plus 6?", which is *verbatim* `notool-train-0030` — the
worked example was handing the policy a solved task from the very category
that exists to measure it. `test_few_shot_examples_leak_no_task_from_the_suite`
now checks every exemplar against every task prompt, every corpus document
title and every coding entry point.

**The recovery nudge went through three wordings, and two of them destroyed
answers the model had already got right.** If a turn ends with neither a tool
call nor an answer, the agent asks once for the answer in tags. "Give it now,
using what you already know" was read as an invitation to start over, and a
correct fenced `sum_of_evens` came back as `<answer>0</answer>`. Adding "if
the task was to write a function, put the function in the tags" fixed coding
and broke arithmetic instead — three math episodes that had reasoned their way
to a number in prose came back as Python function definitions. Showing the
shape as a fill-in template, `<answer>YOUR ANSWER</answer>`, got the tags back
and lost the answer: five episodes answered the literal string "YOUR ANSWER".
The wording that survived says only where the two literal strings go.

**One task in the suite had no determinate answer, and the base model was
right to say so.** The `no_tool` counting template read "A box contains
bottles, folders and chairs. How many items are in the box?" — the pool is
plural nouns, so the stated answer of 3 was really a count of *kinds*. The
model refused, correctly. A task whose stated answer is wrong is worse than a
hard task: it is reward for agreeing with the author. The prompt now asks for
kinds. This is the whole reason the dry run happens before the freeze.

**The nudge recovers format, not correctness — measured.** Over 200 train
episodes with and without it, answer rate moves a great deal (QA 0.075 →
0.300, `no_tool` 0.425 → 0.600) and macro reward does not move at all (0.205 →
0.200). The answers it rescues were wrong anyway. It stays on by default
because it separates a format failure from a task failure, which module 9
needs and the reward curve cannot show; it is now on the record that it costs
an extra turn and buys no reward.

### Baseline results, held out

All 148 held-out tasks, 8 rollouts each — 1184 episodes in 18.3 minutes on the
RTX 4060. Qwen2.5-0.5B-Instruct in bf16, temperature 0.7, top-p 0.9, an 8-call
tool budget, and one recovery nudge. Coding was scored under
`appcontainer+job (confines: filesystem, network, resources)`; the full record,
including every per-episode reward and tool trace, is in
`artifacts/baseline/heldout.json`.

| category | tasks | episodes | mean reward | 95% CI | solved | answer rate | nudged | any tool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math | 60 | 480 | 0.021 | [0.006, 0.042] | 0.021 | 0.756 | 0.377 | 0.071 |
| code | 20 | 160 | 0.439 | [0.342, 0.543] | 0.281 | 0.894 | 0.106 | 0.000 |
| qa | 33 | 264 | 0.000 | [0.000, 0.000] | 0.000 | 0.379 | 0.875 | 0.178 |
| multi_tool | 15 | 120 | 0.000 | [0.000, 0.000] | 0.000 | 0.433 | 0.817 | 0.208 |
| no_tool | 20 | 160 | 0.494 | [0.331, 0.650] | 0.494 | 0.719 | 0.431 | 0.062 |

**Macro-average reward: 0.1908**, the unweighted mean of the five category
means. That is the number module 5 has to beat. Intervals are percentile
bootstraps resampled over **tasks**, not episodes — eight rollouts of one
question are eight draws from one question, and resampling episodes would
report a confidence the 20-problem coding split does not have.

**The model barely uses tools at all, and that is the headline.** It called a
tool in 7% of math episodes, 18% of QA, 21% of multi_tool, and — on 160 coding
episodes with a Python sandbox available and advertised in its prompt —
**zero times**. Prompted with four worked examples including a calculator call
and a search call, a 0.5B model still overwhelmingly answers from its own
weights. That gap is exactly what this project claims RL closes, and it is now
a measured number rather than an assumption.

**It never chains.** 116 of the 1184 episodes called a tool at all, and 114 of
those made exactly one call. On QA and multi_tool — the two categories that
structurally *require* more than one hop, since the supporting facts are
deliberately spread across documents — **not one of the 72 episodes that
called a tool ever called a second time.** The baseline does not do multi-hop
retrieval; it does one search and then commits. Sequential tool use is not a
weak spot in the control, it is absent from it.

**Retrieval is not the only gap, which matters for what RL has to learn.** On
QA, episodes that searched scored 0.000 and episodes that searched nothing
also scored 0.000. Searching moved the answer *rate* (multi_tool 0.379 →
0.640) without moving reward at all. So the model cannot yet use what it
retrieves, and a policy that merely learned "call search more often" would not
move these numbers. The answers it produces instead are confabulations with a
real-world shape — `"British Geological Survey (BGS)"`, `"Portsmouth Naval
Base in England"`, `"Newcastle upon Tyne"` — against a corpus that is
fictional precisely so that answering it requires retrieval. Module 2 made
that choice to stop the reward measuring memorisation, and this is the
evidence it was the right one.

**Nothing ran out of room.** Every episode ended by answering (773) or by
declining to (411); exactly one of 1184 exhausted the 8-call budget and none
hit the turn cap. The zeros above are the model failing the task, not the
harness truncating it — worth establishing, because a loop that quietly
strangled its own episodes would produce the same table.

**This breaks the `no_tool` control, and the control has to say so.** That
category scores a 0.062 tool-call rate, which looks like a policy that knows
when not to reach for a tool. It isn't: the model scarcely reaches for one
anywhere, so the low rate is a property of the whole arm, not discrimination
between task types. The measurement only becomes meaningful once an arm uses
tools at all. Module 9 must report it as a *pair* with the other categories'
rates, never alone.

Three episodes invented a tool that does not exist (`calc`, `calculation`, and
— conflating the two contracts — `answer`). At 2.6% of tool-calling episodes
that is a footnote rather than a finding, but it is the cost of having
deliberately made `<tool>` and `<answer>` look alike, and it is cheap to watch
for once a trained policy starts calling tools in earnest.

**Math is 0.021 because of arithmetic, not formatting.** The split over 480
episodes: 66.9% produced a parseable number that was simply wrong, 24.4%
never emitted an `<answer>` block, 6.7% emitted something unparseable, 2.1%
were correct. So roughly a third is format failure and two thirds is
arithmetic — the verifier is not the bottleneck. Thirteen answers landed
within 2% of the target but outside the 0.005 tolerance; that tolerance is
correct for money rounded to cents, and widening it to collect those would be
tuning the ruler to flatter the result.

Two of the unparseable answers are worth quoting, because they are failures of
this repo's own prompt rather than of the model: `"immediately before it and
the literal text"` is the model parroting the answer-format instruction back,
and `"That reply cannot be scored because it did not contain the answer
tags."` is it echoing the recovery nudge. The nudge wording survived three
rewrites already (above); this is a fourth failure mode it has, and it is on
the record rather than quietly fixed after the freeze.

**Coding's dense score is doing real work.** Of 160 episodes, 28.1% passed
every test, 35.0% passed some, and 36.9% passed none. Reported as
solved/unsolved this category would be 0.281 and would throw away the 35% that
carry most of the gradient signal GRPO will use. It is 0.439 as a fraction of
223 test cases, and the ±0.10 interval on 20 problems is why the project's
eval rules forbid quoting it as a bare percentage.

**`no_tool`'s 0.494 is two different numbers.** Split by task kind, the
arithmetic half scores 0.646 and the reading-comprehension half 0.266. The
category mean is a blend of a thing the model can do and a thing it cannot.

**Checked for reward hacking; found none.** The coding answers were scanned
for `sys.modules` poisoning (the attack that already beat an earlier version
of this harness), sentinel forgery, reading `suite.json`, `builtins` patching
and early clean exits. One submission matched on the word `expected` and turned
out to be writing its own unit tests with an `expected_output` variable, and it
scored 0.0 anyway. No QA or multi_tool episode scored above zero without a tool
call. Ten perfect scores had answers under two characters and all were
`no_tool` tasks whose correct answer really is `3` or `5`.

### What the worked examples are worth

The same 1184 held-out episodes with `--no-few-shot`, which removes the four
worked examples and leaves the description of each tool intact:

| category | with examples | without | answer rate | any tool |
| --- | --- | --- | --- | --- |
| math | 0.021 | 0.006 | 0.756 → 0.283 | 0.071 → 0.015 |
| code | 0.439 | **0.000** | 0.894 → 0.019 | 0.000 → 0.000 |
| qa | 0.000 | 0.000 | 0.379 → 0.053 | 0.178 → **0.000** |
| multi_tool | 0.000 | 0.000 | 0.433 → 0.158 | 0.208 → 0.025 |
| no_tool | 0.494 | 0.056 | 0.719 → 0.106 | 0.062 → 0.006 |
| **macro** | **0.1908** | **0.0125** | | |

Fifteen times the reward, and the mechanism is almost entirely the output
contract rather than capability. Without the examples the model essentially
stops emitting `<answer>` tags — coding goes from 0.894 to 0.019 — and the
recovery nudge fires on 93-100% of episodes in every category without
rescuing them. The coding score falls to exactly zero not because the model
forgot how to write Python but because it stops putting it where the verifier
looks.

The tool columns say something narrower and more interesting: **the search
example is the only reason QA ever searches at all** (0.178 → 0.000). Told in
prose that a search tool exists, a 0.5B model does not use it once in 264
episodes. Shown one example of a search call, it uses it in 18% of them — and
still scores 0.000, which is the same point the main results make.

This is why the control is prompted this way. A description-only baseline
would score 0.0125 and would hand module 5 a fifteen-fold improvement that is
mostly a lesson in tag syntax. The RL arm is prompted with exactly these same
turns, so whatever it gains is gained over a control that already knows the
format.

### Why batch size 32, on a card with room for 48

The first complete held-out run used batch 48 and took 45 minutes. The second
used batch 32 and did the same work in 18. That is backwards until you measure
what the card is actually doing (Qwen2.5-0.5B-Instruct, bf16,
`max_new_tokens=256`):

| batch | prompt tokens | cuda_reserved | tok/s | board W | |
| --- | --- | --- | --- | --- | --- |
| 32 | 1517 | 4.09 GB | 432.7 | 88.2 | fits in VRAM |
| 48 | 1517 | 5.61 GB | 458.1 | 91.5 | fits in VRAM |
| 48 | 3029 | 9.26 GB | 152.9 | 94.9 | spilled to system RAM |
| 48 | 4541 | 15.03 GB | 41.0 | 95.7 | spilled to system RAM |

**An 8GB card on Windows does not enforce 8GB.** WDDM backs the overflow with
system RAM, so exceeding the card never raises `CUDA out of memory` — it
silently costs up to 11x throughput. Batch 48 is genuinely faster than 32
while the transcripts are short, and collapses once a few episodes accumulate
enough tool output to push the batch past the card. Agent rollouts make that
unavoidable: transcript length is set by how many tools the policy decides to
call, so the widest batch in a run is not something the batch size alone
predicts.

Two things worth recording for anyone measuring this themselves. Board power
is *not* the signal — spilling draws 95 W, slightly more than a healthy run,
so a slow run looks perfectly busy. The check that works is
`torch.cuda.memory_reserved()` against the device's `total_memory`. And host
commit charge tracks the GPU reservation almost exactly, at
`commit ≈ cuda_reserved + 3.4 GB` across every configuration measured, which
is how a 0.5B model drove a 16GB machine to 97% of its commit limit.

Two plausible culprits were tested and neither was responsible: `stop_strings`
costs 0.20 GB rather than the gigabytes guessed, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` changes nothing measurable.
There is no leak — the reservation plateaus within about four calls in every
configuration.

### Reproducing

```bash
python -m baseline_agent --split heldout --rollouts 8 \
    --batch-size 32 --tool-workers 4 \
    --out artifacts/baseline/heldout.json
```

Generation is checkpointed to `<out>.rollouts.json` *before* scoring, and
`--score-rollouts <path>` re-scores that checkpoint with no GPU. That is not a
convenience: the first complete held-out run was destroyed by a `SandboxError`
raised during scoring, after all 40 minutes of generation had finished and
before anything was written. The transcripts and the rollouts checkpoint are
not committed — only `heldout.json`, which is what module 9 reads.
