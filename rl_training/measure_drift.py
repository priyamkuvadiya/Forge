"""Did training cost the model anything it had before?

GRPO optimises reward on the task suite and nothing else. The KL term keeps
the policy near the base model *on the prompts it trains on*, which says
nothing directly about everything else the model knew: a run can raise
reward while quietly forgetting who wrote Romeo and Juliet, or while losing
the ability to follow a plain formatting instruction. The reward curve cannot
see either, and neither can module 9, which scores the task suite.

So this compares an adapter against the base model on material the training
never touches - plain chat, the model's default system prompt, no tools, no
few-shot prefix - in three kinds of probe with programmatic checks:

  * **knowledge**: 30 well-known facts with short unambiguous answers.
  * **reasoning**: 15 small problems a 0.5B model can do without a calculator.
  * **instruction**: 10 prompts whose format constraint can be checked
    exactly (one word, all capitals, valid JSON, ...).

Decoding is greedy, so for a fixed prompt a response changes only if the
weights did. The report counts flips in both directions per probe rather than
only net accuracy, because "lost 3, gained 3" and "no change" are different
findings with the same total.

Beside that, **drift**: the base model's own greedy responses are teacher-
forced through both the base and the adapter, and the token-level
KL(adapter || base) is averaged over response tokens. That is a direct measure
of how far the adapter has moved on text it was never trained on, and unlike
the flip counts it is continuous, so it can show movement before any answer
changes.

A probe that cannot fail proves nothing, so `--noise-scale` builds an adapter
with random LoRA weights of a chosen size and runs the same report on it. Zero
noise must reproduce the base exactly (KL 0, no flips); enough noise must show
damage. Both are checked before any trained adapter's clean report is believed.

    python -m rl_training.measure_drift --adapter artifacts/rl/pilot/adapter-final
    python -m rl_training.measure_drift --noise-scale 0 --noise-scale 0.02
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# 256, not 128: at 128 the base model's LaTeX-formatted working was cut off
# before it reached an answer on two reasoning probes, which scored truncation
# as a wrong answer.
MAX_NEW_TOKENS = 256

# --- probes ------------------------------------------------------------------
#
# Self-authored for this file and deliberately generic: none of them is a task
# from the suite, and none needs a tool. Knowledge answers are lists of
# accepted spellings, matched as whole words.

KNOWLEDGE = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("At sea level, what is the boiling point of water in degrees Celsius?", ["100"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("What is the chemical formula for water?", ["h2o", "h₂o"]),
    ("What language is mainly spoken in Brazil?", ["portuguese"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]),
    ("How many legs does a spider have?", ["8", "eight"]),
    ("What is the freezing point of water in degrees Fahrenheit?", ["32"]),
    ("What is the capital of Italy?", ["rome"]),
    ("Which gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2", "co₂"]),
    ("Who wrote the novel Pride and Prejudice?", ["austen"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("What is the currency of Japan?", ["yen"]),
    ("What is the tallest mountain in the world above sea level?", ["everest"]),
    ("How many days are there in a leap year?", ["366"]),
    ("Which organ pumps blood through the human body?", ["heart"]),
    ("What is the capital of Canada?", ["ottawa"]),
    ("What is the longest river in Africa?", ["nile"]),
    ("Who was the first person to walk on the Moon?", ["armstrong"]),
    ("What is the hardest natural substance?", ["diamond"]),
    ("Which element has the chemical symbol H?", ["hydrogen"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("Which planet is closest to the Sun?", ["mercury"]),
]

# Scored on the LAST number in the response, or on accepted words for the few
# whose answer is not a number. Last, not any: a response that lists several
# candidate numbers should not be credited for one of them.
REASONING = [
    ("What is 7 + 5?", "12"),
    ("What is 9 times 6?", "54"),
    ("What is 100 minus 37?", "63"),
    ("Tom has 3 apples and buys 4 more. How many apples does he have now?", "7"),
    ("A train travels 60 km every hour. How many km does it travel in 3 hours?", "180"),
    ("What is half of 50?", "25"),
    ("What is 15 divided by 3?", "5"),
    ("A box holds 12 eggs. How many eggs are in 3 boxes?", "36"),
    ("What is 2 to the power of 5?", "32"),
    ("What number comes next in the sequence 2, 4, 6, 8?", "10"),
    ("A shirt costs 20 dollars and is 50% off. What is the sale price in dollars?", "10"),
    ("How many minutes are there in 2 hours?", "120"),
    ("If today is Monday, what day of the week is it in two days?", ["wednesday"]),
    ("Sarah is older than Tom, and Tom is older than Jim. Who is the youngest?", ["jim"]),
    # Worded so the last-number rule cannot fail a correct answer that goes on
    # to mention the smaller number ("0.5 is larger than 0.25").
    ("Which is larger, 0.5 or 0.25? Reply with the larger number only.", "0.5"),
]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _strip_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.S)
    return fenced.group(1) if fenced else text


def _is_json_name_ada(text: str) -> bool:
    try:
        value = json.loads(_strip_fence(text))
    except ValueError:
        return False
    return isinstance(value, dict) and value.get("name") == "Ada"


def _three_items(text: str) -> bool:
    text = text.strip().rstrip(".")
    if "\n" in text:
        return False
    items = [item.strip() for item in text.split(",")]
    return len(items) == 3 and all(items) and all(len(_words(i)) <= 3 for i in items)


INSTRUCTION_CHECKS: dict[str, Callable[[str], bool]] = {
    "one_word": lambda t: len(_words(t)) == 1,
    "two_words": lambda t: len(_words(t)) == 2,
    "all_caps": lambda t: any(c.isalpha() for c in t) and t.upper() == t,
    "three_items": _three_items,
    "starts_yes_no": lambda t: bool(re.match(r"^\W*(yes|no)\b", t.strip(), re.I)),
    "under_ten_words": lambda t: 0 < len(_words(t)) < 10,
    "json_name_ada": _is_json_name_ada,
    "only_42": lambda t: t.strip().rstrip(".") == "42",
    "ends_exclaim": lambda t: t.strip().endswith("!"),
}

INSTRUCTION = [
    ("Name one color. Reply with exactly one word.", "one_word"),
    ("Write the word hello in all capital letters and nothing else.", "all_caps"),
    ("List three fruits separated by commas on a single line, with no other text.", "three_items"),
    ("Is the sky blue on a clear day? Start your answer with Yes or No.", "starts_yes_no"),
    ("Describe the ocean in fewer than ten words.", "under_ten_words"),
    ('Return a JSON object with a key "name" whose value is "Ada". Output only the JSON.', "json_name_ada"),
    ("Reply with only the number 42.", "only_42"),
    ("Say goodbye in exactly two words.", "two_words"),
    ("Write one sentence about cats that ends with an exclamation mark.", "ends_exclaim"),
    ("Give a word that rhymes with cat. Reply with one word only.", "one_word"),
]


def build_probes() -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for index, (prompt, accept) in enumerate(KNOWLEDGE):
        probes.append({"id": f"knowledge-{index:02d}", "kind": "knowledge",
                       "prompt": f"Answer briefly. {prompt}", "accept": accept})
    for index, (prompt, answer) in enumerate(REASONING):
        probes.append({"id": f"reasoning-{index:02d}", "kind": "reasoning",
                       "prompt": f"Answer briefly. {prompt}", "accept": answer})
    for index, (prompt, check) in enumerate(INSTRUCTION):
        probes.append({"id": f"instruction-{index:02d}", "kind": "instruction",
                       "prompt": prompt, "check": check})
    return probes


def _contains_word(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", text.lower()) is not None


def _numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))


def _answer_number(response: str, prompt: str) -> str | None:
    """The number the response commits to.

    The last number in the response *that is not restating the question*: "There
    are 120 minutes in 2 hours" answers 120, and a plain last-number rule would
    score it as 2 - which is how the first version of this probe marked the
    base model wrong on a question it had answered correctly. When every number
    in the response also appears in the prompt ("0.5" for "0.5 or 0.25?") the
    last one is taken as-is.
    """
    in_prompt = {float(n) for n in _numbers(prompt)}
    numbers = _numbers(response)
    fresh = [n for n in numbers if float(n) not in in_prompt]
    candidates = fresh or numbers
    return candidates[-1] if candidates else None


def score_probe(probe: dict[str, Any], response: str) -> bool:
    """Did this response pass this probe. Pure function, no torch."""
    kind = probe["kind"]
    if kind == "knowledge":
        return any(_contains_word(response, phrase) for phrase in probe["accept"])
    if kind == "reasoning":
        accept = probe["accept"]
        if isinstance(accept, list):
            return any(_contains_word(response, phrase) for phrase in accept)
        found = _answer_number(response, probe["prompt"])
        return found is not None and float(found) == float(accept)
    if kind == "instruction":
        return INSTRUCTION_CHECKS[probe["check"]](response)
    raise ValueError(f"unknown probe kind {kind!r}")


def compare(probes: list[dict[str, Any]], base: list[str], other: list[str]) -> dict[str, Any]:
    """Per-kind accuracy for both, and the flips between them. No torch."""
    by_kind: dict[str, dict[str, Any]] = {}
    for probe, before, after in zip(probes, base, other):
        bucket = by_kind.setdefault(
            probe["kind"], {"probes": 0, "base": 0, "adapter": 0, "lost": [], "gained": [], "changed": 0}
        )
        passed_before, passed_after = score_probe(probe, before), score_probe(probe, after)
        bucket["probes"] += 1
        bucket["base"] += passed_before
        bucket["adapter"] += passed_after
        bucket["changed"] += before != after
        if passed_before and not passed_after:
            bucket["lost"].append(probe["id"])
        if passed_after and not passed_before:
            bucket["gained"].append(probe["id"])
    return by_kind


# --- the GPU half --------------------------------------------------------------


def _generate(model, tokenizer, prompts: list[str], device: str) -> tuple[list[str], list[list[int]], list[list[int]]]:
    import torch

    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.pad_token_id)
    width = encoded["input_ids"].shape[1]
    responses, prompt_ids, response_ids = [], [], []
    for row in range(len(prompts)):
        ids = out[row, width:].tolist()
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()
        mask = encoded["attention_mask"][row].bool()
        prompt_ids.append(encoded["input_ids"][row][mask].tolist())
        response_ids.append(ids)
        responses.append(tokenizer.decode(ids, skip_special_tokens=True))
    return responses, prompt_ids, response_ids


def _response_logprobs(model, prompt_ids: list[int], response_ids: list[int], device: str):
    import torch

    ids = torch.tensor([prompt_ids + response_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    # Position t predicts token t+1, so the response tokens are predicted by the
    # positions from the last prompt token up to the second-to-last token.
    start = len(prompt_ids) - 1
    return torch.log_softmax(logits[start : start + len(response_ids)], dim=-1)


def drift(model, prompt_ids, response_ids, device: str) -> dict[str, Any]:
    """Token-level KL(adapter || base) on the base model's own responses."""
    import torch

    per_probe, total_kl, total_tokens = [], 0.0, 0
    for prompt, response in zip(prompt_ids, response_ids):
        if not response:
            per_probe.append(0.0)
            continue
        adapter = _response_logprobs(model, prompt, response, device)
        with model.disable_adapter():
            base = _response_logprobs(model, prompt, response, device)
        kl = (adapter.exp() * (adapter - base)).sum(dim=-1)
        per_probe.append(float(kl.mean()))
        total_kl += float(kl.sum())
        total_tokens += len(response)
        del adapter, base
        if device == "cuda":
            torch.cuda.empty_cache()
    return {
        "mean_kl_per_token": total_kl / total_tokens if total_tokens else 0.0,
        "max_probe_kl": max(per_probe) if per_probe else 0.0,
        "tokens": total_tokens,
    }


def evaluate(model, tokenizer, probes, device: str, label: str) -> dict[str, Any]:
    prompts = [probe["prompt"] for probe in probes]
    with model.disable_adapter():
        base_responses, prompt_ids, response_ids = _generate(model, tokenizer, prompts, device)
    adapter_responses, _, _ = _generate(model, tokenizer, prompts, device)
    by_kind = compare(probes, base_responses, adapter_responses)
    kl = drift(model, prompt_ids, response_ids, device)
    lost = sum(len(b["lost"]) for b in by_kind.values())
    gained = sum(len(b["gained"]) for b in by_kind.values())
    print(
        f"{label}: lost {lost}, gained {gained}, changed "
        f"{sum(b['changed'] for b in by_kind.values())}/{len(probes)}, "
        f"KL/token {kl['mean_kl_per_token']:.5f}  "
        + "  ".join(f"{k} {b['base']}->{b['adapter']}/{b['probes']}" for k, b in by_kind.items()),
        file=sys.stderr, flush=True,
    )
    return {
        "label": label,
        "by_kind": by_kind,
        "lost": lost,
        "gained": gained,
        "drift": kl,
        "responses": [
            {"id": p["id"], "base": b, "adapter": a, "base_pass": score_probe(p, b), "adapter_pass": score_probe(p, a)}
            for p, b, a in zip(probes, base_responses, adapter_responses)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl_training.measure_drift")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, action="append", default=[],
                        help="a saved LoRA adapter directory; repeatable")
    parser.add_argument("--noise-scale", type=float, action="append", default=[],
                        help="self-test: a fresh adapter with LoRA B ~ N(0, scale); repeatable")
    parser.add_argument("--out", type=Path, default=Path("artifacts/rl/drift.json"))
    arguments = parser.parse_args(argv)
    if not arguments.adapter and not arguments.noise_scale:
        parser.error("give at least one --adapter or --noise-scale")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rl_training.policy import attach_adapter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    probes = build_probes()

    def fresh_base():
        return AutoModelForCausalLM.from_pretrained(
            arguments.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()

    reports = []
    for scale in arguments.noise_scale:
        model = attach_adapter(fresh_base(), rank=16)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    noise = torch.randn(parameter.shape, generator=generator) * scale
                    parameter.copy_(noise.to(parameter.dtype))
        model.eval()
        reports.append({"source": f"noise:{scale}", **evaluate(model, tokenizer, probes, device, f"noise {scale}")})
        del model

    for path in arguments.adapter:
        model = PeftModel.from_pretrained(fresh_base(), str(path)).eval()
        reports.append({"source": str(path), **evaluate(model, tokenizer, probes, device, str(path))})
        del model

    payload = {
        "generated_by": "rl_training.measure_drift",
        "model": arguments.model,
        "decoding": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS},
        "probes": {kind: sum(1 for p in probes if p["kind"] == kind) for kind in ("knowledge", "reasoning", "instruction")},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reports": reports,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {arguments.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
