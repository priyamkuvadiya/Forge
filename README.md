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

## Module 1: from-scratch transformer

`transformer_scratch/` — a GPT-style decoder-only transformer (causal
self-attention, learned positional embeddings, pre-LN residual blocks) built
directly on `torch.nn`, trained on the character-level tinyshakespeare
corpus. Exists solely to demonstrate transformer internals are understood
from scratch; it is not used anywhere else in this project.

Results: `[[RUN TO FILL IN]]`
