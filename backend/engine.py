"""One base model on the GPU, with every served policy as a switchable adapter.

The baseline and each trained arm share the same 0.5B base weights; a trained
arm differs only by a LoRA adapter of a few megabytes. So the server loads the
base once, attaches every adapter to it by name, and picks one per request:
`set_adapter(name)` for a trained arm, `disable_adapter()` for the baseline.
Serving three policies costs about what serving one does, which matters on an
8GB card that also has to leave room for generation.

The adapters stay unmerged, as in `baseline_agent.HFPolicy`: that is the form
the evaluated numbers were produced in, and merging would fold them into bf16
weights and change the numerics of what is served.

torch, transformers and peft are imported inside `ServedPolicies.__init__`,
for the same reason `HFPolicy` does it: the app and its tests have to import on
a machine with none of them.
"""

import re
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from baseline_agent import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    STOP_STRINGS,
    HFPolicy,
    Message,
    describe_adapter,
)

from .app import BASELINE

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def check_adapters(adapters: dict[str, Path]) -> None:
    """Refuse a bad adapter set before the model is loaded, not after."""
    for name, path in adapters.items():
        if name == BASELINE:
            raise ValueError(f"{BASELINE!r} is the base model and cannot name an adapter")
        if not _NAME.match(name):
            raise ValueError(f"adapter name {name!r} must be lowercase letters, digits, - or _")
        if not (Path(path) / "adapter_model.safetensors").exists():
            # Adapters are gitignored, so a fresh checkout has none.
            raise ValueError(f"no adapter weights in {path}")


class ServedPolicies:
    def __init__(
        self,
        adapters: dict[str, Path],
        *,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 0,
    ) -> None:
        check_adapters(adapters)
        self.adapters = {name: Path(path) for name, path in adapters.items()}
        # One conversation per request, so batch size has nothing to batch.
        self.policy = HFPolicy(
            model_name, temperature=temperature, top_p=top_p, batch_size=1, seed=seed
        )

        if self.adapters:
            from peft import PeftModel

            (first, first_path), *rest = self.adapters.items()
            model = PeftModel.from_pretrained(
                self.policy.model, str(first_path), adapter_name=first
            )
            for name, path in rest:
                model.load_adapter(str(path), adapter_name=name)
            model.eval()
            self.policy.model = model

        # One GPU, and the active adapter is model-wide state: two requests
        # interleaving `set_adapter` and `generate` could each be answered by
        # the other's policy.
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        return [BASELINE, *self.adapters]

    def describe(self) -> dict[str, Any]:
        sampling = self.policy.describe()
        return {
            "model": sampling.pop("model"),
            "sampling": sampling,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "policies": [{"name": BASELINE, "adapter": None}]
            + [
                {"name": name, "adapter": describe_adapter(path)}
                for name, path in self.adapters.items()
            ],
        }

    def complete(self, policy: str, conversation: list[Message]) -> str:
        if policy not in self.names():
            raise KeyError(policy)
        with self._lock:
            if policy == BASELINE:
                scope = self.policy.model.disable_adapter() if self.adapters else nullcontext()
            else:
                # inference_mode, or peft marks the adapter trainable again.
                self.policy.model.set_adapter(policy, inference_mode=True)
                scope = nullcontext()
            with scope:
                return self.policy.generate(
                    [conversation], stop=STOP_STRINGS, max_new_tokens=DEFAULT_MAX_NEW_TOKENS
                )[0]
