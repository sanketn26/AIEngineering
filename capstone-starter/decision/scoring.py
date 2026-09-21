"""Fixed-answer scoring: one forward pass, a distribution over declared choices.

Gate 6 (stretch). The triage category is a bounded decision: the service
already knows every valid answer. Instead of generating text and parsing it,
render the choices as single-token labels (A, B, C, ...), read the model's
next-token logits for exactly those labels, and softmax over them only.

The probabilities are *relative to the declared choices*. They are not
calibrated accuracy — see ``decision.calibration`` for that.

Three backends share one interface:

* ``MockScorer`` — deterministic keyword logits. No GPU, no network. Default.
* ``TransformersScorer`` — a Hugging Face model in process. Opt-in; CPU, CUDA,
  or Apple MPS.
* ``MLXScorer`` — the same model through Apple MLX. Opt-in; Apple silicon only.

Application code sends semantic choices (``billing``) and receives semantic
choices back. Labels and token ids never leave this module.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Choice:
    value: str
    description: str


# ``other`` is the escape route. Restricted softmax always spends 100% of the
# mass on the listed choices, so without it a security incident or a prompt
# injection is forced into billing/shipping/account/product.
TRIAGE_CHOICES: tuple[Choice, ...] = (
    Choice("billing", "charges, refunds, invoices, and payment problems"),
    Choice("shipping", "delivery status, tracking, and lost or late packages"),
    Choice("account", "login, password, and account access problems"),
    Choice("product", "app errors, crashes, bugs, and feature questions"),
    Choice("other", "none of the above, security concerns, or unclear requests"),
)


class LabelTokenError(ValueError):
    """A label did not encode to exactly one token. Refuse to score."""


def build_prompt(ticket: str, choices: tuple[Choice, ...]) -> tuple[str, list[str]]:
    """Return the prompt and the label for each choice, in choice order."""
    if len(choices) > len(string.ascii_uppercase):
        raise ValueError("too many choices for single-letter labels")
    labels = list(string.ascii_uppercase[: len(choices)])
    lines = "\n".join(f"{lab} = {c.value}: {c.description}" for lab, c in zip(labels, choices))
    prompt = (
        f"Ticket:\n{ticket}\n\n"
        "Question:\nWhich category matches the ticket?\n\n"
        f"Allowed labels:\n{lines}\n\n"
        "Return only the label.\nLabel:\n"
    )
    return prompt, labels


def ticket_from_prompt(prompt: str) -> str:
    return prompt.split("Ticket:\n", 1)[1].split("\n\nQuestion:", 1)[0]


def restricted_softmax(logits: list[float]) -> list[float]:
    """Softmax over the selected logits only; every other token is excluded."""
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Scorer(Protocol):
    name: str

    def score(self, prompt: str, labels: list[str]) -> list[float]:
        """Probabilities for ``labels`` at the position after ``prompt``."""
        ...


_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("refund", "charge", "billed", "invoice", "payment", "duplicate", "subscription"),
    "shipping": ("package", "shipping", "delivery", "arrived", "tracking", "courier"),
    "account": ("password", "login", "log in", "locked", "2fa", "sign in"),
    "product": ("crash", "bug", "feature", "settings", "error", "freezes"),
    "other": ("hacked", "breach", "phishing", "ignore previous", "system prompt", "lawyer"),
}


class MockScorer:
    """Keyword-count logits. Stands in for a model's next-token scores.

    Mixed tickets produce close logits on purpose, so the routing policy has
    real near-ties to handle without a GPU.
    """

    name = "mock-scorer"

    def __init__(self, weight: float = 1.5):
        self._weight = weight

    def score(self, prompt: str, labels: list[str]) -> list[float]:
        text = ticket_from_prompt(prompt).lower()
        # Read the label -> choice mapping from the prompt, as a model would.
        meaning = dict(
            line.split(" = ", 1)[0:1] + [line.split(" = ", 1)[1].split(":", 1)[0]]
            for line in prompt.splitlines()
            if " = " in line
        )
        logits = []
        for value in (meaning[lab] for lab in labels):
            hits = sum(w in text for w in _KEYWORDS.get(value, ()))
            # "other" carries a small prior so a ticket with no signal lands there.
            prior = 0.5 if value == "other" else 0.0
            logits.append(prior + self._weight * hits)
        return restricted_softmax(logits)


class TransformersScorer:
    """Hugging Face causal LM, in process. Reads the next-token logits itself.

    One forward pass over the prompt; take the last position's vocabulary-sized
    logit vector; keep the label ids; softmax over those only. No decode loop.
    Install the optional deps first: ``pip install -r requirements-decision.txt``.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", *, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = f"transformers:{model}"
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device).eval()

    def label_token_ids(self, prompt: str, labels: list[str]) -> list[int]:
        return resolve_label_ids(self.tokenizer.encode, prompt, labels)

    def score(self, prompt: str, labels: list[str]) -> list[float]:
        prompt = render_chat(self.tokenizer, prompt)
        ids = self.label_token_ids(prompt, labels)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]  # one score per vocabulary token
        selected = [float(logits[i]) for i in ids]  # discard every other token
        return restricted_softmax(selected)

    def generate(self, prompt: str, *, max_new_tokens: int = 32) -> tuple[str, int]:
        """Ordinary greedy decoding on the same model, for the benchmark."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0])


class MLXScorer:
    """Apple MLX (``mlx-lm``). Same mechanism: one forward pass, read the label logits.

    Install with ``pip install -r requirements-decision-mlx.txt`` in its own
    environment; mlx-lm pulls a newer transformers than the Transformers path pins.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        self.name = f"mlx:{model}"
        self.device = "mlx"
        self.model, self.tokenizer = load(model)

    def label_token_ids(self, prompt: str, labels: list[str]) -> list[int]:
        return resolve_label_ids(self.tokenizer.encode, prompt, labels)

    def score(self, prompt: str, labels: list[str]) -> list[float]:
        prompt = render_chat(self.tokenizer, prompt)
        ids = self.label_token_ids(prompt, labels)
        tokens = self._mx.array(self.tokenizer.encode(prompt, add_special_tokens=False))[None]
        logits = self.model(tokens)[0, -1]  # one score per vocabulary token
        selected = [float(v) for v in logits[self._mx.array(ids)].tolist()]
        return restricted_softmax(selected)

    def generate(self, prompt: str, *, max_new_tokens: int = 32) -> tuple[str, int]:
        from mlx_lm import generate

        text = generate(self.model, self.tokenizer, prompt, max_tokens=max_new_tokens)
        return text, len(self.tokenizer.encode(text, add_special_tokens=False))


def render_chat(tokenizer, prompt: str) -> str:
    """Wrap in the model's chat template; the answer position opens the assistant turn.

    Instruct models are trained on their template. Scoring the raw prompt reads
    logits at a position the model never saw in training.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.removesuffix("Label:\n").rstrip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def resolve_label_ids(encode, prompt: str, labels: list[str]) -> list[int]:
    """Token id of each label *as the continuation of this prompt*.

    ``"A"`` and ``" A"`` are different tokens, and a tokenizer may merge the
    label with the prompt's trailing text. Encode prompt and prompt+label and
    require exactly one extra token with an unchanged prefix. Otherwise refuse.
    """
    base = encode(prompt, add_special_tokens=False)
    ids = []
    for label in labels:
        full = encode(prompt + label, add_special_tokens=False)
        if len(full) != len(base) + 1 or full[: len(base)] != base:
            raise LabelTokenError(f"{label!r} is not one clean continuation token after the prompt")
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise LabelTokenError(f"labels share token ids: {dict(zip(labels, ids))}")
    return ids
