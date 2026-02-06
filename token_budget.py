"""Token budgeting helpers for safe LLM requests."""
from __future__ import annotations

from dataclasses import dataclass


try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


@dataclass
class TokenBudget:
    prompt_tokens: int
    max_output_tokens: int


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count for the given text.

    Uses tiktoken when available; otherwise falls back to a rough heuristic.
    """
    if not text:
        return 0
    if tiktoken is None:
        # Conservative heuristic: 4 chars per token.
        return max(1, int(len(text) / 4))

    try:
        encoding = tiktoken.encoding_for_model(model or "gpt-4")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def compute_max_output_tokens(
    *,
    context_window: int,
    prompt_tokens: int,
    default_budget: int,
    safety_margin: int = 512,
    min_tokens: int = 256,
) -> int:
    """Compute a safe max output token budget.

    Ensures prompt + output + safety fits the model context window.
    """
    available = max(context_window - prompt_tokens - safety_margin, min_tokens)
    return max(min_tokens, min(default_budget, available))


def build_budget(
    *,
    text: str,
    model: str,
    context_window: int,
    default_budget: int,
    safety_margin: int,
    min_tokens: int = 256,
) -> TokenBudget:
    """Create a TokenBudget for a given prompt text."""
    prompt_tokens = estimate_tokens(text, model=model)
    max_output_tokens = compute_max_output_tokens(
        context_window=context_window,
        prompt_tokens=prompt_tokens,
        default_budget=default_budget,
        safety_margin=safety_margin,
        min_tokens=min_tokens,
    )
    return TokenBudget(prompt_tokens=prompt_tokens, max_output_tokens=max_output_tokens)


def truncate_text(text: str, max_tokens: int, model: str | None = None) -> str:
    """Truncate text to a maximum token length.

    Uses tiktoken when available; otherwise falls back to character slicing.
    """
    if not text or max_tokens <= 0:
        return ""
    if tiktoken is None:
        return text[: max_tokens * 4]

    try:
        encoding = tiktoken.encoding_for_model(model or "gpt-4")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])