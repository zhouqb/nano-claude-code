"""Register models whose pricing/metadata LiteLLM's bundled map gets wrong.

LiteLLM ships a static cost map. When a provider repoints an endpoint at a new
model, the bundled entry over-counts cost until LiteLLM catches up — e.g.
``deepseek-chat`` now serves DeepSeek-V4-Flash but is still priced as the older
V3 chat model, so ``/cost`` and the turn footer read ~2x high.

Rather than mutate the stale alias, we register the real model explicitly under
its own id (``deepseek/deepseek-v4-flash``) with correct pricing and metadata,
and default to that. Calling :func:`register_known_models` is idempotent.
"""

from __future__ import annotations

_PER_MILLION = 1_000_000

# Source: https://api-docs.deepseek.com/quick_start/pricing (verified 2026-06-03).
# `deepseek-chat`/`deepseek-reasoner` are aliases for V4-Flash, deprecated
# 2026-07-24; we register the explicit id so pricing stays correct past then.
_KNOWN_MODELS: dict[str, dict[str, object]] = {
    "deepseek/deepseek-v4-flash": {
        "litellm_provider": "deepseek",
        "mode": "chat",
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.14 / _PER_MILLION,  # cache miss
        "output_cost_per_token": 0.28 / _PER_MILLION,
        "cache_read_input_token_cost": 0.0028 / _PER_MILLION,
        "supports_function_calling": True,
        "supports_prompt_caching": True,
    },
}


def register_known_models() -> None:
    """Merge our model definitions into LiteLLM's cost map (best-effort)."""
    try:
        import litellm

        litellm.register_model(_KNOWN_MODELS)
    except Exception:  # noqa: BLE001 - pricing metadata is non-essential; never crash startup
        return
