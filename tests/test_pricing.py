"""Tests for the custom model registration in nano_claude.pricing."""

from __future__ import annotations

import litellm

from nano_claude.pricing import register_known_models


def test_registers_v4_flash_with_corrected_pricing():
    register_known_models()

    info = litellm.get_model_info("deepseek/deepseek-v4-flash")
    assert info.get("litellm_provider") == "deepseek"
    assert info.get("max_input_tokens") == 1_000_000
    assert info.get("supports_function_calling") is True

    # Corrected DeepSeek-V4-Flash pricing: $0.14 / $0.28 per 1M tokens.
    pin, pout = litellm.cost_per_token(
        model="deepseek/deepseek-v4-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert round(pin, 4) == 0.14
    assert round(pout, 4) == 0.28


def test_routes_to_deepseek_api_model_id():
    register_known_models()
    model, provider, *_ = litellm.get_llm_provider("deepseek/deepseek-v4-flash")
    assert provider == "deepseek"
    assert model == "deepseek-v4-flash"


def test_registration_is_idempotent():
    register_known_models()
    register_known_models()
    info = litellm.get_model_info("deepseek/deepseek-v4-flash")
    assert info.get("max_input_tokens") == 1_000_000
