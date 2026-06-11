"""Adapters between nano-claude and Google's Agent Development Kit (ADK).

Everything ADK-specific lives in this package: message-format converters,
the tool adapter, callback factories, the turn driver, and the JSONL session
service. The rest of the codebase keeps operating on OpenAI-format message
dicts; ADK types never leak past this boundary.

The ``google-adk`` dependency is pinned exactly (see pyproject.toml) because
the extension-point APIs used here have churned across minor versions.
"""
