"""nano-claude-code: a minimal viable implementation of Claude Code in Python."""

import logging

# LiteLLM logs noisy import-time warnings (e.g. missing optional bedrock deps).
# Quiet it before any submodule imports litellm.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

__version__ = "0.1.0"
