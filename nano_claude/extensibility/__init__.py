"""Extensibility subsystems: hooks, skills, MCP servers, and plugins.

Each mechanism feeds the same agent loop — hooks fire at lifecycle events,
skills inject ``/command`` prompts, MCP servers contribute external tools, and
plugins bundle the other three. They are discovered once at startup and wired
before the first turn.
"""
