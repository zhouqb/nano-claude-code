"""The Task tool — delegate to an isolated subagent.

Spawning is itself safe (every tool the subagent runs is still gated
individually), so ``check_permissions`` only validates the requested agent
exists. The tool's description enumerates the registered agents so the model
knows what it can delegate to.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nano_claude.subagents.loader import AGENT_REGISTRY
from nano_claude.subagents.runner import run_subagent_loop
from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult


class TaskInput(BaseModel):
    subagent_type: str = Field(description="Name of the agent to spawn (see the list above).")
    description: str = Field(description="A short 3-5 word label for the task.")
    prompt: str = Field(description="The complete, self-contained task for the subagent.")


class TaskTool(Tool):
    name = "Task"
    input_schema = TaskInput

    @property
    def description(self) -> str:
        lines = [
            "Delegate a task to an isolated subagent. The subagent runs with its "
            "own context and tool set; only its final summary returns to you, so "
            "use it for large, noisy explorations you don't want filling your "
            "context. Available agents (pass one as subagent_type):",
        ]
        for agent in AGENT_REGISTRY.values():
            lines.append(f"- {agent.name}: {agent.description}")
        return "\n".join(lines)

    async def check_permissions(self, args: TaskInput, context: ToolContext) -> PermissionDecision:
        if args.subagent_type not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY) or "(none)"
            return PermissionDecision(
                behavior="deny",
                reason=f"Unknown agent '{args.subagent_type}'. Available: {available}.",
            )
        return PermissionDecision(behavior="allow")

    async def call(self, args: TaskInput, context: ToolContext) -> ToolResult:
        agent = AGENT_REGISTRY[args.subagent_type]
        result = await run_subagent_loop(agent, args.prompt, context)
        return ToolResult(output=result.final_text or "(subagent produced no output)")
