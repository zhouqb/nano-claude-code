"""Edit tool: exact string replacement within a file."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult


class EditInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to edit.")
    old_string: str = Field(description="Exact text to replace.")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring uniqueness."
    )


class EditTool(Tool):
    name = "Edit"
    description = (
        "Replace an exact string in a file. By default the old_string must appear "
        "exactly once; set replace_all to replace every occurrence."
    )
    input_schema = EditInput

    async def check_permissions(self, args: EditInput, context: ToolContext) -> PermissionDecision:
        path = self._resolve(args.file_path, context)
        return PermissionDecision(behavior="ask", prompt=f"Edit file {path}?")

    @staticmethod
    def _resolve(file_path: str, context: ToolContext) -> Path:
        path = Path(file_path)
        return path if path.is_absolute() else Path(context.cwd) / path

    async def call(self, args: EditInput, context: ToolContext) -> ToolResult:
        path = self._resolve(args.file_path, context)
        if not path.is_file():
            return ToolResult.fail(f"File not found: {path}")
        if args.old_string == args.new_string:
            return ToolResult.fail("old_string and new_string are identical; nothing to do.")

        try:
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()
        except OSError as exc:
            return ToolResult.fail(f"Could not read {path}: {exc}")

        count = content.count(args.old_string)
        if count == 0:
            return ToolResult.fail("old_string not found in file.")
        if count > 1 and not args.replace_all:
            return ToolResult.fail(
                f"old_string is not unique ({count} occurrences); pass replace_all=true "
                "or provide more surrounding context."
            )

        updated = content.replace(args.old_string, args.new_string)
        try:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(updated)
        except OSError as exc:
            return ToolResult.fail(f"Could not write {path}: {exc}")

        replaced = count if args.replace_all else 1
        return ToolResult(output=f"Replaced {replaced} occurrence(s) in {path}")
