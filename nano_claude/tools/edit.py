"""Edit tool: exact string replacement within a file."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from nano_claude.tools.base import (
    FileReadSnapshot,
    PermissionDecision,
    Tool,
    ToolContext,
    ToolResult,
)


class EditInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to edit.")
    old_string: str = Field(
        description=(
            "Exact text to replace. To INSERT text, anchor on existing nearby "
            "text: set old_string to that anchor and new_string to the anchor "
            "plus your inserted text. To create a whole new file, use Write."
        )
    )
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False,
        description=(
            "Default false: old_string must match exactly once, otherwise the "
            "edit errors — this guards against unintentionally changing the wrong "
            "occurrence. Set true to deliberately replace every occurrence "
            "(e.g. renaming a symbol)."
        ),
    )


class EditTool(Tool):
    name = "Edit"
    description = (
        "Replace an exact string in a file. By default the old_string must appear "
        "exactly once; set replace_all to replace every occurrence. This tool "
        "replaces text — to insert, anchor on surrounding text; to create a file, "
        "use Write."
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
        if args.old_string == "":
            # An empty old_string matches between every character, so it can't
            # target an insertion point. Reject it with actionable guidance
            # rather than failing later with a confusing "not unique" error.
            return ToolResult.fail(
                "old_string is empty. To insert text, anchor on existing nearby "
                "text; to create a new file, use Write."
            )
        if args.old_string == args.new_string:
            return ToolResult.fail("old_string and new_string are identical; nothing to do.")

        try:
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()
        except OSError as exc:
            return ToolResult.fail(f"Could not read {path}: {exc}")

        stale_error = self._stale_write_error(path, context)
        if stale_error is not None:
            return ToolResult.fail(stale_error)

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

        context.read_file_state[str(path)] = FileReadSnapshot(
            content=updated,
            timestamp=path.stat().st_mtime,
            offset=None,
            limit=None,
            is_partial_view=False,
        )
        replaced = count if args.replace_all else 1
        return ToolResult(output=f"Replaced {replaced} occurrence(s) in {path}")

    @staticmethod
    def _stale_write_error(path: Path, context: ToolContext) -> str | None:
        # Mirrors Claude Code's guards: the file must have been read at least
        # once this session, and must not have changed on disk since. A partial
        # (offset/limit or truncated) read counts as a read — correctness of the
        # edit itself is enforced by the old_string found/unique checks below, so
        # there is no need to have seen the whole file. Staleness is keyed on the
        # file's mtime (not full-content equality, which a partial snapshot can't
        # satisfy since it only holds the slice that was shown).
        snapshot = context.read_file_state.get(str(path))
        if snapshot is None:
            return "File has not been read yet. Read it first before editing it."
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return None
        if current_mtime != snapshot.timestamp:
            return "File has been modified since read. Read it again before attempting to edit it."
        return None
