"""Tests for the built-in tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nano_claude.permissions.modes import PermissionMode
from nano_claude.tools.base import ToolContext
from nano_claude.tools.bash import BashInput, BashTool, is_dangerous
from nano_claude.tools.edit import EditInput, EditTool
from nano_claude.tools.glob_tool import GlobInput, GlobTool
from nano_claude.tools.grep import GrepInput, GrepTool
from nano_claude.tools.read import ReadInput, ReadTool
from nano_claude.tools.registry import BASE_TOOLS, get_tool
from nano_claude.tools.write import WriteInput, WriteTool


def ctx(cwd: str) -> ToolContext:
    return ToolContext(
        cwd=str(cwd),
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.DEFAULT,
        output_dir=Path(str(cwd)) / "_overflow",
    )


async def read_first(file_path: Path, context: ToolContext) -> None:
    result = await ReadTool().call(ReadInput(file_path=str(file_path)), context)
    assert not result.is_error


# --- registry ---------------------------------------------------------------


def test_registry_exposes_six_tools():
    assert len(BASE_TOOLS) == 6
    assert get_tool("Read") is not None
    assert get_tool("Nonexistent") is None


def test_api_schema_shape():
    schema = ReadTool().to_api_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "Read"
    assert "file_path" in schema["function"]["parameters"]["properties"]


# --- Read -------------------------------------------------------------------


async def test_read_returns_numbered_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("alpha\nbeta\n")
    result = await ReadTool().call(ReadInput(file_path=str(f)), ctx(tmp_path))
    assert not result.is_error
    assert "1\talpha" in result.output
    assert "2\tbeta" in result.output


async def test_read_missing_file_errors(tmp_path):
    result = await ReadTool().call(ReadInput(file_path=str(tmp_path / "no.txt")), ctx(tmp_path))
    assert result.is_error


async def test_read_offset_and_limit(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("\n".join(str(i) for i in range(1, 11)))
    result = await ReadTool().call(ReadInput(file_path=str(f), offset=2, limit=3), ctx(tmp_path))
    assert "3\t3" in result.output
    assert "truncated" in result.output


# --- Write ------------------------------------------------------------------


async def test_write_creates_file(tmp_path):
    target = tmp_path / "sub" / "new.txt"
    result = await WriteTool().call(
        WriteInput(file_path=str(target), content="hello\nworld\n"), ctx(tmp_path)
    )
    assert not result.is_error
    assert target.read_text() == "hello\nworld\n"


async def test_write_permission_is_ask(tmp_path):
    decision = await WriteTool().check_permissions(
        WriteInput(file_path=str(tmp_path / "x.txt"), content="x"), ctx(tmp_path)
    )
    assert decision.behavior == "ask"


# --- Edit -------------------------------------------------------------------


async def test_edit_replaces_unique_string(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("foo bar baz")
    context = ctx(tmp_path)
    await read_first(f, context)
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="bar", new_string="qux"), context
    )
    assert not result.is_error
    assert f.read_text() == "foo qux baz"


async def test_edit_non_unique_without_replace_all_errors(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x x x")
    context = ctx(tmp_path)
    await read_first(f, context)
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="x", new_string="y"), context
    )
    assert result.is_error
    assert "not unique" in result.output


async def test_edit_replace_all(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x x x")
    context = ctx(tmp_path)
    await read_first(f, context)
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="x", new_string="y", replace_all=True),
        context,
    )
    assert not result.is_error
    assert f.read_text() == "y y y"


async def test_edit_empty_old_string_errors(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="", new_string="x"), ctx(tmp_path)
    )
    assert result.is_error
    assert "empty" in result.output
    assert f.read_text() == "hello"  # unchanged


async def test_edit_missing_string_errors(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    context = ctx(tmp_path)
    await read_first(f, context)
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="absent", new_string="z"), context
    )
    assert result.is_error


async def test_edit_requires_prior_full_read(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="hello", new_string="hi"), ctx(tmp_path)
    )
    assert result.is_error
    assert "Read it first" in result.output


async def test_edit_rejects_file_modified_since_read(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    context = ctx(tmp_path)
    await read_first(f, context)
    f.write_text("hello user edit")
    result = await EditTool().call(
        EditInput(file_path=str(f), old_string="hello", new_string="hi"), context
    )
    assert result.is_error
    assert "modified since read" in result.output


async def test_write_existing_requires_prior_full_read(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("old")
    result = await WriteTool().call(WriteInput(file_path=str(f), content="new"), ctx(tmp_path))
    assert result.is_error
    assert "Read it first" in result.output
    assert f.read_text() == "old"


async def test_write_existing_succeeds_after_read(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("old")
    context = ctx(tmp_path)
    await read_first(f, context)
    result = await WriteTool().call(WriteInput(file_path=str(f), content="new"), context)
    assert not result.is_error
    assert f.read_text() == "new"


# --- Glob -------------------------------------------------------------------


async def test_glob_finds_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    result = await GlobTool().call(GlobInput(pattern="*.py"), ctx(tmp_path))
    assert "a.py" in result.output
    assert "b.txt" not in result.output


# --- Grep -------------------------------------------------------------------


async def test_grep_finds_matches(tmp_path):
    (tmp_path / "a.txt").write_text("needle here\nhaystack\n")
    result = await GrepTool().call(GrepInput(pattern="needle"), ctx(tmp_path))
    assert not result.is_error
    assert "needle" in result.output


async def test_grep_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("haystack\n")
    result = await GrepTool().call(GrepInput(pattern="needle"), ctx(tmp_path))
    assert not result.is_error
    assert "No matches" in result.output


# --- Bash -------------------------------------------------------------------


async def test_bash_runs_command(tmp_path):
    result = await BashTool().call(BashInput(command="echo hi"), ctx(tmp_path))
    assert not result.is_error
    assert result.output == "hi"


async def test_bash_nonzero_exit_is_error(tmp_path):
    result = await BashTool().call(BashInput(command="exit 3"), ctx(tmp_path))
    assert result.is_error
    assert "exit code 3" in result.output


async def test_bash_timeout(tmp_path):
    result = await BashTool().call(BashInput(command="sleep 5", timeout=1), ctx(tmp_path))
    assert result.is_error
    assert "timed out" in result.output


async def test_bash_dangerous_denied(tmp_path):
    decision = await BashTool().check_permissions(BashInput(command="rm -rf /"), ctx(tmp_path))
    assert decision.behavior == "deny"


def test_is_dangerous_patterns():
    assert is_dangerous("rm -rf /")
    assert is_dangerous("sudo rm -rf /")
    assert not is_dangerous("rm -rf ./build")
    assert not is_dangerous("echo hello")


# --- output overflow (spill to disk) ----------------------------------------


def _spill_files(tmp_path):
    return list((tmp_path / "_overflow").glob("*.txt"))


async def test_bash_spills_large_output(tmp_path):
    # Produce well over MAX_OUTPUT_BYTES (60k) of stdout.
    result = await BashTool().call(
        BashInput(command="python3 -c \"print('x' * 70000)\""), ctx(tmp_path)
    )
    assert not result.is_error
    assert "output truncated" in result.output
    assert "Full output saved to" in result.output
    files = _spill_files(tmp_path)
    assert len(files) == 1
    assert len(files[0].read_text()) >= 70000  # full output preserved


async def test_grep_spills_large_output(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("".join(f"match line number {i}\n" for i in range(4000)))
    result = await GrepTool().call(GrepInput(pattern="match"), ctx(tmp_path))
    assert not result.is_error
    assert "Full output saved to" in result.output
    assert len(_spill_files(tmp_path)) == 1


async def test_glob_spills_when_over_max_results(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(1100):
        (src / f"f{i}.py").write_text("")
    result = await GlobTool().call(GlobInput(pattern="src/*.py"), ctx(tmp_path))
    assert "matches" in result.output
    assert "Full output saved to" in result.output
    files = _spill_files(tmp_path)
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == 1100  # every match preserved


async def test_no_spill_when_output_small(tmp_path):
    result = await BashTool().call(BashInput(command="echo hi"), ctx(tmp_path))
    assert "truncated" not in result.output
    assert not (tmp_path / "_overflow").exists()
