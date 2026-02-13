"""Tests for mindloop.cli.build."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mindloop.chunker import Chunk
from mindloop.cli.build import main, process_file
from mindloop.memory import MemoryStore
from mindloop.summarizer import ChunkSummary


def _mock_summarize_chunks(chunks: list[Chunk], **_kw: object) -> list[ChunkSummary]:
    return [ChunkSummary(chunk=c, abstract="abs", summary="sum") for c in chunks]


def _mock_save_memory(
    store: MemoryStore,
    text: str,
    abstract: str,
    summary: str,
    model: str = "openrouter/free",
    **_kw: object,
) -> int:
    return 1


@patch("mindloop.cli.build.save_memory", side_effect=_mock_save_memory)
@patch("mindloop.cli.build.summarize_chunks", side_effect=_mock_summarize_chunks)
def test_process_md_file(
    mock_summarize: MagicMock, mock_save: MagicMock, tmp_path: Path
) -> None:
    """Markdown files use parse_turns_md."""
    md = tmp_path / "test.md"
    md.write_text("# Heading A\nSome content.\n\n# Heading B\nMore content.\n")

    store = MemoryStore(db_path=tmp_path / "test.db")
    n = process_file(md, store, "openrouter/free")

    assert n > 0
    assert mock_summarize.call_count == 1
    assert mock_save.call_count == n
    store.close()


@patch("mindloop.cli.build.save_memory", side_effect=_mock_save_memory)
@patch("mindloop.cli.build.summarize_chunks", side_effect=_mock_summarize_chunks)
def test_process_jsonl_file(
    mock_summarize: MagicMock, mock_save: MagicMock, tmp_path: Path
) -> None:
    """Non-md files use parse_turns."""
    import json

    jsonl = tmp_path / "test.jsonl"
    lines = [
        {"role": "user", "content": "hello", "timestamp": "2025-01-01T00:00:00"},
        {
            "role": "assistant",
            "content": "hi there",
            "timestamp": "2025-01-01T00:00:01",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(entry) for entry in lines) + "\n")

    store = MemoryStore(db_path=tmp_path / "test.db")
    n = process_file(jsonl, store, "openrouter/free")

    assert n > 0
    assert mock_summarize.call_count == 1
    assert mock_save.call_count == n
    store.close()


@patch("mindloop.cli.build.save_memory", side_effect=_mock_save_memory)
@patch("mindloop.cli.build.summarize_chunks", side_effect=_mock_summarize_chunks)
def test_files_processed_in_sorted_order(
    mock_summarize: MagicMock, mock_save: MagicMock, tmp_path: Path
) -> None:
    """Files are processed in sorted (deterministic) order."""
    processed: list[str] = []
    original_process = process_file

    def _tracking_process(path: Path, store: Any, model: str, **kw: Any) -> int:
        processed.append(path.name)
        return original_process(path, store, model, **kw)

    # Create files in reverse alphabetical order.
    for name in ["c.md", "a.md", "b.md"]:
        (tmp_path / name).write_text(f"# {name}\nContent for {name}.\n")

    with patch("mindloop.cli.build.process_file", side_effect=_tracking_process):
        with patch(
            "sys.argv",
            ["mindloop-build", str(tmp_path / "*.md"), "--db", str(tmp_path / "t.db")],
        ):
            main()

    assert processed == ["a.md", "b.md", "c.md"]


def test_empty_glob_no_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty glob pattern produces a message but no error."""
    with patch(
        "sys.argv",
        ["mindloop-build", str(tmp_path / "*.xyz"), "--db", str(tmp_path / "t.db")],
    ):
        main()

    captured = capsys.readouterr()
    assert "No files match" in captured.out


@patch("mindloop.cli.build.save_memory", side_effect=_mock_save_memory)
@patch("mindloop.cli.build.summarize_chunks", side_effect=_mock_summarize_chunks)
def test_directories_are_skipped(
    mock_summarize: MagicMock, mock_save: MagicMock, tmp_path: Path
) -> None:
    """Directories matching the glob are skipped."""
    subdir = tmp_path / "subdir.md"
    subdir.mkdir()
    real_file = tmp_path / "real.md"
    real_file.write_text("# Title\nContent.\n")

    with patch(
        "sys.argv",
        ["mindloop-build", str(tmp_path / "*.md"), "--db", str(tmp_path / "t.db")],
    ):
        main()

    # Only the real file should be processed.
    assert mock_summarize.call_count > 0
    assert mock_save.call_count > 0


@patch("mindloop.cli.build.save_memory", side_effect=_mock_save_memory)
@patch("mindloop.cli.build.summarize_chunks", side_effect=_mock_summarize_chunks)
def test_verbose_prints_stages(
    mock_summarize: MagicMock,
    mock_save: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbose mode prints original, summarized, and saved stages."""
    md = tmp_path / "test.md"
    md.write_text("# Heading\nSome content.\n")

    store = MemoryStore(db_path=tmp_path / "test.db")
    process_file(md, store, "openrouter/free", verbose=True)
    store.close()

    out = capsys.readouterr().out
    assert "Original chunks:" in out
    assert "Summarized [1]:" in out
    assert "Saved [1]" in out
