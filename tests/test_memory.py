"""Tests for mindloop.memory."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore
from mindloop.summarizer import ChunkSummary


def _chunk(text: str) -> Chunk:
    return Chunk(turns=[Turn(timestamp=datetime.min, role="You", text=text)])


def _summary(text: str, abstract: str = "abs", summary: str = "sum") -> ChunkSummary:
    return ChunkSummary(chunk=_chunk(text), abstract=abstract, summary=summary)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


def test_save_and_count(store: MemoryStore) -> None:
    emb = [1.0, 0.0, 0.0]
    row_id = store.save(_summary("hello"), emb)
    assert row_id > 0
    assert store.count() == 1


def test_save_many(store: MemoryStore) -> None:
    summaries = [_summary("a"), _summary("b"), _summary("c")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    ids = store.save_many(summaries, embeddings)
    assert len(ids) == 3
    assert store.count() == 3


def test_search_empty(store: MemoryStore) -> None:
    with patch("mindloop.memory.get_embeddings", return_value=[[1.0, 0.0]]):
        results = store.search("anything")
    assert results == []


def test_search_returns_most_similar(store: MemoryStore) -> None:
    # Store three chunks with distinct embeddings.
    store.save(_summary("about cats", abstract="cats"), [1.0, 0.0, 0.0])
    store.save(_summary("about dogs", abstract="dogs"), [0.0, 1.0, 0.0])
    store.save(_summary("about fish", abstract="fish"), [0.0, 0.0, 1.0])

    # Query embedding is close to "cats".
    with patch("mindloop.memory.get_embeddings", return_value=[[0.9, 0.1, 0.0]]):
        results = store.search("cats", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_summary.abstract == "cats"
    assert results[0].score > results[1].score


def test_search_top_k_limits_results(store: MemoryStore) -> None:
    for i in range(10):
        store.save(_summary(f"chunk {i}"), [float(i), 0.0])

    with patch("mindloop.memory.get_embeddings", return_value=[[9.0, 0.0]]):
        results = store.search("query", top_k=3)

    assert len(results) == 3
