"""Tests for mindloop.memory."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    row_id = store.save(_summary("hello"), emb)
    assert row_id > 0
    assert store.count() == 1


def test_save_many(store: MemoryStore) -> None:
    summaries = [_summary("a"), _summary("b"), _summary("c")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    ids = store.save_many(summaries, embeddings)
    assert len(ids) == 3
    assert store.count() == 3


def test_search_empty(store: MemoryStore) -> None:
    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("anything")
    assert results == []


def test_search_returns_most_similar(store: MemoryStore) -> None:
    # Store three chunks with distinct embeddings.
    store.save(
        _summary("about cats", abstract="cats"),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    store.save(
        _summary("about dogs", abstract="dogs"),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    store.save(
        _summary("about fish", abstract="fish"),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    # Query embedding is close to "cats".
    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[0.9, 0.1, 0.0]], dtype=np.float32),
    ):
        results = store.search("cats", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_summary.abstract == "cats"
    assert results[0].score > results[1].score


def test_search_top_k_limits_results(store: MemoryStore) -> None:
    for i in range(10):
        store.save(_summary(f"chunk {i}"), np.array([float(i), 0.0], dtype=np.float32))

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[9.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("query", top_k=3)

    assert len(results) == 3


def test_deactivate_excludes_from_search(store: MemoryStore) -> None:
    id1 = store.save(
        _summary("cats", abstract="cats"), np.array([1.0, 0.0], dtype=np.float32)
    )
    store.save(
        _summary("dogs", abstract="dogs"), np.array([0.0, 1.0], dtype=np.float32)
    )

    store.deactivate([id1])
    assert store.count() == 1
    assert store.count(active_only=False) == 2

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("cats")
    # "cats" is deactivated, only "dogs" should appear.
    assert len(results) == 1
    assert results[0].chunk_summary.abstract == "dogs"


def test_activate_restores_to_search(store: MemoryStore) -> None:
    id1 = store.save(
        _summary("cats", abstract="cats"), np.array([1.0, 0.0], dtype=np.float32)
    )
    store.deactivate([id1])
    assert store.count() == 0

    store.activate([id1])
    assert store.count() == 1
