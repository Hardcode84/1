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


def test_specificity_no_chunks(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    assert store.specificity(emb) == 1.0


def test_specificity_no_neighbors(store: MemoryStore) -> None:
    # Store a chunk orthogonal to the query.
    store.save(_summary("a"), np.array([0.0, 1.0], dtype=np.float32))
    emb = np.array([1.0, 0.0], dtype=np.float32)
    assert store.specificity(emb) == 1.0


def test_specificity_all_neighbors(store: MemoryStore) -> None:
    # Store chunks identical to the query.
    for _ in range(5):
        store.save(_summary("a"), np.array([1.0, 0.0], dtype=np.float32))
    emb = np.array([1.0, 0.0], dtype=np.float32)
    assert store.specificity(emb) == 0.0


def test_specificity_partial_neighbors(store: MemoryStore) -> None:
    # 2 similar, 2 orthogonal.
    store.save(_summary("a"), np.array([1.0, 0.0], dtype=np.float32))
    store.save(_summary("b"), np.array([0.95, 0.31], dtype=np.float32))
    store.save(_summary("c"), np.array([0.0, 1.0], dtype=np.float32))
    store.save(_summary("d"), np.array([-1.0, 0.0], dtype=np.float32))
    emb = np.array([1.0, 0.0], dtype=np.float32)
    # 2 out of 4 are above 0.5 threshold.
    assert store.specificity(emb) == 0.5


def test_specificity_ignores_inactive(store: MemoryStore) -> None:
    id1 = store.save(_summary("a"), np.array([1.0, 0.0], dtype=np.float32))
    store.save(_summary("b"), np.array([0.0, 1.0], dtype=np.float32))
    store.deactivate([id1])
    emb = np.array([1.0, 0.0], dtype=np.float32)
    # Only the orthogonal chunk is active.
    assert store.specificity(emb) == 1.0


def test_activate_restores_to_search(store: MemoryStore) -> None:
    id1 = store.save(
        _summary("cats", abstract="cats"), np.array([1.0, 0.0], dtype=np.float32)
    )
    store.deactivate([id1])
    assert store.count() == 0

    store.activate([id1])
    assert store.count() == 1


# --- transaction ---


def test_transaction_commits_on_success(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    with store.transaction():
        store.save(_summary("a"), emb)
        store.save(_summary("b"), emb)
    assert store.count() == 2


def test_transaction_rolls_back_on_error(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    store.save(_summary("existing"), emb)
    try:
        with store.transaction():
            store.deactivate([1])
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    # Deactivation should have been rolled back.
    assert store.count() == 1


def test_transaction_nested(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    with store.transaction():
        store.save(_summary("a"), emb)
        with store.transaction():
            store.save(_summary("b"), emb)
        # Inner transaction doesn't commit yet.
    # Outer transaction commits both.
    assert store.count() == 2


def test_transaction_nested_inner_error_rolls_back_all(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    try:
        with store.transaction():
            store.save(_summary("a"), emb)
            with store.transaction():
                store.save(_summary("b"), emb)
                raise RuntimeError("inner failure")
    except RuntimeError:
        pass
    # Both saves should be rolled back.
    assert store.count() == 0
