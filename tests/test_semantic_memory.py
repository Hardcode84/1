"""Tests for mindloop.semantic_memory."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore
from mindloop.merge_llm import MergeResult
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import ChunkSummary

_EMB_A = np.array([1.0, 0.0], dtype=np.float32)
_EMB_B = np.array([0.0, 1.0], dtype=np.float32)


def _chunk(text: str) -> Chunk:
    return Chunk(turns=[Turn(timestamp=datetime.min, role="You", text=text)])


def _summary(text: str, abstract: str = "abs", summary: str = "sum") -> ChunkSummary:
    return ChunkSummary(chunk=_chunk(text), abstract=abstract, summary=summary)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


def _mock_get_embeddings(
    emb: "np.ndarray[Any, Any]",
) -> Callable[..., "np.ndarray[Any, Any]"]:
    """Return a mock get_embeddings that always returns the given vector."""

    def _get_embeddings(texts: list[str], **_kw: object) -> np.ndarray:
        return np.tile(emb, (len(texts), 1))

    return _get_embeddings


@contextmanager
def _patch_embeddings(emb: "np.ndarray[Any, Any]") -> Iterator[None]:
    """Patch get_embeddings in both semantic_memory and memory modules."""
    mock = _mock_get_embeddings(emb)
    with (
        patch("mindloop.semantic_memory.get_embeddings", side_effect=mock),
        patch("mindloop.memory.get_embeddings", side_effect=mock),
    ):
        yield


def test_save_no_existing_chunks(store: MemoryStore) -> None:
    with _patch_embeddings(_EMB_A):
        row_id = save_memory(store, "new fact", "abs", "sum")
    assert row_id > 0
    assert store.count() == 1


def test_save_no_merge_when_dissimilar(store: MemoryStore) -> None:
    store.save(_summary("old fact"), _EMB_B)

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=False),
    ):
        save_memory(store, "new fact", "abs", "sum")
    assert store.count() == 2


def test_save_merges_similar_chunk(store: MemoryStore) -> None:
    store.save(_summary("old fact", abstract="old_abs"), _EMB_A)

    mr = MergeResult(text="merged fact", abstract="merged_abs", summary="merged_sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        row_id = save_memory(store, "new fact", "abs", "sum", min_specificity=0.0)

    # Old chunk deactivated, incoming leaf preserved, merged chunk active.
    assert store.count() == 1
    assert store.count(active_only=False) == 3
    assert row_id > 0


def test_save_cascading_merges(store: MemoryStore) -> None:
    store.save(_summary("fact A"), _EMB_A)
    store.save(_summary("fact B"), _EMB_A)

    merge_count = 0

    def _counting_should_merge(*_a: object, **_kw: object) -> bool:
        nonlocal merge_count
        merge_count += 1
        return True

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch(
            "mindloop.semantic_memory.should_merge",
            side_effect=_counting_should_merge,
        ),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(store, "new fact", "abs", "sum", min_specificity=0.0)

    # Both old chunks deactivated, incoming leaf + intermediate + final.
    assert store.count() == 1
    assert store.count(active_only=False) == 5
    assert merge_count == 2


def test_save_stops_at_max_rounds(store: MemoryStore) -> None:
    store.save(_summary("fact"), _EMB_A)

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    merge_count = 0

    def _counting_merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        return mr

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_counting_merge),
    ):
        save_memory(store, "new", "abs", "sum", max_rounds=3)

    assert merge_count <= 3


def test_save_aborts_merge_when_too_generic(store: MemoryStore) -> None:
    # Store many similar chunks so post-merge specificity is low.
    for _ in range(10):
        store.save(_summary("similar"), _EMB_A)

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    merge_count = 0

    def _counting_merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        return mr

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_counting_merge),
    ):
        save_memory(store, "new", "abs", "sum", min_specificity=0.95)

    # First merge attempt triggers specificity check — too low, no merge.
    assert merge_count == 1
    # All original chunks still active + incoming leaf activated.
    assert store.count() == 11
    assert store.count(active_only=False) == 11


def test_save_records_sources_on_merge(store: MemoryStore) -> None:
    old_id = store.save(_summary("old fact"), _EMB_A)

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        new_id = save_memory(store, "new", "abs", "sum", min_specificity=0.0)

    # source_a=incoming leaf, source_b=absorbed chunk.
    row = store.conn.execute(
        "SELECT source_a, source_b FROM chunks WHERE id = ?", (new_id,)
    ).fetchone()
    incoming_id = row[0]
    assert incoming_id is not None
    assert row[1] == old_id

    # Incoming leaf is preserved as disabled original.
    leaf = store.conn.execute(
        "SELECT source_a, source_b, active FROM chunks WHERE id = ?",
        (incoming_id,),
    ).fetchone()
    assert leaf[0] is None
    assert leaf[1] is None
    assert leaf[2] == 0


def test_save_records_tree_on_cascade(store: MemoryStore) -> None:
    id_a = store.save(_summary("fact A"), _EMB_A)
    id_b = store.save(_summary("fact B"), _EMB_A)

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        final_id = save_memory(store, "new", "abs", "sum", min_specificity=0.0)

    # Final node points to first merge + one absorbed chunk.
    final = store.conn.execute(
        "SELECT source_a, source_b FROM chunks WHERE id = ?", (final_id,)
    ).fetchone()
    first_merge_id = final[0]
    absorbed_second = final[1]
    assert first_merge_id is not None
    assert absorbed_second in (id_a, id_b)

    # First merge node: source_a=incoming leaf, source_b=other absorbed chunk.
    merge1 = store.conn.execute(
        "SELECT source_a, source_b, active FROM chunks WHERE id = ?",
        (first_merge_id,),
    ).fetchone()
    incoming_id = merge1[0]
    assert incoming_id is not None
    assert merge1[1] in (id_a, id_b)
    assert merge1[1] != absorbed_second
    assert merge1[2] == 0  # Intermediate is disabled.

    # Incoming leaf at the bottom of the tree.
    leaf = store.conn.execute(
        "SELECT source_a, source_b, active FROM chunks WHERE id = ?",
        (incoming_id,),
    ).fetchone()
    assert leaf[0] is None
    assert leaf[1] is None
    assert leaf[2] == 0


def test_save_no_sources_without_merge(store: MemoryStore) -> None:
    with _patch_embeddings(_EMB_A):
        new_id = save_memory(store, "new fact", "abs", "sum")

    row = store.conn.execute(
        "SELECT source_a, source_b FROM chunks WHERE id = ?", (new_id,)
    ).fetchone()
    assert row[0] is None
    assert row[1] is None


def test_save_is_atomic_on_error(store: MemoryStore) -> None:
    store.save(_summary("existing"), _EMB_A)

    def _exploding_merge(*_a: object, **_kw: object) -> MergeResult:
        raise RuntimeError("LLM failure")

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_exploding_merge),
    ):
        with pytest.raises(RuntimeError, match="LLM failure"):
            save_memory(store, "new", "abs", "sum")

    # Original chunk should still be active — transaction rolled back.
    assert store.count() == 1


def test_save_deduplicates_exact_text(store: MemoryStore) -> None:
    """Saving identical text twice returns the existing id without creating a duplicate."""
    with _patch_embeddings(_EMB_A):
        first_id = save_memory(store, "exact same text", "abs1", "sum1")
        second_id = save_memory(store, "exact same text", "abs2", "sum2")

    assert first_id == second_id
    assert store.count() == 1


def test_specificity_excludes_absorbed_chunk(store: MemoryStore) -> None:
    """Specificity check does not count the absorbed chunk as a neighbor.

    With 4 chunks (3 similar, 1 dissimilar) and min_specificity=0.3:
    - Old (absorbed counted): 3/4 neighbors → specificity 0.25 → blocked.
    - New (absorbed excluded): 2/3 neighbors → specificity 0.33 → passes.
    """
    _emb_sim = np.array([1.0, 0.0], dtype=np.float32)
    _emb_diff = np.array([0.0, 1.0], dtype=np.float32)

    # 3 similar + 1 dissimilar active chunks.
    store.save(_summary("similar A"), _emb_sim)
    store.save(_summary("similar B"), _emb_sim)
    store.save(_summary("similar C"), _emb_sim)
    store.save(_summary("different"), _emb_diff)

    mr = MergeResult(text="merged similar", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_emb_sim),
        patch("mindloop.semantic_memory.should_merge", return_value=True),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(store, "new similar", "abs", "sum", min_specificity=0.3)

    # Merge should have happened (not blocked by specificity).
    assert store.count() < 5
