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
    """Patch get_embeddings in the memory module (only consumer now)."""
    mock = _mock_get_embeddings(emb)
    with patch("mindloop.memory.get_embeddings", side_effect=mock):
        yield


def test_save_no_existing_chunks(store: MemoryStore) -> None:
    with _patch_embeddings(_EMB_A):
        row_id = save_memory(store, "new fact", "abs", "sum")
    assert row_id > 0
    assert store.count() == 1


def test_save_no_merge_when_dissimilar(store: MemoryStore) -> None:
    store.save(_summary("old fact"))

    # Both thresholds above max cosine (1.0) ensures everything is auto-skipped.
    with _patch_embeddings(_EMB_A):
        save_memory(store, "new fact", "abs", "sum", sim_high=2.0, sim_low=2.0)
    assert store.count() == 2


def test_save_merges_similar_chunk(store: MemoryStore) -> None:
    store.save(_summary("old fact", abstract="old_abs"))

    mr = MergeResult(text="merged fact", abstract="merged_abs", summary="merged_sum")
    # Uniform embeddings → cosine_sim=1.0 → auto-merge (above sim_high).
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        row_id = save_memory(store, "new fact", "abs", "sum", max_neighbor_score=1.0)

    # Old chunk deactivated, incoming leaf preserved, merged chunk active.
    assert store.count() == 1
    assert store.count(active_only=False) == 3
    assert row_id > 0


def test_save_cascading_merges(store: MemoryStore) -> None:
    store.save(_summary("fact A"))
    store.save(_summary("fact B"))

    merge_count = 0

    def _counting_merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        return MergeResult(text="merged", abstract="abs", summary="sum")

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_counting_merge),
    ):
        save_memory(store, "new fact", "abs", "sum", max_neighbor_score=1.0)

    # Both old chunks deactivated, incoming leaf + intermediate + final.
    assert store.count() == 1
    assert store.count(active_only=False) == 5
    assert merge_count == 2


def test_save_stops_at_max_rounds(store: MemoryStore) -> None:
    store.save(_summary("fact"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    merge_count = 0

    def _counting_merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        return mr

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_counting_merge),
    ):
        save_memory(store, "new", "abs", "sum", max_rounds=3)

    assert merge_count <= 3


def test_save_aborts_merge_when_too_generic(store: MemoryStore) -> None:
    """Neighbor score exceeds threshold → merge aborted."""
    # Multiple chunks so neighbors remain after absorbing one.
    for _ in range(3):
        store.save(_summary("similar"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    merge_count = 0

    def _counting_merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        return mr

    # Uniform embeddings → faithfulness passes (sim=1.0).
    # max_neighbor_score=0.0 → any positive neighbor score rejects.
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_counting_merge),
    ):
        save_memory(store, "new", "abs", "sum", max_neighbor_score=0.0)

    # Merge attempted, but neighbor score too high → aborted.
    assert merge_count == 1
    # All original chunks still active + incoming leaf activated.
    assert store.count() == 4


def test_save_records_sources_on_merge(store: MemoryStore) -> None:
    old_id = store.save(_summary("old fact"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        new_id = save_memory(store, "new", "abs", "sum", max_neighbor_score=1.0)

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
    id_a = store.save(_summary("fact A"))
    id_b = store.save(_summary("fact B"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        final_id = save_memory(store, "new", "abs", "sum", max_neighbor_score=1.0)

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
    store.save(_summary("existing"))

    def _exploding_merge(*_a: object, **_kw: object) -> MergeResult:
        raise RuntimeError("LLM failure")

    with (
        _patch_embeddings(_EMB_A),
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


def test_neighbor_score_excludes_absorbed_chunk(store: MemoryStore) -> None:
    """Absorbed chunk is deactivated before neighbor score check.

    The absorbed chunk should not inflate the neighbor score that would
    otherwise block the merge.
    """
    # Two similar chunks — one will be absorbed during merge.
    store.save(_summary("similar A"))
    store.save(_summary("similar B"))

    mr = MergeResult(text="merged similar", abstract="abs", summary="sum")
    # Uniform embeddings → faithfulness passes.
    # After absorbing one chunk, only one neighbor remains.
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(store, "new similar", "abs", "sum", max_neighbor_score=1.0)

    # Merge should have happened.
    assert store.count() < 3


def test_save_borderline_calls_should_merge(store: MemoryStore) -> None:
    """Cosine sim between low and high triggers LLM-based should_merge."""
    store.save(_summary("old fact"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=True) as mock_sm,
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        # sim=1.0, set sim_high above 1 to force borderline path.
        save_memory(
            store,
            "new",
            "abs",
            "sum",
            max_neighbor_score=1.0,
            sim_high=2.0,
            sim_low=0.5,
        )

    mock_sm.assert_called_once()


def test_save_auto_skip_below_sim_low(store: MemoryStore) -> None:
    """Cosine sim below sim_low skips without calling should_merge."""
    store.save(_summary("old fact"))

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge") as mock_sm,
    ):
        # Both thresholds above max cosine (1.0) → auto-skip.
        save_memory(store, "new", "abs", "sum", sim_high=2.0, sim_low=2.0)

    mock_sm.assert_not_called()
    assert store.count() == 2


def test_save_logs_progress(store: MemoryStore) -> None:
    """Log callback receives round and merge messages."""
    store.save(_summary("old fact"))
    logged: list[str] = []

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(
            store, "new", "abs", "sum", max_neighbor_score=1.0, log=logged.append
        )

    round_msgs = [m for m in logged if "Round" in m]
    merge_msgs = [m for m in logged if "Merged" in m]
    assert len(round_msgs) >= 1
    assert len(merge_msgs) >= 1


def test_save_aborts_on_faithfulness_failure(store: MemoryStore) -> None:
    """Faithfulness failure aborts merge without deactivating any chunks."""
    store.save(_summary("old fact"))

    mr = MergeResult(text="drifted", abstract="abs", summary="sum")

    # Merged text is orthogonal to source_b (old fact).
    _merged_emb = np.array([1.0, 0.0], dtype=np.float32)
    _old_emb = np.array([0.0, 1.0], dtype=np.float32)

    call_count = 0

    def _faith_emb(texts: list[str], **_kw: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        # faithfulness() passes 3 texts: [merged, new, old].
        if len(texts) == 3:
            return np.stack([_merged_emb, _merged_emb, _old_emb])
        # search / other calls: uniform embeddings.
        return np.tile(_EMB_A, (len(texts), 1))

    with (
        patch("mindloop.memory.get_embeddings", side_effect=_faith_emb),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(store, "new fact", "abs", "sum")

    # Merge aborted — old chunk still active, incoming leaf also active.
    assert store.count() == 2
