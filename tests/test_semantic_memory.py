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
        row_id = save_memory(store, "new fact", "abs", "sum", model="test-model")
    assert row_id > 0
    assert store.count() == 1


def test_save_no_merge_when_dissimilar(store: MemoryStore) -> None:
    store.save(_summary("old fact"))

    # Both thresholds above max cosine (1.0) ensures everything is auto-skipped.
    with _patch_embeddings(_EMB_A):
        save_memory(
            store,
            "new fact",
            "abs",
            "sum",
            model="test-model",
            sim_high=2.0,
            sim_low=2.0,
        )
    assert store.count() == 2


def test_save_merges_similar_chunk(store: MemoryStore) -> None:
    store.save(_summary("old fact", abstract="old_abs"))

    mr = MergeResult(text="merged fact", abstract="merged_abs", summary="merged_sum")
    # Uniform embeddings → cosine_sim=1.0 → auto-merge (above sim_high).
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        row_id = save_memory(
            store, "new fact", "abs", "sum", model="test-model", neighbor_margin=2.0
        )

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
        save_memory(
            store, "new fact", "abs", "sum", model="test-model", neighbor_margin=2.0
        )

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
        save_memory(store, "new", "abs", "sum", model="test-model", max_rounds=3)

    assert merge_count <= 3


def test_save_aborts_merge_when_density_increases(store: MemoryStore) -> None:
    """Density increase exceeds margin → merge aborted, edge recorded."""
    # Multiple chunks so neighbors remain after absorbing one.
    for _ in range(3):
        store.save(_summary("similar"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    merge_count = 0

    def _counting_merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        return mr

    # Uniform embeddings → candidate_ns == merged_ns == 1.0, increase = 0.
    # Negative margin forces rejection (0.0 > -0.01).
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_counting_merge),
    ):
        save_memory(
            store, "new", "abs", "sum", model="test-model", neighbor_margin=-0.01
        )

    # Merge attempted, but density check failed → aborted.
    assert merge_count == 1
    # All original chunks still active + incoming leaf activated.
    assert store.count() == 4

    # Edge recorded between the incoming leaf and the absorbed candidate.
    edges = store.conn.execute("SELECT edge_type FROM chunk_edges").fetchall()
    assert len(edges) == 1
    assert edges[0][0] == "related_to"


def test_save_records_sources_on_merge(store: MemoryStore) -> None:
    old_id = store.save(_summary("old fact"))

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        new_id = save_memory(
            store, "new", "abs", "sum", model="test-model", neighbor_margin=2.0
        )

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
        final_id = save_memory(
            store, "new", "abs", "sum", model="test-model", neighbor_margin=2.0
        )

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
        new_id = save_memory(store, "new fact", "abs", "sum", model="test-model")

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
            save_memory(store, "new", "abs", "sum", model="test-model")

    # Original chunk should still be active — transaction rolled back.
    assert store.count() == 1


def test_save_deduplicates_exact_text(store: MemoryStore) -> None:
    """Saving identical text twice returns the existing id without creating a duplicate."""
    with _patch_embeddings(_EMB_A):
        first_id = save_memory(
            store, "exact same text", "abs1", "sum1", model="test-model"
        )
        second_id = save_memory(
            store, "exact same text", "abs2", "sum2", model="test-model"
        )

    assert first_id == second_id
    assert store.count() == 1


def test_neighbor_score_excludes_absorbed_chunk(store: MemoryStore) -> None:
    """Absorbed chunk is deactivated before density check.

    Both candidate and merged neighbor scores are computed against the same
    ambient set (excluding the absorbed chunk).
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
        save_memory(
            store,
            "new similar",
            "abs",
            "sum",
            model="test-model",
            neighbor_margin=2.0,
        )

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
            model="test-model",
            neighbor_margin=2.0,
            sim_high=2.0,
            sim_low=0.5,
        )

    mock_sm.assert_called_once()


def test_save_borderline_llm_rejects_records_edge(store: MemoryStore) -> None:
    """LLM says no merge in the borderline zone → edge recorded."""
    store.save(_summary("old fact"))

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge", return_value=False),
    ):
        save_memory(
            store,
            "new",
            "abs",
            "sum",
            model="test-model",
            sim_high=2.0,
            sim_low=0.5,
        )

    # No merge happened — both chunks active.
    assert store.count() == 2

    # Edge recorded between incoming leaf and old chunk.
    edges = store.conn.execute("SELECT edge_type, score FROM chunk_edges").fetchall()
    assert len(edges) == 1
    assert edges[0][0] == "related_to"


def test_save_auto_skip_below_sim_low(store: MemoryStore) -> None:
    """Cosine sim below sim_low skips without calling should_merge."""
    store.save(_summary("old fact"))

    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.should_merge") as mock_sm,
    ):
        # Both thresholds above max cosine (1.0) → auto-skip.
        save_memory(
            store, "new", "abs", "sum", model="test-model", sim_high=2.0, sim_low=2.0
        )

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
            store,
            "new",
            "abs",
            "sum",
            model="test-model",
            neighbor_margin=2.0,
            log=logged.append,
        )

    round_msgs = [m for m in logged if "Round" in m]
    merge_msgs = [m for m in logged if "Merged" in m]
    assert len(round_msgs) >= 1
    assert len(merge_msgs) >= 1


def test_save_aborts_on_faithfulness_failure(store: MemoryStore) -> None:
    """Faithfulness failure aborts merge and records an edge."""
    store.save(_summary("old fact"))

    mr = MergeResult(text="drifted", abstract="abs", summary="sum")

    # Merged text is orthogonal to old fact's leaf text.
    _merged_emb = np.array([1.0, 0.0], dtype=np.float32)
    _old_emb = np.array([0.0, 1.0], dtype=np.float32)

    def _faith_emb(texts: list[str], **_kw: object) -> np.ndarray:
        # leaf_faithfulness passes [merged] + leaf_texts.
        # For a single-leaf merge: [merged, incoming_leaf, old_leaf] = 3 texts.
        if len(texts) == 3:
            return np.stack([_merged_emb, _merged_emb, _old_emb])
        # search / other calls: uniform embeddings.
        return np.tile(_EMB_A, (len(texts), 1))

    with (
        patch("mindloop.memory.get_embeddings", side_effect=_faith_emb),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(store, "new fact", "abs", "sum", model="test-model")

    # Merge aborted — old chunk still active, incoming leaf also active.
    assert store.count() == 2

    # Edge recorded between the incoming leaf and the old chunk.
    edges = store.conn.execute(
        "SELECT source_id, target_id, edge_type FROM chunk_edges"
    ).fetchall()
    assert len(edges) == 1
    assert edges[0][2] == "related_to"


def test_save_cascade_stops_on_original_drift(store: MemoryStore) -> None:
    """Round-1 merge passes leaf check, round-2 drifts from original input.

    Chunks A and B exist. New text merges with A (faithful to original +
    A's leaves), but the round-2 merge with B drifts from the original
    input text. Verify merge stops at round 1.
    """
    store.save(_summary("fact A"))
    store.save(_summary("fact B"))

    merge_count = 0

    def _merge(*_a: object, **_kw: object) -> MergeResult:
        nonlocal merge_count
        merge_count += 1
        if merge_count == 1:
            return MergeResult(text="merged AB", abstract="abs", summary="sum")
        return MergeResult(text="drifted ABC", abstract="abs", summary="sum")

    # Stored texts use "You: " prefix via Chunk.text.
    _orig = np.array([1.0, 0.0], dtype=np.float32)
    _a_emb = np.array([0.9, 0.44], dtype=np.float32)
    _b_emb = np.array([0.0, 1.0], dtype=np.float32)
    _merged_ab = np.array([0.95, 0.22], dtype=np.float32)  # Close to orig + A.
    _drifted = np.array([0.0, 1.0], dtype=np.float32)  # Orthogonal to orig.

    leaf_faith_call = 0

    def _emb(texts: list[str], **_kw: object) -> np.ndarray:
        nonlocal leaf_faith_call
        # leaf_faithfulness: [merged] + leaves.
        # Round 1: 3 texts [merged_ab, incoming_leaf, a_leaf].
        # Round 2: 4 texts [drifted, incoming_leaf, a_leaf, b_leaf].
        if len(texts) == 3 and leaf_faith_call == 0:
            leaf_faith_call += 1
            return np.stack([_merged_ab, _orig, _a_emb])
        if len(texts) == 4:
            leaf_faith_call += 1
            return np.stack([_drifted, _orig, _a_emb, _b_emb])
        # search / other: uniform.
        return np.tile(_EMB_A, (len(texts), 1))

    with (
        patch("mindloop.memory.get_embeddings", side_effect=_emb),
        patch("mindloop.semantic_memory.merge_texts", side_effect=_merge),
    ):
        save_memory(store, "new", "abs", "sum", model="test-model", neighbor_margin=2.0)

    # Round-2 merge aborted → only first merge happened.
    assert merge_count == 2
    # B still active, round-1 merge result active. Original A + incoming deactivated.
    assert store.count() == 2

    # Edge recorded between round-1 merge and B.
    edges = store.conn.execute("SELECT edge_type FROM chunk_edges").fetchall()
    assert len(edges) == 1
    assert edges[0][0] == "related_to"


def test_save_merges_paraphrase_via_cosine_search(store: MemoryStore) -> None:
    """Cosine-only pass surfaces paraphrases that hybrid search misses.

    When an existing chunk uses completely different vocabulary but means
    the same thing, BM25 scores near zero drag down the RRF rank.  The
    cosine-only pass finds the match by embedding similarity alone.
    """
    # Existing chunk with specific vocabulary.
    store.save(_summary("practice restraint in all actions", abstract="restraint"))

    mr = MergeResult(text="merged restraint", abstract="merged_abs", summary="sum")

    # Paraphrase: high cosine similarity, zero keyword overlap.
    # The stored text will be "You: practice restraint in all actions".
    _existing_emb = np.array([0.95, 0.31], dtype=np.float32)
    _new_emb = np.array([1.0, 0.0], dtype=np.float32)

    def _emb(texts: list[str], **_kw: object) -> np.ndarray:
        vecs = []
        for t in texts:
            if "restraint" in t:
                vecs.append(_existing_emb)
            else:
                vecs.append(_new_emb)
        return np.stack(vecs)

    with (
        patch("mindloop.memory.get_embeddings", side_effect=_emb),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        save_memory(
            store,
            "exercise self-control always",
            "abs",
            "sum",
            model="test-model",
            neighbor_margin=2.0,
        )

    # Merge should have happened — only 1 active chunk.
    assert store.count() == 1


def test_save_core_separate_merge_pool(store: MemoryStore) -> None:
    """Core chunk does not merge with episodic chunks."""
    store.save(_summary("episodic fact", abstract="episodic"))

    # Save a core memory — should NOT merge with the episodic chunk
    # because it searches only core tier, finding nothing.
    with _patch_embeddings(_EMB_A):
        core_id = save_memory(
            store,
            "core principle",
            "core_abs",
            "core_sum",
            model="test-model",
            tier="core",
        )

    # Both chunks active: episodic untouched, core saved separately.
    assert store.count() == 2
    tier = store.get_tier(core_id)
    assert tier == "core"


def test_save_episodic_dedup_against_core(store: MemoryStore) -> None:
    """Episodic save restating a core memory returns the core id."""
    # Pre-existing core memory.
    store.save(_summary("practice restraint", abstract="restraint"), tier="core")

    # Episodic save with near-identical meaning — uniform embeddings give
    # cosine=1.0 which exceeds _CORE_DEDUP_THRESHOLD, triggering early bail.
    with _patch_embeddings(_EMB_A):
        result_id = save_memory(
            store,
            "I practiced restraint",
            "abs",
            "sum",
            model="test-model",
        )

    # Should return the core chunk's id without creating anything.
    assert result_id == 1
    assert store.count() == 1


def test_save_inherits_edges_on_merge(store: MemoryStore) -> None:
    """Edges from absorbed chunks are inherited onto the merged chunk."""
    id_a = store.save(_summary("fact A"))
    id_b = store.save(_summary("fact B"))
    # B has a pre-existing edge to A (from a previous rejected merge).
    store.add_edge(id_b, id_a, "related_to", score=0.3)

    mr = MergeResult(text="merged", abstract="abs", summary="sum")
    # Merge absorbs B. Edge B↔A should be inherited onto the merge result.
    with (
        _patch_embeddings(_EMB_A),
        patch("mindloop.semantic_memory.merge_texts", return_value=mr),
    ):
        final_id = save_memory(
            store,
            "new",
            "abs",
            "sum",
            model="test-model",
            neighbor_margin=2.0,
            max_rounds=1,
        )

    # The final merged chunk should have an inherited edge to whichever
    # chunk was not absorbed. With uniform embeddings both A and B are
    # candidates; the first search hit gets absorbed.
    edges = store.edges(final_id)
    inherited = [(eid, etype) for eid, etype, _ in edges if etype == "related_to"]
    assert len(inherited) >= 1
