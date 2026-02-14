"""Tests for mindloop.memory."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore, faithfulness
from mindloop.summarizer import ChunkSummary


def _chunk(text: str) -> Chunk:
    return Chunk(turns=[Turn(timestamp=datetime.min, role="You", text=text)])


def _summary(text: str, abstract: str = "abs", summary: str = "sum") -> ChunkSummary:
    return ChunkSummary(chunk=_chunk(text), abstract=abstract, summary=summary)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


def _emb_for(*fixed: np.ndarray) -> Callable[..., np.ndarray]:
    """Return a mock get_embeddings returning *fixed* vectors in order.

    Each text is mapped to the next vector in *fixed*, cycling if needed.
    """

    def _get_embeddings(texts: list[str], **_kw: object) -> np.ndarray:
        return np.stack([fixed[i % len(fixed)] for i in range(len(texts))])

    return _get_embeddings


# Convenience embeddings.
_E1 = np.array([1.0, 0.0], dtype=np.float32)
_E2 = np.array([0.0, 1.0], dtype=np.float32)
_E3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_save_and_count(store: MemoryStore) -> None:
    row_id = store.save(_summary("hello"))
    assert row_id > 0
    assert store.count() == 1


def test_save_with_sources(store: MemoryStore) -> None:
    id_a = store.save(_summary("a"))
    id_b = store.save(_summary("b"))
    id_merged = store.save(_summary("merged"), source_a=id_a, source_b=id_b)

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        results = store.search("merged", top_k=1)
    # The top result should be the merged chunk with source info.
    assert results[0].id == id_merged
    assert results[0].source_a == id_a
    assert results[0].source_b == id_b


def test_save_without_sources_returns_none(store: MemoryStore) -> None:
    store.save(_summary("no sources"))

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        results = store.search("no sources", top_k=1)
    assert results[0].source_a is None
    assert results[0].source_b is None


def test_save_many(store: MemoryStore) -> None:
    summaries = [_summary("a"), _summary("b"), _summary("c")]
    ids = store.save_many(summaries)
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
    # Store three chunks with distinct texts.
    store.save(_summary("about cats", abstract="cats"))
    store.save(_summary("about dogs", abstract="dogs"))
    store.save(_summary("about fish", abstract="fish"))

    # Map stored texts to distinct embeddings (Chunk.text prepends "You: ").
    emb_map = {
        "You: about cats": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "You: about dogs": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "You: about fish": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    # Query embedding close to "cats".
    query_emb = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    def _get_emb(texts: list[str], **_kw: object) -> np.ndarray:
        return np.stack([emb_map.get(t, query_emb) for t in texts])

    with patch("mindloop.memory.get_embeddings", side_effect=_get_emb):
        results = store.search("cats", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_summary.abstract == "cats"
    assert results[0].score > results[1].score


def test_search_top_k_limits_results(store: MemoryStore) -> None:
    for i in range(10):
        store.save(_summary(f"chunk {i}"))

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        results = store.search("query", top_k=3)

    assert len(results) == 3


def test_deactivate_excludes_from_search(store: MemoryStore) -> None:
    id1 = store.save(_summary("cats", abstract="cats"))
    store.save(_summary("dogs", abstract="dogs"))

    store.deactivate([id1])
    assert store.count() == 1
    assert store.count(active_only=False) == 2

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        results = store.search("cats")
    # "cats" is deactivated, only "dogs" should appear.
    assert len(results) == 1
    assert results[0].chunk_summary.abstract == "dogs"


def test_faithfulness_passes_similar() -> None:
    """Identical embeddings → (True, 1.0, 1.0)."""
    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        passed, sim_a, sim_b = faithfulness("merged", "source a", "source b")
    assert passed
    assert sim_a == pytest.approx(1.0)
    assert sim_b == pytest.approx(1.0)


def test_faithfulness_fails_drift() -> None:
    """Merged text diverges from one source → fails."""
    _merged = np.array([1.0, 0.0], dtype=np.float32)
    _src_a = np.array([0.9, 0.44], dtype=np.float32)  # Moderate similarity.
    _src_b = np.array([0.0, 1.0], dtype=np.float32)  # Orthogonal.

    def _get_emb(texts: list[str], **_kw: object) -> np.ndarray:
        return np.stack([_merged, _src_a, _src_b])

    with patch("mindloop.memory.get_embeddings", side_effect=_get_emb):
        passed, sim_a, sim_b = faithfulness("m", "a", "b")
    assert not passed
    assert sim_a > sim_b


def test_faithfulness_custom_threshold() -> None:
    """Boundary: exactly at threshold passes."""
    # cos(merged, src) = 0.6 for both — passes at 0.6, fails at 0.61.
    _merged = np.array([1.0, 0.0], dtype=np.float32)
    _src = np.array([0.6, 0.8], dtype=np.float32)

    def _get_emb(texts: list[str], **_kw: object) -> np.ndarray:
        return np.stack([_merged, _src, _src])

    with patch("mindloop.memory.get_embeddings", side_effect=_get_emb):
        passed_low, _, _ = faithfulness("m", "a", "b", threshold=0.6)
        passed_high, _, _ = faithfulness("m", "a", "b", threshold=0.61)
    assert passed_low
    assert not passed_high


def test_neighbor_score_empty(store: MemoryStore) -> None:
    """No active chunks → 0.0."""
    with patch("mindloop.memory.get_embeddings", side_effect=_emb_for(_E1)):
        assert store.neighbor_score("anything") == 0.0


def test_neighbor_score_returns_mean(store: MemoryStore) -> None:
    """Mean of top-k search scores."""
    store.save(_summary("a"))
    store.save(_summary("b"))

    with patch("mindloop.memory.get_embeddings", side_effect=_emb_for(_E1)):
        score = store.neighbor_score("query", top_k=2)
    # Uniform embeddings → all RRF scores identical and positive.
    assert score > 0.0


def test_neighbor_score_ignores_inactive(store: MemoryStore) -> None:
    """Deactivated chunks don't contribute."""
    id1 = store.save(_summary("a"))
    store.save(_summary("b"))
    store.deactivate([id1])

    with patch("mindloop.memory.get_embeddings", side_effect=_emb_for(_E1)):
        score = store.neighbor_score("query", top_k=5)
    # Only one active chunk — result is its single score.
    assert score > 0.0


def test_activate_restores_to_search(store: MemoryStore) -> None:
    id1 = store.save(_summary("cats", abstract="cats"))
    store.deactivate([id1])
    assert store.count() == 0

    store.activate([id1])
    assert store.count() == 1


# --- transaction ---


def test_transaction_commits_on_success(store: MemoryStore) -> None:
    with store.transaction():
        store.save(_summary("a"))
        store.save(_summary("b"))
    assert store.count() == 2


def test_transaction_rolls_back_on_error(store: MemoryStore) -> None:
    store.save(_summary("existing"))
    try:
        with store.transaction():
            store.deactivate([1])
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    # Deactivation should have been rolled back.
    assert store.count() == 1


def test_transaction_nested(store: MemoryStore) -> None:
    with store.transaction():
        store.save(_summary("a"))
        with store.transaction():
            store.save(_summary("b"))
        # Inner transaction doesn't commit yet.
    # Outer transaction commits both.
    assert store.count() == 2


def test_transaction_nested_inner_error_rolls_back_all(store: MemoryStore) -> None:
    try:
        with store.transaction():
            store.save(_summary("a"))
            with store.transaction():
                store.save(_summary("b"))
                raise RuntimeError("inner failure")
    except RuntimeError:
        pass
    # Both saves should be rolled back.
    assert store.count() == 0


# --- lineage ---


def test_lineage_leaf(store: MemoryStore) -> None:
    leaf_id = store.save(_summary("leaf", abstract="leaf_abs"))
    node = store.lineage(leaf_id)
    assert node is not None
    assert node.id == leaf_id
    assert node.abstract == "leaf_abs"
    assert node.is_leaf
    assert node.source_a is None
    assert node.source_b is None


def test_lineage_single_merge(store: MemoryStore) -> None:
    id_a = store.save(_summary("a", abstract="abs_a"))
    id_b = store.save(_summary("b", abstract="abs_b"))
    merged_id = store.save(
        _summary("merged", abstract="abs_m"), source_a=id_a, source_b=id_b
    )

    node = store.lineage(merged_id)
    assert node is not None
    assert node.id == merged_id
    assert not node.is_leaf
    assert node.source_a is not None and node.source_a.id == id_a
    assert node.source_b is not None and node.source_b.id == id_b
    assert node.source_a.is_leaf
    assert node.source_b.is_leaf


def test_lineage_deep_tree(store: MemoryStore) -> None:
    id_a = store.save(_summary("a"))
    id_b = store.save(_summary("b"))
    id_ab = store.save(_summary("ab"), source_a=id_a, source_b=id_b)
    id_c = store.save(_summary("c"))
    id_abc = store.save(_summary("abc"), source_a=id_ab, source_b=id_c)

    root = store.lineage(id_abc)
    assert root is not None
    assert root.id == id_abc
    # Left subtree is itself a merge.
    assert root.source_a is not None and root.source_a.id == id_ab
    assert root.source_a.source_a is not None and root.source_a.source_a.id == id_a
    assert root.source_a.source_b is not None and root.source_a.source_b.id == id_b
    # Right is a leaf.
    assert root.source_b is not None and root.source_b.id == id_c
    assert root.source_b.is_leaf


def test_lineage_nonexistent(store: MemoryStore) -> None:
    assert store.lineage(999) is None


def test_lineage_tracks_active_status(store: MemoryStore) -> None:
    id_a = store.save(_summary("a"))
    store.deactivate([id_a])
    id_b = store.save(_summary("b"))
    merged_id = store.save(_summary("m"), source_a=id_a, source_b=id_b)

    node = store.lineage(merged_id)
    assert node is not None
    assert node.active
    assert node.source_a is not None and not node.source_a.active
    assert node.source_b is not None and node.source_b.active


# --- hybrid search (BM25 + embeddings) ---


def test_search_bm25_boosts_keyword_match(store: MemoryStore) -> None:
    """BM25 should boost a chunk that matches the query keyword even when
    all embeddings are identical."""
    store.save(
        _summary("the gfx942 architecture requires special flags", abstract="gfx")
    )
    store.save(_summary("general information about compilers", abstract="compilers"))
    store.save(_summary("how to set up the environment", abstract="env"))

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        results = store.search("gfx942")

    # With identical embeddings, BM25 keyword match should push gfx to the top.
    assert results[0].chunk_summary.abstract == "gfx"


def test_search_hybrid_combines_both_signals(store: MemoryStore) -> None:
    """A chunk that scores well on both embedding and BM25 should rank above
    chunks that only score well on one."""
    # Chunk A: good embedding, good keyword match.
    store.save(_summary("cats are wonderful pets", abstract="cats"))
    # Chunk B: good embedding, no keyword match.
    store.save(_summary("dogs are loyal animals", abstract="dogs"))
    # Chunk C: bad embedding, good keyword match.
    store.save(_summary("cats in ancient history", abstract="ancient cats"))

    # Keys use stored text format (Chunk.text prepends "You: ").
    emb_map = {
        "You: cats are wonderful pets": np.array([0.9, 0.1], dtype=np.float32),
        "You: dogs are loyal animals": np.array([0.85, 0.15], dtype=np.float32),
        "You: cats in ancient history": np.array([0.1, 0.9], dtype=np.float32),
    }
    query_emb = np.array([1.0, 0.0], dtype=np.float32)

    def _get_emb(texts: list[str], **_kw: object) -> np.ndarray:
        return np.stack([emb_map.get(t, query_emb) for t in texts])

    with patch("mindloop.memory.get_embeddings", side_effect=_get_emb):
        results = store.search("cats", top_k=3)

    # Chunk A should rank first — strong in both signals.
    assert results[0].chunk_summary.abstract == "cats"


def test_search_original_only_with_bm25(store: MemoryStore) -> None:
    """original_only flag should filter correctly with hybrid search."""
    id_a = store.save(_summary("unique keyword xyzzy", abstract="leaf"))
    store.save(
        _summary("merged result with xyzzy", abstract="merged"),
        source_a=id_a,
        source_b=id_a,
    )

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        results = store.search("xyzzy", original_only=True)

    # Only the leaf should be returned.
    abstracts = [r.chunk_summary.abstract for r in results]
    assert "leaf" in abstracts
    assert "merged" not in abstracts


def test_search_graceful_without_fts_match(store: MemoryStore) -> None:
    """Search still works when no FTS5 matches exist (falls back to embeddings)."""
    store.save(_summary("about something", abstract="something"))

    with patch(
        "mindloop.memory.get_embeddings",
        side_effect=_emb_for(_E1),
    ):
        # Query with a word not in any chunk text.
        results = store.search("nonexistent_word_zzz")

    # Should still return the embedding match.
    assert len(results) == 1
    assert results[0].chunk_summary.abstract == "something"


def test_find_exact_returns_active_match(store: MemoryStore) -> None:
    """find_exact returns the id of an active chunk with identical text."""
    cs = _summary("hello world")
    row_id = store.save(cs)
    assert store.find_exact(cs.chunk.text) == row_id


def test_find_exact_ignores_inactive(store: MemoryStore) -> None:
    """find_exact does not return deactivated chunks."""
    cs = _summary("hello world")
    row_id = store.save(cs)
    store.deactivate([row_id])
    assert store.find_exact(cs.chunk.text) is None


def test_find_exact_no_match(store: MemoryStore) -> None:
    """find_exact returns None when no chunk matches."""
    assert store.find_exact("nonexistent") is None
