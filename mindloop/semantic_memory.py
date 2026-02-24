"""Semantic memory: save with automatic merge loop."""

from collections.abc import Callable
from datetime import datetime

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore, SearchResult, leaf_faithfulness
from mindloop.merge_llm import MergeResult, merge_texts, should_merge
from mindloop.summarizer import ChunkSummary
from mindloop.util import noop

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_ROUNDS = 10
_DEFAULT_MIN_FAITHFULNESS = 0.7
_DEFAULT_NEIGHBOR_MARGIN = 0.05
_DEFAULT_NEIGHBOR_K = 3
_DEFAULT_SIM_HIGH = 0.9
_DEFAULT_SIM_LOW = 0.2
_CORE_DEDUP_THRESHOLD = 0.8


def save_memory(
    store: MemoryStore,
    text: str,
    abstract: str,
    summary: str,
    model: str,
    top_k: int = _DEFAULT_TOP_K,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    min_faithfulness: float = _DEFAULT_MIN_FAITHFULNESS,
    neighbor_margin: float = _DEFAULT_NEIGHBOR_MARGIN,
    neighbor_k: int = _DEFAULT_NEIGHBOR_K,
    prefer: str = "equal",
    log: Callable[[str], None] = noop,
    sim_high: float = _DEFAULT_SIM_HIGH,
    sim_low: float = _DEFAULT_SIM_LOW,
    tier: str | None = None,
) -> int:
    """Save a memory, merging with similar existing memories until fixed point.

    The entire operation runs inside a transaction — all deactivations and
    the final save are atomic.  Returns the row id of the final saved chunk.
    If an active chunk with identical text already exists, returns its id
    without creating a duplicate.
    """
    chunk = Chunk(turns=[Turn(timestamp=datetime.now(), role="memory", text=text)])
    stored_text = chunk.text

    existing = store.find_exact(stored_text)
    if existing is not None:
        log("[memory] Exact duplicate found, skipping.")
        return existing

    # Episodic memories that restate a core memory are redundant — check
    # early before entering the merge loop.
    if tier != "core":
        core_hits = store.search(text, top_k=1, tier="core", mode="cosine")
        if core_hits and core_hits[0].cosine_score >= _CORE_DEDUP_THRESHOLD:
            log(
                f"[memory] Restates core #{core_hits[0].id}"
                f" (sim={core_hits[0].cosine_score:.3f}), skipping."
            )
            return core_hits[0].id

    # Determine search tier: core merges with core, episodic with episodic.
    search_tier = tier or "episodic"

    with store.transaction():
        cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
        last_id = store.save(cs, tier=tier)
        store.deactivate([last_id])

        incoming_leaves = [stored_text]

        for round_idx in range(max_rounds):
            # Union hybrid + cosine-only results to catch paraphrases.
            hybrid = store.search(text, top_k=top_k, tier=search_tier)
            cosine = store.search(text, top_k=top_k, mode="cosine", tier=search_tier)
            seen: set[int] = set()
            results: list[SearchResult] = []
            for r in hybrid + cosine:
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)
            log(f"[memory] Round {round_idx + 1}: {len(results)} candidates.")

            merged = False
            for result in results:
                existing_text = result.chunk_summary.chunk.text
                sim = result.cosine_score

                # Auto-merge / auto-skip by cosine similarity thresholds.
                if sim >= sim_high:
                    log(
                        f"[memory]   #{result.id} sim={sim:.3f}"
                        f" >= high={sim_high} → auto-merge."
                    )
                    do_merge = True
                elif sim < sim_low:
                    log(
                        f"[memory]   #{result.id} sim={sim:.3f}"
                        f" < low={sim_low} → skip."
                    )
                    continue
                else:
                    log(
                        f"[memory]   #{result.id} sim={sim:.3f}"
                        f" (low={sim_low}..high={sim_high}) → asking LLM..."
                    )
                    do_merge = should_merge(text, existing_text, model=model)
                    log(f"[memory]   LLM says {'merge' if do_merge else 'no merge'}.")

                if not do_merge:
                    # LLM decided chunks are distinct but related.
                    store.add_edge(last_id, result.id, "related_to", score=sim)
                    continue

                mr: MergeResult = merge_texts(
                    text, existing_text, prefer=prefer, model=model
                )

                # Check 1: leaf faithfulness (before deactivate — no store interaction).
                existing_leaves = store.leaf_texts(result.id)
                all_leaves = incoming_leaves + existing_leaves
                passed, min_sim = leaf_faithfulness(
                    mr.text, all_leaves, threshold=min_faithfulness
                )
                if not passed:
                    log(
                        f"[memory]   Leaf faithfulness {min_sim:.3f}"
                        f" (of {len(all_leaves)} leaves)"
                        f" < {min_faithfulness} → aborting merge."
                    )
                    store.add_edge(last_id, result.id, "related_to", score=sim)
                    break

                # Check 2: relative density (deactivate absorbed chunk first).
                store.deactivate([result.id])
                candidate_ns = store.neighbor_score(existing_text, top_k=neighbor_k)
                merged_ns = store.neighbor_score(mr.text, top_k=neighbor_k)
                density_increase = merged_ns - candidate_ns
                if density_increase > neighbor_margin:
                    log(
                        f"[memory]   Density increase {density_increase:.3f}"
                        f" ({candidate_ns:.3f} → {merged_ns:.3f})"
                        f" > margin {neighbor_margin} → aborting merge."
                    )
                    store.activate([result.id])
                    store.add_edge(last_id, result.id, "related_to", score=sim)
                    break

                # Save merge node (disabled) to preserve the full tree.
                chunk = Chunk(
                    turns=[Turn(timestamp=datetime.now(), role="memory", text=mr.text)]
                )
                cs = ChunkSummary(chunk=chunk, abstract=mr.abstract, summary=mr.summary)
                old_last_id = last_id
                last_id = store.save(
                    cs, source_a=old_last_id, source_b=result.id, tier=tier
                )
                store.inherit_edges(last_id, [old_last_id, result.id])
                store.deactivate([last_id])

                text = mr.text
                abstract = mr.abstract
                summary = mr.summary
                incoming_leaves = all_leaves
                merged = True
                log(f"[memory]   Merged → #{last_id}: {mr.abstract}")
                break  # Restart search with merged text.

            if not merged:
                log("[memory] Fixed point reached.")
                break

        # Activate whichever node ended up final (leaf or last merge).
        store.activate([last_id])
        return last_id
