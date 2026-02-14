"""Semantic memory: save with automatic merge loop."""

from collections.abc import Callable
from datetime import datetime

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore, faithfulness
from mindloop.merge_llm import MergeResult, merge_texts, should_merge
from mindloop.summarizer import ChunkSummary
from mindloop.util import noop

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_ROUNDS = 10
_DEFAULT_MIN_FAITHFULNESS = 0.7
_DEFAULT_MAX_NEIGHBOR_SCORE = 0.6
_DEFAULT_NEIGHBOR_K = 3
_DEFAULT_SIM_HIGH = 0.9
_DEFAULT_SIM_LOW = 0.2


def save_memory(
    store: MemoryStore,
    text: str,
    abstract: str,
    summary: str,
    model: str = "openrouter/free",
    top_k: int = _DEFAULT_TOP_K,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    min_faithfulness: float = _DEFAULT_MIN_FAITHFULNESS,
    max_neighbor_score: float = _DEFAULT_MAX_NEIGHBOR_SCORE,
    neighbor_k: int = _DEFAULT_NEIGHBOR_K,
    prefer: str = "equal",
    log: Callable[[str], None] = noop,
    sim_high: float = _DEFAULT_SIM_HIGH,
    sim_low: float = _DEFAULT_SIM_LOW,
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

    with store.transaction():
        cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
        last_id = store.save(cs)
        store.deactivate([last_id])

        for round_idx in range(max_rounds):
            results = store.search(text, top_k=top_k)
            log(f"[memory] Round {round_idx + 1}: {len(results)} candidates.")

            merged = False
            for result in results:
                existing_text = result.chunk_summary.chunk.text
                sim = result.score

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
                    continue

                mr: MergeResult = merge_texts(
                    text, existing_text, prefer=prefer, model=model
                )

                # Check 1: faithfulness (before deactivate — no store interaction).
                passed, sim_a, sim_b = faithfulness(
                    mr.text, text, existing_text, threshold=min_faithfulness
                )
                if not passed:
                    log(
                        f"[memory]   Faithfulness {sim_a:.3f}/{sim_b:.3f}"
                        f" < {min_faithfulness} → aborting merge."
                    )
                    break

                # Check 2: neighbor score (deactivate absorbed chunk first).
                store.deactivate([result.id])
                ns = store.neighbor_score(mr.text, top_k=neighbor_k)
                if ns > max_neighbor_score:
                    log(
                        f"[memory]   Neighbor score {ns:.3f}"
                        f" > {max_neighbor_score} → aborting merge."
                    )
                    store.activate([result.id])
                    break

                # Save merge node (disabled) to preserve the full tree.
                chunk = Chunk(
                    turns=[Turn(timestamp=datetime.now(), role="memory", text=mr.text)]
                )
                cs = ChunkSummary(chunk=chunk, abstract=mr.abstract, summary=mr.summary)
                last_id = store.save(cs, source_a=last_id, source_b=result.id)
                store.deactivate([last_id])

                text = mr.text
                abstract = mr.abstract
                summary = mr.summary
                merged = True
                log(f"[memory]   Merged → #{last_id}: {mr.abstract}")
                break  # Restart search with merged text.

            if not merged:
                log("[memory] Fixed point reached.")
                break

        # Activate whichever node ended up final (leaf or last merge).
        store.activate([last_id])
        return last_id
