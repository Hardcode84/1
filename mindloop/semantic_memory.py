"""Semantic memory: save with automatic merge loop."""

from collections.abc import Callable
from datetime import datetime

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore
from mindloop.merge_llm import MergeResult, merge_texts, should_merge
from mindloop.summarizer import ChunkSummary

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_ROUNDS = 10
_DEFAULT_MIN_SPECIFICITY = 0.3


def save_memory(
    store: MemoryStore,
    text: str,
    abstract: str,
    summary: str,
    model: str = "openrouter/free",
    top_k: int = _DEFAULT_TOP_K,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    min_specificity: float = _DEFAULT_MIN_SPECIFICITY,
    prefer: str = "equal",
    on_merge: Callable[[MergeResult], None] | None = None,
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
        return existing

    with store.transaction():
        cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
        last_id = store.save(cs)
        store.deactivate([last_id])

        for _ in range(max_rounds):
            results = store.search(text, top_k=top_k)

            merged = False
            for result in results:
                existing_text = result.chunk_summary.chunk.text
                if not should_merge(text, existing_text, result.score, model=model):
                    continue

                mr: MergeResult = merge_texts(
                    text, existing_text, prefer=prefer, model=model
                )

                # Deactivate before specificity check so the absorbed chunk
                # doesn't inflate the neighbor count against the merged result.
                store.deactivate([result.id])

                # Check if the merge would make the chunk too generic.
                if store.specificity(mr.text) < min_specificity:
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
                if on_merge is not None:
                    on_merge(mr)
                break  # Restart search with merged text.

            if not merged:
                break  # Fixed point.

        # Activate whichever node ended up final (leaf or last merge).
        store.activate([last_id])
        return last_id
