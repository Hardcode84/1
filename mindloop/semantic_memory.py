"""Semantic memory: save with automatic merge loop."""

from collections.abc import Callable
from datetime import datetime

from mindloop.chunker import Chunk, Turn
from mindloop.client import get_embeddings
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
    """
    embedding = get_embeddings([text])[0]

    with store.transaction():
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
                new_embedding = get_embeddings([mr.text])[0]

                # Check if the merge would make the chunk too generic.
                if store.specificity(new_embedding) < min_specificity:
                    break

                store.deactivate([result.id])
                text = mr.text
                abstract = mr.abstract
                summary = mr.summary
                embedding = new_embedding
                merged = True
                if on_merge is not None:
                    on_merge(mr)
                break  # Restart search with merged text.

            if not merged:
                break  # Fixed point.

        chunk = Chunk(turns=[Turn(timestamp=datetime.now(), role="memory", text=text)])
        cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
        return store.save(cs, embedding)
