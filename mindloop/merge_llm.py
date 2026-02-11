"""LLM-based chunk merge decisions."""

from mindloop.client import Message, chat

_SYSTEM_PROMPT = """\
You decide whether two consecutive text chunks belong to the same topic \
and should be merged into one.

Answer "yes" if:
- They discuss the same subject or continue the same line of thought.
- One provides context, detail, or follow-up for the other.
- Splitting them would break a coherent narrative.

Answer "no" if:
- There is a clear topic shift between them.
- They are self-contained and cover distinct subjects.
- Merging would make the result unfocused.

Reply with a single word: yes or no."""

_USER_TEMPLATE = """\
--- Chunk A ---
{chunk_a}

--- Chunk B ---
{chunk_b}"""


def should_merge(
    text_a: str,
    text_b: str,
    cosine_sim: float,
    model: str = "openrouter/free",
    high: float = 0.9,
    low: float = 0.2,
) -> bool:
    """Decide whether two chunks should be merged.

    Uses cosine similarity for clear-cut cases, falls back to LLM
    for the borderline zone between *low* and *high*.
    """
    if cosine_sim >= high:
        return True
    if cosine_sim < low:
        return False

    messages: list[Message] = [
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(chunk_a=text_a, chunk_b=text_b),
        },
    ]
    resp = chat(
        messages,
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        stream=False,
        temperature=0.0,
    )
    answer = (resp.get("content") or "").strip().lower()
    return answer.startswith("yes")
