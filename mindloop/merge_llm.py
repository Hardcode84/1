"""LLM-based chunk merge decisions."""

from dataclasses import dataclass

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
    model: str = "openrouter/free",
) -> bool:
    """Ask the LLM whether two chunks belong together and should merge."""
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
        seed=42,
        cache_messages=False,
    )
    answer = (resp.get("content") or "").strip().lower()
    return answer.startswith("yes")


@dataclass
class MergeResult:
    text: str
    abstract: str
    summary: str


_RELEVANCE_HINTS = {
    "a": "Chunk A is the primary source. Use it as the base and fold in unique details from Chunk B.",
    "b": "Chunk B is the primary source. Use it as the base and fold in unique details from Chunk A.",
    "equal": "Both chunks are equally relevant. Interleave their content logically.",
}

_MERGE_SYSTEM_PROMPT = """\
You merge two text chunks into a single coherent piece.

{relevance_hint}

Rules:
- Deduplicate: remove information that appears in both chunks.
- Preserve specific facts, names, numbers, and conclusions from both.
- Do not silently discard content from the secondary chunk — \
fold in its unique details.
- Keep the tone and style consistent with the originals.
- Do not add commentary or meta-text like "merged from two chunks".
- Output only the merged text."""

_MERGE_USER_TEMPLATE = """\
--- Chunk A ---
{chunk_a}

--- Chunk B ---
{chunk_b}

Merge these into a single coherent text."""

_SUMMARIZE_PROMPT = """\
Now summarize the merged text you just produced.
Respond in exactly this format (two lines, keep the prefixes):
ABSTRACT: <one sentence TL;DR>
SUMMARY: <2-4 sentence expanded overview>"""


def merge_texts(
    text_a: str,
    text_b: str,
    prefer: str = "equal",
    model: str = "openrouter/free",
) -> MergeResult:
    """Merge two chunk texts and generate abstract + summary.

    *prefer* controls relevance: "a", "b", or "equal".
    Returns merged text, abstract, and summary.
    """
    hint = _RELEVANCE_HINTS.get(prefer, _RELEVANCE_HINTS["equal"])
    system = _MERGE_SYSTEM_PROMPT.format(relevance_hint=hint)

    # Turn 1: merge.
    messages: list[Message] = [
        {
            "role": "user",
            "content": _MERGE_USER_TEMPLATE.format(chunk_a=text_a, chunk_b=text_b),
        },
    ]
    merge_resp = chat(
        messages,
        model=model,
        system_prompt=system,
        stream=False,
        temperature=0.0,
        seed=42,
        cache_messages=False,
    )
    merged_text = (merge_resp.get("content") or "").strip()

    # Turn 2: summarize in same conversation context.
    messages.append({"role": "assistant", "content": merged_text})
    messages.append({"role": "user", "content": _SUMMARIZE_PROMPT})
    summary_resp = chat(
        messages,
        model=model,
        system_prompt=system,
        stream=False,
        temperature=0.0,
        seed=42,
        cache_messages=False,
    )
    raw = (summary_resp.get("content") or "").strip()

    abstract = ""
    summary = ""
    for line in raw.splitlines():
        if line.upper().startswith("ABSTRACT:"):
            abstract = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    return MergeResult(text=merged_text, abstract=abstract, summary=summary)
