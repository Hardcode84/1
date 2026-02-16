"""LLM-based chunk merge decisions."""

from dataclasses import dataclass

from mindloop.client import DETERMINISTIC_PARAMS, Message, chat

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
    model: str,
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
        **DETERMINISTIC_PARAMS,
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
You merge two text chunks into a single coherent piece and summarize the result.

{relevance_hint}

Rules:
- Deduplicate: remove information that appears in both chunks.
- Preserve specific facts, names, numbers, and conclusions from both.
- Do not silently discard content from the secondary chunk — \
fold in its unique details.
- Keep the tone, style, and point of view consistent with the originals.
- Do not add commentary or meta-text like "merged from two chunks"."""

_MERGE_USER_TEMPLATE = """\
--- Chunk A ---
{chunk_a}

--- Chunk B ---
{chunk_b}

Merge these into a single coherent text, then summarize it.
Respond in exactly this format (keep the prefixes, no blank lines between them):
MERGED: <the merged text>
ABSTRACT: <one sentence TL;DR>
SUMMARY: <2-4 sentence expanded overview>"""


_MERGE_RETRIES = 3

_RETRY_PROMPT = "Wrong format. Respond with exactly three lines: MERGED: ..., ABSTRACT: ..., SUMMARY: ..."


def _parse_merge_response(raw: str) -> MergeResult | None:
    """Parse a MERGED/ABSTRACT/SUMMARY response. Returns None if malformed."""
    merged_text = ""
    abstract = ""
    summary = ""
    for line in raw.splitlines():
        upper = line.upper()
        if upper.startswith("MERGED:"):
            merged_text = line.split(":", 1)[1].strip()
        elif upper.startswith("ABSTRACT:"):
            abstract = line.split(":", 1)[1].strip()
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
    if merged_text and abstract and summary:
        return MergeResult(text=merged_text, abstract=abstract, summary=summary)
    return None


def merge_texts(
    text_a: str,
    text_b: str,
    prefer: str = "equal",
    model: str = "",
) -> MergeResult:
    """Merge two chunk texts and generate abstract + summary.

    *prefer* controls relevance: "a", "b", or "equal".
    Retries up to ``_MERGE_RETRIES`` times on malformed responses.
    Returns merged text, abstract, and summary.
    """
    hint = _RELEVANCE_HINTS.get(prefer, _RELEVANCE_HINTS["equal"])
    system = _MERGE_SYSTEM_PROMPT.format(relevance_hint=hint)

    messages: list[Message] = [
        {
            "role": "user",
            "content": _MERGE_USER_TEMPLATE.format(chunk_a=text_a, chunk_b=text_b),
        },
    ]

    for _ in range(_MERGE_RETRIES):
        resp = chat(
            messages,
            model=model,
            system_prompt=system,
            stream=False,
            **DETERMINISTIC_PARAMS,
            cache_messages=False,
        )
        raw = (resp.get("content") or "").strip()
        result = _parse_merge_response(raw)
        if result is not None:
            return result
        # Feed the bad output back so the model can correct itself.
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": _RETRY_PROMPT})

    # All retries exhausted — return best-effort.
    return MergeResult(text=raw, abstract="", summary="")
