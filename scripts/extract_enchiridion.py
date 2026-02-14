"""Extract passages from the Enchiridion and distill into prompts."""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mindloop.client import chat

SRC = Path("Enchiridion.txt")
PASSAGES_OUT = Path("enchiridion_passages.json")
PROMPTS_OUT = Path("enchiridion_prompts.json")

# Max chars for a passage to be used as a direct quote.
_DIRECT_USE_MAX = 200

_DISTILL_MODEL = "deepseek/deepseek-v3.2"
_DISTILL_PROMPT = """\
Rephrase the following passage as a single reflective prompt for an autonomous AI agent.
Write as a question or challenge. One line only. Do not summarize — provoke thought.\
"""

_CHAPTERS = {
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
    "XXI",
    "XXII",
    "XXIII",
    "XXIV",
    "XXV",
    "XXVI",
    "XXVII",
    "XXVIII",
    "XXIX",
    "XXX",
    "XXXI",
    "XXXII",
    "XXXIII",
    "XXXIV",
    "XXXV",
    "XXXVI",
    "XXXVII",
    "XXXVIII",
    "XXXIX",
    "XL",
    "XLI",
    "XLII",
    "XLIII",
    "XLIV",
    "XLV",
    "XLVI",
    "XLVII",
    "XLVIII",
    "XLIX",
    "L",
    "LI",
}


def clean(text: str) -> str:
    """Strip footnote markers, italic markers, and normalize whitespace."""
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("_", "")
    # Collapse line breaks within paragraphs (PG wraps at ~72 chars).
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces.
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_chapters(raw: str) -> list[tuple[str, str]]:
    """Split into (chapter_num, text) pairs."""
    chapters: list[tuple[str, str]] = []
    current_num = ""
    current_lines: list[str] = []

    for line in raw.splitlines():
        stripped = re.sub(r"\[\d+\]", "", line).strip()
        if stripped in _CHAPTERS:
            if current_num:
                chapters.append((current_num, "\n".join(current_lines)))
            current_num = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_num:
        chapters.append((current_num, "\n".join(current_lines)))

    return chapters


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, return non-empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def distill_passage(text: str) -> str:
    """Distill a passage into a reflective prompt via LLM."""
    msg = chat(
        [{"role": "user", "content": text}],
        model=_DISTILL_MODEL,
        system_prompt=_DISTILL_PROMPT,
        stream=False,
        temperature=0,
        seed=42,
        cache_messages=False,
    )
    result: str = msg.get("content", "")
    return result.strip()


def main() -> None:
    raw = SRC.read_text()
    raw = re.sub(r"^\s*THE ENCHIRIDION\s*\n", "", raw)

    chapters = split_chapters(raw)
    print(f"Chapters: {len(chapters)}")

    passages: list[dict[str, str]] = []
    for num, text in chapters:
        for para in split_paragraphs(text):
            cleaned = clean(para)
            if len(cleaned) < 30:
                continue
            passages.append({"chapter": num, "text": cleaned})

    print(f"Passages: {len(passages)}")
    PASSAGES_OUT.write_text(json.dumps(passages, indent=2, ensure_ascii=False))

    # Split into direct quotes and distillation candidates.
    direct = [p for p in passages if len(p["text"]) <= _DIRECT_USE_MAX]
    to_distill = [p for p in passages if len(p["text"]) > _DIRECT_USE_MAX]
    print(f"Direct quotes: {len(direct)}")
    print(f"To distill: {len(to_distill)}")

    if "--dry-run" in sys.argv:
        print("Dry run, skipping distillation.")
        return

    # Distill in parallel.
    prompts: list[dict[str, str]] = []

    # Add direct quotes.
    for p in direct:
        prompts.append(
            {
                "type": "quote",
                "text": p["text"],
                "author": "Epictetus",
                "source": f"Enchiridion {p['chapter']}",
            }
        )

    # Distill longer passages.
    print(f"Distilling {len(to_distill)} passages...")
    ordered: list[tuple[int, dict[str, str], str]] = [None] * len(to_distill)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(distill_passage, p["text"]): i for i, p in enumerate(to_distill)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            p = to_distill[idx]
            result = future.result()
            ordered[idx] = (idx, p, result)
            done += 1
            print(f"  [{done}/{len(to_distill)}] {result[:80]}")

    for _, p, result in ordered:
        prompts.append(
            {
                "type": "distilled",
                "text": result,
                "source": f"Enchiridion {p['chapter']}",
            }
        )

    PROMPTS_OUT.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    print(
        f"\nTotal prompts: {len(prompts)} ({len(direct)} quotes + {len(to_distill)} distilled)"
    )
    print(f"Wrote {PROMPTS_OUT}")


if __name__ == "__main__":
    main()
