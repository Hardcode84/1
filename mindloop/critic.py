"""Cross-context critic: fresh-context review of agent actions."""

from collections.abc import Callable
from dataclasses import dataclass

from mindloop.client import chat

_CRITIC_PROMPT = (
    "You're reviewing an autonomous agent's recent actions. "
    "The agent sets its own goals — your job is to check "
    "whether it's reasoning well, not whether its goals are correct.\n\n"
    "Respond with exactly two lines:\n\n"
    "Line 1 — review:\n"
    '- "ok" if the agent is making meaningful progress\n'
    "- A single sentence describing the concern if you see: "
    "circular actions, lost thread, stagnation, or "
    "the agent doubling down on a failing approach\n\n"
    "Line 2 — pin:\n"
    '- "no_pin" if this window is routine\n'
    "- A short reason to preserve this window if it contains: "
    "a key decision with rationale, a surprising error or discovery, "
    "a breakthrough, or an unresolved problem that needs follow-up"
)
_CRITIC_MAX_TOKENS = 150


@dataclass
class CriticResult:
    """Parsed two-line critic response."""

    concern: str  # Empty if ok, "[critic] ..." if concern.
    pin_reason: str  # Empty if routine, reason string if worth pinning.


def _parse_critic_response(answer: str) -> CriticResult:
    """Parse a two-line critic response into a CriticResult.

    Graceful: single-line → no pin, empty → both empty.
    """
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]

    if not lines:
        return CriticResult(concern="", pin_reason="")

    # Line 1: concern.
    concern = "" if lines[0].lower() == "ok" else f"[critic] {lines[0]}"

    # Line 2: pin reason.
    pin_reason = ""
    if len(lines) >= 2 and lines[1].lower() != "no_pin":
        pin_reason = lines[1]

    return CriticResult(concern=concern, pin_reason=pin_reason)


def critic_review(
    activity: str,
    model: str,
    log: Callable[[str], None],
) -> CriticResult:
    """Review recent agent actions with fresh context.

    Returns a :class:`CriticResult` with concern and pin_reason fields.
    """
    if not activity:
        return CriticResult(concern="", pin_reason="")
    resp = chat(
        [
            {
                "role": "user",
                "content": f"{_CRITIC_PROMPT}\n\nRecent activity:\n{activity}",
            }
        ],
        model=model,
        stream=False,
        cache_messages=False,
        max_tokens=_CRITIC_MAX_TOKENS,
    )
    answer = str(resp.get("content", "")).strip()
    log(f"\n[critic] {answer}")
    return _parse_critic_response(answer)
