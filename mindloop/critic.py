"""Cross-context critic: fresh-context review of agent actions."""

from collections.abc import Callable

from mindloop.client import chat

_CRITIC_PROMPT = (
    "You're reviewing an autonomous agent's recent actions. "
    "The agent sets its own goals — your job is to check "
    "whether it's reasoning well, not whether its goals are correct.\n\n"
    "Respond with ONLY one of:\n"
    '- "ok" if the agent is making meaningful progress\n'
    "- A single sentence describing the concern if you see: "
    "circular actions, lost thread, stagnation, or "
    "the agent doubling down on a failing approach"
)
_CRITIC_MAX_TOKENS = 100


def critic_review(
    activity: str,
    model: str,
    log: Callable[[str], None],
) -> str:
    """Review recent agent actions with fresh context.

    Returns empty string if the agent looks fine, or a ``[critic] ...``
    message describing the concern.
    """
    if not activity:
        return ""
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
    if answer.lower() == "ok":
        return ""
    return f"[critic] {answer}"
