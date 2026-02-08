"""OpenRouter API client for chat and embeddings."""

import json
import os
from collections.abc import Callable
from typing import Any

import requests

API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "openrouter/free"
DEFAULT_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

Message = dict[str, Any]
Tool = dict[str, Any]


def _default_on_token(token: str) -> None:
    print(token, end="", flush=True)


def chat(
    messages: list[Message],
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
    stream: bool = True,
    on_token: Callable[[str], None] = _default_on_token,
    temperature: float | None = None,
    seed: int | None = None,
) -> str:
    """Send a chat completion request. Streams by default."""
    full_messages = list(messages)
    if system_prompt is not None:
        full_messages.insert(0, {"role": "system", "content": system_prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
    }
    if tools:
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed

    if not stream:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        )
        response.raise_for_status()
        result: str = response.json()["choices"][0]["message"]["content"]
        return result

    payload["stream"] = True
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        stream=True,
    )
    response.raise_for_status()

    full_reply: list[str] = []
    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        token = chunk["choices"][0]["delta"].get("content", "")
        on_token(token)
        full_reply.append(token)

    return "".join(full_reply)


def get_embeddings(
    texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL
) -> list[list[float]]:
    """Fetch embeddings for a list of texts in a single batch request."""
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "input": texts},
    )
    response.raise_for_status()
    data = response.json()["data"]
    # Sort by index to match input order.
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
