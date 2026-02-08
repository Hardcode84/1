"""OpenRouter API client for chat and embeddings."""

import json
import os

import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def chat(messages, model=DEFAULT_MODEL):
    """Stream a chat response, printing tokens as they arrive."""
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
    )
    response.raise_for_status()

    full_reply = []
    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        token = chunk["choices"][0]["delta"].get("content", "")
        print(token, end="", flush=True)
        full_reply.append(token)

    print()
    return "".join(full_reply)


def get_embeddings(texts, model=DEFAULT_EMBEDDING_MODEL):
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
