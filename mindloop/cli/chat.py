"""Interactive chat CLI using OpenRouter."""

import json
import logging
from datetime import datetime
from pathlib import Path

import requests

from mindloop.client import API_KEY, DEFAULT_MODEL, chat


def setup_logging() -> tuple[Path, Path]:
    """Set up human-readable log and structured JSONL log. Returns (log_path, jsonl_path)."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.log"
    jsonl_path = log_dir / f"{timestamp}.jsonl"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return log_path, jsonl_path


def _log_jsonl(path: Path, role: str, content: str) -> None:
    """Append a structured JSON line to the JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    if not API_KEY:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    log_path, jsonl_path = setup_logging()
    print(f"Chatting with {DEFAULT_MODEL} via OpenRouter. Type 'quit' to exit.")
    print(f"Logging to {log_path}\n")
    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        logging.info("You: %s", user_input)
        _log_jsonl(jsonl_path, "user", user_input)
        try:
            print("Bot: ", end="", flush=True)
            msg = chat(messages)
            print()
            reply = msg.get("content", "")
            messages.append(msg)
            logging.info("Bot: %s", reply)
            _log_jsonl(jsonl_path, "assistant", reply)
            print()
        except requests.HTTPError as e:
            err = f"Error: {e.response.status_code} - {e.response.text}"
            logging.error(err)
            print(f"{err}\n")


if __name__ == "__main__":
    main()
