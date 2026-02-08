"""Interactive chat CLI using OpenRouter."""

import logging
from datetime import datetime
from pathlib import Path

import requests

from mindloop.client import API_KEY, DEFAULT_MODEL, chat


def setup_logging() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return log_path


def main() -> None:
    if not API_KEY:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    log_path = setup_logging()
    print(f"Chatting with {DEFAULT_MODEL} via OpenRouter. Type 'quit' to exit.")
    print(f"Logging to {log_path}\n")
    messages = []

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
        try:
            print("Bot: ", end="", flush=True)
            reply = chat(messages)
            messages.append({"role": "assistant", "content": reply})
            logging.info("Bot: %s", reply)
            print()
        except requests.HTTPError as e:
            err = f"Error: {e.response.status_code} - {e.response.text}"
            logging.error(err)
            print(f"{err}\n")


if __name__ == "__main__":
    main()
