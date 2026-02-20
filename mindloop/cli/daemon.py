"""CLI entry point for the session daemon."""

import argparse
import logging

from mindloop.daemon import _DEFAULT_MAX_FAILURES, run_daemon


def main() -> None:
    """Parse arguments and start the daemon."""
    parser = argparse.ArgumentParser(
        description="Run an agent session on a cron schedule."
    )
    parser.add_argument("--session", required=True, help="Session name.")
    parser.add_argument(
        "--schedule",
        required=True,
        help="Cron expression (e.g. '0 9 * * *' for daily at 9am).",
    )
    parser.add_argument("--model", default=None, help="Agent model.")
    parser.add_argument("--summarizer-model", default=None, help="Summarizer model.")
    parser.add_argument("--n-experts", type=int, default=1, help="Best-of-N.")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Output token budget per session."
    )
    parser.add_argument(
        "--allow-exec",
        action="store_true",
        help="Enable the sandboxed 'run' tool for agent sessions.",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run immediately on start, then follow schedule.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=_DEFAULT_MAX_FAILURES,
        help=f"Stop after N consecutive failures (default: {_DEFAULT_MAX_FAILURES}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_daemon(
        session=args.session,
        schedule=args.schedule,
        model=args.model,
        summarizer_model=args.summarizer_model,
        n_experts=args.n_experts,
        max_tokens=args.max_tokens,
        allow_exec=args.allow_exec,
        run_now=args.run_now,
        max_failures=args.max_failures,
    )
