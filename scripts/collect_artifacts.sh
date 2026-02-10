#!/usr/bin/env bash
# Move untracked .md files from the repo root to artifacts/.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mkdir -p artifacts

git ls-files --others --exclude-standard -- '*.md' | while read -r f; do
    # Only root-level files.
    [[ "$f" == */* ]] && continue
    mv "$f" artifacts/
    echo "moved $f -> artifacts/$f"
done
