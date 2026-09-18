#!/usr/bin/env bash
# Run a command in the image with this repo at /app and your pack mounted read-only.
#
#   PACK=/path/to/Ternary-Bonsai-2-27B-mlx-2bit docker/run.sh \
#       python run.py --pack /pack "your prompt"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="${PACK:?set PACK to your Ternary-Bonsai-2-27B-mlx-2bit directory}"

TTY=()
[[ -t 0 && -t 1 ]] && TTY=(-it)

exec docker run --rm "${TTY[@]}" \
    --user "$(id -u):$(id -g)" \
    -v "$REPO:/app" \
    -v "$PACK:/pack:ro" \
    -e PYTHONPATH=/app \
    -w /app \
    bonsai2-abliterate "$@"
