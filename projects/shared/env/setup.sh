#!/usr/bin/env bash
# Set up a shared venv for all three research projects.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
VENV="$ROOT/.venv"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$HERE/requirements.txt"

# Optional extras (best-effort; failures are non-fatal)
pip install transformer_lens || echo "transformer_lens not installed (project 2 will fall back to manual hooks)"

echo "Done. Activate with: source $VENV/bin/activate"
