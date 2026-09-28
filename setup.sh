#!/usr/bin/env bash
# FlyKart one-command setup (Linux, macOS, or Windows via WSL2).
#   ./setup.sh
# Installs uv if needed, then Python + dependencies, then the connectome data.
# Safe to re-run: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
trap 'printf "\n\033[31mSetup stopped with an error (see above).\033[0m Run it again, or check README.md > Troubleshooting.\n"' ERR

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) echo "Please run this inside WSL2 (Windows), Linux, or macOS. See README.md."; exit 1 ;;
esac

bold "1/4  uv (Python package manager)"
if ! command -v uv >/dev/null 2>&1; then
  [ -x "$HOME/.local/bin/uv" ] || {
    echo "uv not found, installing it (https://docs.astral.sh/uv/)..."
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    else
      wget -qO- https://astral.sh/uv/install.sh | sh
    fi
    NEW_UV=1
  }
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

bold "2/4  Python + dependencies (first time: a few GB, mostly PyTorch/CUDA)"
uv sync

bold "3/4  Connectome data (~1.2 GB download, one time)"
uv run flykart prepare

bold "4/4  Checking your setup"
uv run flykart doctor

bold "Done! Start FlyKart with:"
echo "    uv run flykart"
echo "then open http://localhost:8765 and press Run."
if [ "${NEW_UV:-0}" = 1 ]; then
  echo
  echo "(uv was just installed: open a new terminal first, or run: source \$HOME/.local/bin/env)"
fi
