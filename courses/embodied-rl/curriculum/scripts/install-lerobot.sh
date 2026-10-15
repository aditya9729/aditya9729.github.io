#!/usr/bin/env bash
set -euo pipefail
# Separate from the learning environment. Requires Python3.12; ffmpeg guidance in INSTALL.md.
python_bin="${PYTHON_BIN:-python3.12}"
target="${1:-lerobot-course}"
git clone https://github.com/huggingface/lerobot.git "$target"
cd "$target"
git checkout 297774fca4e3ad0aa34e7425fbf937111637e77b
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[training]'
.venv/bin/python -c 'import lerobot; print(lerobot.__file__)'
.venv/bin/python -m pip freeze > course-freeze.txt
