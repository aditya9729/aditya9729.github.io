#!/usr/bin/env bash
set -euo pipefail
# Advanced Linux GPU path; choose a documented model/env combination in INSTALL.md.
if [[ "$(uname -s)" != Linux ]]; then
  echo 'RLinf course path targets a Linux GPU server. Use core/SB3 on Mac.' >&2
  exit 2
fi
target="${1:-rlinf-course}"
git clone https://github.com/RLinf/RLinf.git "$target"
cd "$target"
git checkout 7e90641c4135229db6d3426950cdb45cee6259a6
bash requirements/install.sh embodied --model openvla --env maniskill_libero
.venv/bin/python -c 'import rlinf; print(rlinf.__file__)'
.venv/bin/python -m pip freeze > course-freeze.txt
