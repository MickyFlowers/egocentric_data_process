#!/usr/bin/env bash

set -euo pipefail

python3 -m pip install --upgrade -r requirements.txt
python3 scripts/patch_chumpy.py
