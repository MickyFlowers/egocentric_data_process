#!/usr/bin/env bash

set -euo pipefail


cd ../
git clone https://gh-proxy.org/https://github.com/hassony2/manopth.git
cd manopth
pip install -e .
cd ../egocentric_data_process
pip install chumpy --no-build-isolation
python3 -m pip install --upgrade -r requirements.txt

python3 scripts/patch_chumpy.py

conda install -c conda-forge pinocchio hpp-fcl
pip install genesis-world