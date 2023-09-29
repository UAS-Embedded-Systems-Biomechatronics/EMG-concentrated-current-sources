#!/usr/bin/env bash
#

orig_loc=$(pwd)

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
cd $SCRIPT_DIR

maturin develop
python3 ./setup.py build
pip install -e ./

cd $orig_loc
