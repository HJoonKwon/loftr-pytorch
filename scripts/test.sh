#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC="${SCRIPT_DIR}/../python"

cd "${PYTHON_SRC}"
PYTHONPATH="${PYTHON_SRC}" pytest -s  
