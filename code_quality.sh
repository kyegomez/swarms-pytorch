#!/bin/bash

# Navigate to the directory containing the 'swarms_torch' folder
# cd /path/to/your/code/directory

# Run autopep8 with max aggressiveness (-aaa) and in-place modification (-i)
# on all Python files (*.py) under the 'swarms_torch' directory.
autopep8 --in-place --aggressive --aggressive --recursive --experimental --list-fixes swarms_torch/

# Run black with default settings, since black does not have an aggressiveness level.
# Black will format all Python files it finds in the 'swarms_torch' directory.
black --experimental-string-processing swarms_torch/

# Run ruff on the 'swarms_torch' directory.
# Add any additional flags if needed according to your version of ruff.
ruff swarms_torch/

# YAPF
# yapf --recursive --in-place --verbose --style=google --parallel swarms_torch
