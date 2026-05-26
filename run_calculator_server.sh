#!/bin/bash
set -euo pipefail

# Paths to define
VENV_PATH="./venv"
SERVER_PATH="./calculator_server.py"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the server with the default stdio transport
python "$SERVER_PATH" --stdio
