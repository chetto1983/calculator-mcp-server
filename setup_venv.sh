#!/bin/bash
set -euo pipefail

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package and developer tools
pip install -e ".[dev]"

# Print success message
echo "Virtual environment setup complete! You can now run the server using ./run_calculator_server.sh"
