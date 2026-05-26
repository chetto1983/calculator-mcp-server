#!/bin/bash
set -euo pipefail

python -m doctest -v calculator_server.py
