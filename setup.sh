#!/bin/bash
set -e

uv sync
cd models/GroundingDINO/ops
uv run python setup.py build install

# test the installation
uv run python test.py
