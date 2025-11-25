#!/bin/bash
# Universal run script for all Python scripts in the project

cd "$(dirname "$0")"
export PYTHONPATH=.
exec ./venv/bin/python "$@"
