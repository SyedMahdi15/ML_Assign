#!/bin/bash
# Use the project virtualenv (Python 3.11 + TensorFlow).
# Python 3.13 system-wide crashes TensorFlow on Mac ("Python quit unexpectedly").

set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating .venv with Python 3.11..."
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Install Python 3.11 first: brew install python@3.11"
    exit 1
  fi
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

exec "$@"
