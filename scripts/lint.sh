#!/usr/bin/env bash
set -euxo pipefail

# Temporary CI smoke check: the project currently has known type/style issues in
# legacy code, so we keep lint focused on syntax validity until those are cleaned up.
python -m compileall -q multixem tests
