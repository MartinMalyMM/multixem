#!/bin/bash
set -euxo pipefail

poetry run isort multixem/ tests/
poetry run black multixem/ tests/
