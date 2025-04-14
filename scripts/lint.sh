#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run mypy --ignore-missing-imports multixem/
poetry run isort --check --diff multixem/ tests/
poetry run black --check multixem/ tests/
poetry run flake8 multixem/ tests/
poetry run safety check -i 39462 -i 40291
poetry run bandit -r multixem/
