.PHONY: install lint format typecheck test run download-model setup-service

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/edgeguard

test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

run:
	python -m edgeguard.api.server

download-model:
	bash scripts/download_model.sh

setup-service:
	bash scripts/setup_service.sh

setup-runner:
	bash scripts/setup_runner.sh

# Run the full check suite locally before pushing
check: lint typecheck test
