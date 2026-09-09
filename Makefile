install:
	uv sync

run:
	uv run python -m src

tests:
	uv run python -m unittest discover -s tests

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} +

lint:
	uv run flake8 src tests
	uv run mypy src tests --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
