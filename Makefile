.PHONY: style

style:
	uv run ruff check --fix --select I .
	uv run ruff format .
