.PHONY: install lint fmt test data retrieval simulator validate check-env click-model dqn slateq ppo evaluate

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

data:
	uv run python scripts/prepare_data.py

retrieval:
	uv run python scripts/train_retrieval.py

simulator:
	uv run python scripts/train_simulator.py

validate:
	uv run python scripts/validate_simulator.py

check-env:
	uv run python scripts/check_env.py

click-model:
	uv run python scripts/train_click_model.py

dqn:
	uv run python scripts/train_agent.py agent=dqn

slateq:
	uv run python scripts/train_agent.py agent=slateq

ppo:
	uv run python scripts/train_agent.py agent=ppo

evaluate:
	uv run python scripts/evaluate.py
