.PHONY: install lint fmt test data retrieval simulator validate check-env click-model dqn slateq ppo evaluate seeds evaluate-seeds trajectories api viewer up

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

seeds:
	uv run python scripts/run_seeds.py --parallel 6

evaluate-seeds:
	uv run python scripts/evaluate_seeds.py

trajectories:
	uv run python scripts/export_trajectories.py

api:
	uv run uvicorn slate_rl.serving.api:app --port 8000

viewer:
	uv run streamlit run viewer/app.py

up:
	docker compose up --build
