.PHONY: install lint format test api ui mlflow docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

api:
	uv run uvicorn geofence.api.main:app --reload --port 8290

ui:
	GEOFENCE_API_URL=http://localhost:8290 uv run streamlit run src/geofence/ui/app.py --server.port 8791

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5030

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
