.PHONY: install train api dashboard test lint simulate simulate-drift new-data retrain up down

install:
	pip install -r requirements-dev.txt

train:
	python -m src.pipeline.training_pipeline

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest

lint:
	ruff check .

simulate:
	python -m scripts.simulate_predictions --n 600

simulate-drift:
	python -m scripts.simulate_predictions --n 800 --drift --with-feedback --seed 7

new-data:
	python -m scripts.generate_new_data --n 6000 --drift

retrain:
	python -m src.pipeline.retraining_pipeline

up:
	docker compose up --build

down:
	docker compose down
