# 🚚 RouteMind AI — Predictive Delivery Disruption \& Recovery System

Predicts **whether a delivery will be late before it happens**, estimates **how late**, explains **why**, and recommends a **recovery action** — packaged as a full MLOps project (pipelines, API, dashboard, Docker, CI/CD, AWS + Azure, monitoring, retraining).

```
Delay Risk: HIGH   |   Expected delay: 37 min   |   P(delay): 99.9 %
Main factors: long distance 29 %, heavy traffic 23 %, low speed 17 %, rain 15 %
Action: reroute driver · notify customer
```

\*\*Docker images:\*\* \[routemind-api](https://hub.docker.com/r/shrawaniwanare/routemind-api) · \[routemind-dashboard](https://hub.docker.com/r/shrawaniwanare/routemind-dashboard)



Run it with one command:

`docker run -p 8000:8000 shrawaniwanare/routemind-api:latest`







## Quick start (5 minutes)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt

pytest                               # 39 tests
python -m src.pipeline.training\_pipeline      # generates data, trains, registers model

uvicorn api.main:app --reload --port 8000     # API  -> http://localhost:8000/docs
streamlit run dashboard/app.py                # UI   -> http://localhost:8501  (2nd terminal)
```

Try the monitoring loop (drift → retrain → safe promotion):

```bash
python -m scripts.simulate\_predictions --n 600                              # normal traffic
python -m scripts.simulate\_predictions --n 800 --drift --with-feedback --seed 7   # drifted traffic + real outcomes
python -m scripts.generate\_new\_data --n 6000 --drift                        # fresh labelled data
python -m src.pipeline.retraining\_pipeline                                  # retrains only if drift is detected
curl -X POST http://localhost:8000/model/reload                             # hot-swap the new model
```

Or everything in containers: `docker compose up --build` (API :8000, dashboard :8501).

👉 **Full step-by-step build guide:** [**BUILD\_GUIDE.md**](BUILD_GUIDE.md)

## What is inside

|Piece|Where|
|-|-|
|Pipelines: ingestion → validation → transformation → training → evaluation → registry|`src/components/`, `src/pipeline/training\_pipeline.py`|
|3 models: delay **classifier**, delay **regressor**, rule-based **recovery engine**|`model\_trainer.py`, `recovery\_engine.py`|
|Explanations (counterfactual risk factors)|`src/pipeline/prediction\_pipeline.py`|
|Versioned model registry + safe promotion|`src/models/model\_loader.py`, `model\_evaluation.py`|
|Monitoring: drift (PSI), latency, live accuracy from feedback|`src/components/monitoring.py`|
|Retraining (drift-triggered, champion/challenger)|`src/pipeline/retraining\_pipeline.py`|
|FastAPI service|`api/`|
|Streamlit dashboard|`dashboard/app.py`|
|Logging + custom exceptions|`src/utils/logger.py`, `exception.py`|
|Tests (pytest)|`tests/`|
|Docker / Compose|`docker/`, `docker-compose.yml`|
|CI/CD (GitHub Actions)|`.github/workflows/`|
|AWS (ECR + ECS Fargate) / Azure (ACR + Container Apps)|`deployment/`|

## API

|Method|Path|Purpose|
|-|-|-|
|POST|`/predict`|risk, probability, expected delay, factors, action|
|POST|`/predict/batch`|up to 500 deliveries|
|POST|`/feedback`|report the real delay of a finished delivery|
|GET|`/monitoring/summary` `/drift` `/recent`|model health|
|POST|`/model/reload`|load newest production model|
|GET|`/health`|liveness + model version|

## Important note about the data

No public dataset combines GPS, traffic, weather and delay, so `src/components/data\_generator.py` **simulates** deliveries with realistic cause-and-effect rules. The metrics you see (\~84 % accuracy, \~2.4 min MAE) describe *how well the model learned the simulator*, not real-world performance. To use real data, drop a CSV with the same columns at `data/raw/deliveries.csv` — nothing else changes.


**Docker images:** [routemind-api](https://hub.docker.com/r/shrawaniwanare/routemind-api) � [routemind-dashboard](https://hub.docker.com/r/shrawaniwanare/routemind-dashboard)

Run it with one command: `docker run -p 8000:8000 shrawaniwanare/routemind-api:latest`
