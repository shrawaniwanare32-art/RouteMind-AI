# RouteMind AI — Step-by-Step Build Guide

This guide walks the project in the order you would build it. Every step names the file(s) that implement it, what the step does, and how to check it works. The complete code is in this repository; read each file as you reach its step.

---

## 0. How the whole system works

```
                    ┌─────────────── TRAINING (offline) ───────────────┐
 data/raw/*.csv ─►  Ingestion ─► Validation ─► Transformation ─► Trainer ─► Evaluation ─► Registry
 (or simulator)      merge        schema,        hash split,      2 models    metrics,      versions +
                     new data     ranges,        reference        (clf+reg)   champion vs   "latest.json"
                                  duplicates     sample                       challenger    pointer
                    └───────────────────────────────────────────────────┘
                                                   │  production model + drift baseline
                    ┌─────────────── SERVING (online) ─────────────────┐
 client ──POST /predict──► FastAPI ──► PredictionPipeline
                                        1 classifier → P(delay), class
                                        2 regressor  → expected minutes
                                        3 what-if explanation → risk factors
                                        4 recovery engine → action
                                        5 log request → logs/predictions.jsonl
                    └───────────────────────────────────────────────────┘
                    ┌─────────────── MONITORING (loop) ────────────────┐
 predictions.jsonl + /feedback ──► drift (PSI) + live accuracy ──► "RETRAIN_RECOMMENDED"
                                   ──► retraining_pipeline ──► promote ONLY if better ──► /model/reload
                    └───────────────────────────────────────────────────┘
```

**A request, end to end.** `POST /predict` → Pydantic validates types/ranges (bad input = HTTP 422) → `PredictionPipeline` builds a DataFrame → the *same* sklearn pipeline used in training (feature engineering + imputation + encoding + model) runs, so there is no train/serve skew → outputs are turned into a risk level, factors and an action → the request is appended to the prediction log.

**Three models.**
1. **Classifier** (`ON_TIME` / `SLIGHT_DELAY` / `HIGH_DELAY`) — gives P(delay) = 1 − P(ON_TIME).
2. **Regressor** — expected delay in minutes.
3. **Recovery engine** — rules mapping risk level (+ main factor) to an action. Rule-based on purpose; it is the easiest piece to upgrade later.

---

## 1. Prerequisites

Python 3.11+ (3.12 works), Git, Docker (for steps 12+), a GitHub account. AWS/Azure accounts only for steps 14–15.

## 2. Project skeleton

```bash
mkdir RouteMind-AI && cd RouteMind-AI
git init
mkdir -p .github/workflows config data/{raw,processed,external} notebooks \
  src/{components,pipeline,models,utils,config} api/routes dashboard tests \
  artifacts/{models,metrics} logs docker deployment/{aws,azure} scripts
# make every code folder a package
touch src/__init__.py src/{components,pipeline,models,utils,config}/__init__.py api/__init__.py api/routes/__init__.py scripts/__init__.py
```

Why this layout: `src/` = reusable ML logic, `api/` and `dashboard/` = thin front doors that only call `src/`, `tests/` mirrors it, `deployment/` and `.github/` = ops. Nothing lives in one giant `app.py`.

## 3. Foundations (build these first — everything imports them)

| File | What it does |
|---|---|
| `config/config.yaml` | Every tunable in one place: paths, split size, model params, risk thresholds, drift thresholds, promotion rule. |
| `src/config/configuration.py` | Loads the YAML. `ROUTEMIND_HOME` env var relocates data/artifacts/logs (tests and containers use it). |
| `src/utils/schema.py` | Column names and allowed values — single source of truth. |
| `src/utils/logger.py` | One logger → console **and** `logs/application.log` (rotating). |
| `src/utils/exception.py` | `CustomException` + subclasses (`DataValidationException`, `PredictionException`, …). Messages include the **file and line** where things broke. |
| `src/utils/helpers.py` | Atomic JSON save/load. |

Check: `python -c "from src.utils.logger import get_logger; get_logger('x').info('hello')"` prints a line and creates `logs/application.log`.

## 4. Data layer (the pipeline components)

| Step | File | What it does |
|---|---|---|
| 4a | `src/components/data_generator.py` | Simulates deliveries with cause-and-effect rules (traffic, rain, peak hour, congestion, speed, driver experience → delay). Has a `drift=True` mode for the monitoring demo. **Replace with real data when you have it.** |
| 4b | `src/components/data_ingestion.py` | Loads `data/raw/deliveries.csv` (auto-generates if missing), optionally merges `data/external/new_deliveries.csv` for retraining → `data/processed/ingested.csv`. |
| 4c | `src/components/data_validation.py` | Schema check, type coercion, category cleaning, duplicate removal (by `order_id`), missing-ratio limit, range checks (drops impossible rows), IQR outlier report → `validated.csv` + `artifacts/metrics/validation_report.json`. |
| 4d | `src/components/data_transformation.py` | Feature engineering (`is_peak_hour`, `hour_sin/cos`, `eta_travel_min`, interactions), the sklearn preprocessing pipeline, delay labels, and the **train/test split**. |

**Two design decisions worth knowing (and explaining in interviews):**
- Feature engineering + imputation + encoding are *inside* the saved sklearn `Pipeline`, so serving cannot drift from training.
- The split is a **deterministic hash of `order_id`**, not a random split. An order always lands in the same split on every retrain, so the production model is never scored on rows it was trained on. (An earlier version of this project used a random split and the champion/challenger comparison was silently biased toward the old model — the hash split fixes that. `tests/test_transformation.py::test_split_is_stable_across_retrains` guards it.)

## 5. Models

| File | What it does |
|---|---|
| `src/components/model_trainer.py` | Trains classifier + regressor (scikit-learn `HistGradientBoosting*`; swap for XGBoost/LightGBM by changing the estimator in one line). |
| `src/components/model_evaluation.py` | Accuracy, precision, recall, macro-F1, ROC-AUC (delay vs on-time), MAE, RMSE, R², confusion matrix. Also the **promotion rule**: deploy only if macro-F1 doesn't drop and MAE isn't >5 % worse. |
| `src/models/model_loader.py` | Model registry: `artifacts/models/<version>/{classifier,regressor}.joblib + metadata.json + reference.csv`, and `latest.json` pointing at production. `promote()` also activates that version's drift baseline. |
| `src/pipeline/training_pipeline.py` | Runs steps 4→5 in order, saves a candidate, scores the current production model on the *same* test set, promotes only if better. |

Run it: `python -m src.pipeline.training_pipeline` → prints accuracy/F1/MAE and the new version id.

## 6. Recovery engine

`src/components/recovery_engine.py`

Risk is decided by **severity, gated by probability**: LOW if P(delay) < 0.35 or expected delay < 5 min; otherwise MEDIUM (< 20 min), HIGH (≥ 20), CRITICAL (≥ 40). Actions: `CONTINUE`, `MONITOR`, `REROUTE`, `PRIORITIZE_AND_NOTIFY`, plus a factor-specific tip (e.g. rain + bike → consider a car). All thresholds are in `config.yaml`.

## 7. Prediction pipeline

`src/pipeline/prediction_pipeline.py`

Loads the production model, validates required inputs (`Required feature 'traffic_level' is missing` + file/line), predicts, and **explains** each prediction by counterfactuals: for each factor (traffic, rain, peak hour, congestion, distance, speed, driver experience) it re-predicts with that factor set to a "normal" value, and the minutes that disappear are that factor's impact. Optional inputs (`temperature`, `route_congestion`) are imputed.

## 8. API

| File | What it does |
|---|---|
| `api/schemas.py` | Pydantic request/response models with ranges (`traffic_level` 0–10, `hour` 0–23, …). Accepts `distance` as an alias of `distance_km`. |
| `api/routes/prediction.py` | `/predict`, `/predict/batch`, `/feedback`, `/model/reload` |
| `api/routes/monitoring.py` | `/monitoring/summary`, `/drift`, `/recent` |
| `api/main.py` | App + lifespan (loads model; trains one on first start if none exists) + exception handlers that return the file/line JSON. |

Run: `uvicorn api.main:app --reload --port 8000` → open `http://localhost:8000/docs`.

## 9. Dashboard

`dashboard/app.py` (Streamlit, talks to the API via `API_URL`): **Overview** (KPIs, risk distribution, highest-risk deliveries), **Predict** (form → risk, factors, actions), **Model monitor** (training metrics, per-feature PSI, live accuracy, latency).

Run in a second terminal: `streamlit run dashboard/app.py`.

## 10. Monitoring & retraining

| File | What it does |
|---|---|
| `src/components/monitoring.py` | Logs every prediction to `logs/predictions.jsonl` and feedback to `feedback.jsonl`. Computes **PSI** per feature against the production model's training sample (PSI > 0.2 = drifted; ≥ 2 drifted features = flag), latency stats, and live MAE/accuracy once `/feedback` outcomes arrive. |
| `src/pipeline/retraining_pipeline.py` | If drift / live-accuracy drop / `--force`: retrain on original + new data, promote only if better. |
| `scripts/simulate_predictions.py` | Fills the log with normal or drifted traffic (optionally with real outcomes). |
| `scripts/generate_new_data.py` | Writes labelled new data to `data/external/`. |

The demo loop is in the README. Note the drift window is the **last 1000 predictions** (`monitoring.window_size`) so it reflects recent traffic rather than being diluted by history.

## 11. Tests

`pytest` runs 39 tests in a throw-away temp home (they never touch your real models): ingestion, validation (bad rows, missing columns), feature engineering, split stability, training + promotion rules, drift baseline only changes on promotion, recovery rules, prediction shape/behaviour/exceptions, PSI + drift + live accuracy + retraining, and every API endpoint incl. invalid input.

## 12. Docker

`docker/Dockerfile.api` bakes a trained model into the image at build time and runs as a non-root user with a health check. `docker/Dockerfile.dashboard` is separate and small. `docker-compose.yml` runs both with named volumes for logs and model versions.

```bash
docker compose up --build
# API http://localhost:8000/docs   Dashboard http://localhost:8501
```

## 13. CI — `.github/workflows/ci.yml`

On every push/PR: install → `ruff check` → `pytest` → build the Docker image → Trivy scan (fails on CRITICAL with a fix available) → start the container and hit `/health` and `/predict`.

## 14. CD to AWS (ECR + ECS Fargate)

1. `aws configure` (an IAM user with ECR + ECS + IAM-role permissions), then `bash deployment/aws/setup_aws.sh` — creates the ECR repo, log group, cluster, task definition, security group and Fargate service, and pushes the first image.
2. GitHub → Settings → Secrets and variables → Actions: secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; variables `AWS_REGION`, `DEPLOY_AWS=true`.
3. Push to `main` → CI passes → `cd.yml` builds, pushes to ECR, renders the task definition with the new image, deploys with `wait-for-service-stability`, then health-checks the task's public IP.

## 15. CD to Azure (ACR + Container Apps)

1. `az login`, then `bash deployment/azure/setup_azure.sh` — creates the resource group, ACR, Container Apps environment and app, builds the first image inside ACR, and prints the service-principal JSON.
2. GitHub secrets/variables: secret `AZURE_CREDENTIALS` (that JSON); variables `ACR_NAME`, `AZURE_RESOURCE_GROUP`, `DEPLOY_AZURE=true`.
3. The same push now also builds in ACR (`az acr build`), rolls a new Container Apps revision, and health-checks the public URL.

---

## What has and hasn't been verified

Verified by running it: the full training pipeline, all 39 tests, the FastAPI server over real HTTP, all three dashboard pages against the live API, and the drift → retrain → promote → hot-reload loop. YAML/JSON/bash syntax of the deployment files was checked.

**Not run** (needs Docker / your cloud accounts): the Docker builds, the GitHub Actions workflows, and the AWS/Azure scripts. Expect to debug small things on the first real run (account IDs, region, quota, permissions) — that is normal for a first cloud deploy.

## Limits & the upgrade path

- **Data is simulated.** The metrics measure the simulator. With real data, expect lower numbers and revisit features/thresholds.
- **Logs are local files.** In the cloud each container has its own logs and they vanish on redeploy. For real monitoring, write `predictions.jsonl` / `feedback.jsonl` to PostgreSQL, S3 or Azure Blob and have the retraining job read from there.
- **Model artifacts are baked into the image.** Next step: store versions in S3 / Azure Blob (or MLflow's registry) and have the API pull `latest` at start-up.
- **Retraining is manual/CLI.** Next step: a scheduled GitHub Actions workflow (`on: schedule`) that runs the retraining pipeline against your data store and triggers a redeploy.
- **The API is open and the AWS demo exposes port 8000 to the internet.** Add an API key or OAuth, an ALB with HTTPS, and least-privilege IAM/OIDC (instead of long-lived access keys) before calling it production.
- **Other upgrades:** MLflow experiment tracking, PostgreSQL for orders, XGBoost/LightGBM, SHAP for explanations, a learned recovery policy.

## Interview talking points

- Why the preprocessing lives inside the model pipeline (no train/serve skew).
- The hash-split leakage bug and fix (evaluation integrity across retrains).
- Champion/challenger promotion instead of "always deploy the newest model".
- Drift baseline tied to the production model, PSI on a recent window.
- What you'd change to take it from portfolio to production (the limits above).
