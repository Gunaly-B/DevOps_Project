# 🫀 MLOps Capstone Project: End-to-End Heart Disease Prediction Pipeline

A complete, production-grade MLOps pipeline for Heart Disease Risk Prediction built with **DVC**, **MLflow**, **FastAPI**, **Docker**, **Pytest**, and **GitHub Actions**.

---

## 📐 Pipeline Architecture

```
                                    +-----------------------------------+
                                    |     UCI Cleveland Dataset        |
                                    +-----------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |    1. Data Ingestion & Split      |
                                    |       (src/data_loader.py)        |
                                    +-----------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |     2. Preprocessing & Scaling    |
                                    |       (src/preprocessing.py)      |
                                    +-----------------------------------+
                                                      |
                                                      v
+---------------------------------------------------------------------------------------------------+
|                                   3. Model Training & Factory                                     |
| Logistic Regression (F1: 0.8814)  |  Random Forest (F1: 0.8621)  |  Gradient Boosting (F1: 0.8667) |
+---------------------------------------------------------------------------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |    4. MLflow Experiment Tracking  |
                                    |       (src/mlflow_utils.py)       |
                                    +-----------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |   5. MLflow Model Registry        |
                                    |   (Promoted alias: Production)    |
                                    +-----------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |     6. FastAPI Service            |
                                    |     POST /predict | GET /health   |
                                    +-----------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |     7. Docker Containerization    |
                                    |    (Multi-stage build / Uvicorn)  |
                                    +-----------------------------------+
                                                      |
                                                      v
                                    +-----------------------------------+
                                    |   8. GitHub Actions CI/CD         |
                                    |  (Checkout -> Test -> Docker Build)|
                                    +-----------------------------------+
```

---

## 📁 Required Project Structure

```
Devops project/
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── mlflow_utils.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── schemas.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_models.py
│   ├── test_predict.py
│   └── test_preprocessing.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── dvc.yaml
├── dvc.lock
├── config.yaml
├── README.md
└── .gitignore
```

---

## ⚡ Quickstart & Setup Guide

### 1. Environment Setup

Clone the repository and install requirements:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Data Ingestion & DVC Pipeline Execution

Initialize DVC and reproduce the data ingestion and training stages:

```bash
# Initialize DVC
dvc init --no-scm

# Check pipeline status
dvc status

# Reproduce pipeline end-to-end (prepare -> train)
dvc repro
```

---

### 3. MLflow Tracking & Model Registry UI

Launch the MLflow UI to view model experiment comparisons and registered model aliases:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open your browser at `http://127.0.0.1:5000` to inspect:
- Experiment runs for **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
- Hyperparameter comparisons and metrics (F1-score, ROC-AUC, Accuracy, Precision, Recall).
- Registered Model: `heart-disease-classifier` promoted to **Production** stage.

---

### 4. Running the FastAPI Prediction Service

Start the FastAPI ASGI server locally:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive Swagger API documentation at `http://127.0.0.1:8000/docs`.

#### Sample Payload (`POST /predict`):

```json
{
  "age": 57.0,
  "sex": 1.0,
  "cp": 4.0,
  "trestbps": 140.0,
  "chol": 241.0,
  "fbs": 0.0,
  "restecg": 1.0,
  "thalach": 123.0,
  "exang": 1.0,
  "oldpeak": 1.5,
  "slope": 2.0,
  "ca": 0.0,
  "thal": 7.0
}
```

#### Sample Response:

```json
{
  "prediction": 1,
  "heart_disease_risk": "High Risk (Heart Disease Present)",
  "probability": 0.9412,
  "model_version": "MLflow Registry Alias: Production"
}
```

---

### 5. Running Containerized Application with Docker

Build and execute the multi-stage Docker container:

```bash
# Build Docker image
docker build -t heart-disease-mlops:latest .

# Run Docker container
docker run -p 8000:8000 heart-disease-mlops:latest
```

Verify readiness endpoint:

```bash
curl http://localhost:8000/health
```

---

### 6. Running Pytest Test Suite

Execute unit and integration tests:

```bash
pytest -v
```

---

## 📊 Deliverables & Verification Checklist

- [x] **Modular Pipeline (`src/`)**: Single-purpose modules for ingestion, preprocessing, factory models, evaluation, tracking, and inference.
- [x] **DVC Versioning (`dvc.yaml`)**: Verified with `dvc status` ("Data and pipelines are up to date.") and reproducible via `dvc repro`.
- [x] **MLflow Tracking**: Complete hyperparameter & metric tracking with `sqlite:///mlflow.db`.
- [x] **Model Registry**: Best model `logistic_regression` (F1: `0.8814`) registered as `heart-disease-classifier` version 1/2 with alias `Production`.
- [x] **FastAPI API**: Validated `GET /health` and `POST /predict` endpoints with Pydantic schemas.
- [x] **Docker Container**: Multi-stage `Dockerfile` serving FastAPI via `uvicorn`.
- [x] **Pytest Coverage**: All 9 unit and integration tests passing.
- [x] **GitHub Actions CI/CD**: `.github/workflows/ci.yml` verifying build, test execution, and Docker container creation.
