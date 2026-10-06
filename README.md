# 🏥 Patient Readmission Prediction & Real-Time MLOps Deployment Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.5+-blueviolet.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course**: AI & Machine Learning — Unit 5: Machine Learning Operations (MLOps)  
> **Assignment**: Unit End Project 1: Predictive Modeling and Deployment  
> **Student**: Saif Pathan  

---

## 📌 Executive Summary

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline designed for healthcare analytics to predict **patient 30-day readmission risk** (mapped to the demographic and clinical features of the Titanic dataset). 

The system implements:
1. **Data Cleaning & Preprocessing**: Missing value imputation and categorical encoding using Scikit-Learn `ColumnTransformer`.
2. **Model Tournament**: Multi-model evaluation comparing Logistic Regression, Random Forest, and Gradient Boosting.
3. **MLflow Experiment Tracking & Governance**: Systematic logging of hyperparameters, evaluation metrics, and champion model registration in the MLflow Model Registry.
4. **Real-Time Serving via FastAPI**: Production REST API (`/health` and `/predict`) with Pydantic payload validation.
5. **Containerization**: Portable, production-grade `Dockerfile` with healthcheck probes.
6. **Continuous Monitoring & Drift Detection**: Automated statistical feature drift diagnostic with automated retraining threshold logic.

---

## 🏗️ System Architecture

```
   +------------------------+
   |   titanic.csv Data     |
   +------------------------+
               │
               ▼
   +------------------------+
   |   EDA & Preprocessing  | ──► ColumnTransformer (StandardScaler + OneHotEncoder)
   +------------------------+
               │
               ▼
   +------------------------+
   |    Model Tournament    | ──► Logistic Regression vs. Random Forest vs. Gradient Boosting
   +------------------------+
               │
               ▼
   +------------------------+
   | MLflow Model Registry  | ──► Champion Model Tracked & Versioned (sqlite:///mlflow.db)
   +------------------------+
               │
               ▼
   +------------------------+
   |  FastAPI Microservice  | ──► REST Endpoints: GET /health | POST /predict (Port 8000)
   +------------------------+
               │
               ▼
   +------------------------+
   |  Docker Container      | ──► Portable image (python:3.11-slim) with healthcheck
   +------------------------+
               │
               ▼
   +------------------------+
   | Drift & Monitoring     | ──► Evidently AI / Statistical Drift Diagnostic & Retraining Trigger
   +------------------------+
```

---

## 📊 Model Tournament Results

All models were evaluated on an 80/20 stratified holdout test split ($N=179$):

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **82.12%** | **80.33%** | **71.01%** | **0.7538** | **0.8725** | 🏆 **CHAMPION MODEL** |
| **Random Forest** | 79.89% | 78.95% | 65.22% | 0.7143 | 0.8415 | Runner-up |
| **Gradient Boosting** | 79.33% | 78.57% | 63.77% | 0.7040 | 0.8189 | Baseline |

The Champion model was serialized as `best_model.pkl` and cataloged in the **MLflow Model Registry** under `Patient_Readmission_Prediction` (Version 1).

---

## ⚡ Real-Time FastAPI Serving

### Endpoints:
- `GET /health`: Microservice health status probe.
- `POST /predict`: Real-time inference endpoint accepting patient features.

### Sample Request:
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 38.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 71.28,
  "Embarked": "C"
}
```

### Sample Response:
```json
{
  "readmission_probability": 0.9127,
  "prediction": 1,
  "risk_level": "High Risk"
}
```

---

## 🐳 Docker Deployment Instructions

```bash
# 1. Build the Docker container image
docker build -t patient-readmission-api .

# 2. Run the container mapping port 8000
docker run -d -p 8000:8000 --name readmission-service patient-readmission-api

# 3. Test the container healthcheck
curl http://localhost:8000/health
```

---

## 📁 Repository Structure

```
├── Unit_5_Project_1.ipynb   # Master interactive Jupyter Notebook
├── app.py                   # Production FastAPI REST microservice
├── test_api.py              # Automated test client for REST endpoints
├── train_pipeline.py        # Standalone training & MLflow tracking script
├── best_model.pkl           # Serialized Champion model pipeline artifact
├── Dockerfile               # Production Docker container specification
├── requirements.txt         # Project dependencies manifest
├── titanic.csv              # Healthcare demographic dataset
└── README.md                # Project documentation & architecture
```
