# 🏥 MEDINSIGHT AI: Enterprise Medical Insurance Risk Intelligence & Actuarial Analytics Platform
### Production-Grade Machine Learning Underwriting Engine with Explainable AI (SHAP), Conformal Uncertainty & 17 Functional Modules

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Bun](https://img.shields.io/badge/Bun-1.4.4-fbf0df.svg)](https://bun.sh/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg)](https://tailwindcss.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384.svg)](https://www.chartjs.org/)
[![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP_0.49-green.svg)](https://shap.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, academic and viva-ready Medical Insurance Premium Prediction & Risk Intelligence Platform built by Senior ML Engineers, Healthcare Data Scientists, and Full-Stack Architects. Features genuine Kaggle benchmark dataset integration (1,338 records), multi-model benchmarking (8 algorithms), split-conformal 90% prediction intervals, exact Tree SHAP explainability, persistent SQLite/SQLAlchemy database, JWT role-based security, automated ReportLab PDF underwriting reports, an interactive What-If scenario simulator, an AI conversational insurance assistant, and a responsive 17-module single page application.

---

## 🌟 The 17 Fully Functional Modules

| # | Module View | Primary Capabilities & Live Actions | Backend API Endpoints |
|---|---|---|---|
| **1** | **Landing & Overview** | Executive platform overview, architecture highlights, Kaggle dataset origin, interactive risk calculator preview. | `/api/health`, `/api/analytics/dashboard` |
| **2** | **User Dashboard** | KPI overview (average cost, active risk breakdown, population distribution), interactive Chart.js graphs. | `/api/analytics/dashboard` |
| **3** | **Insurance Prediction** | Multi-category input (Personal, Medical, Lifestyle, Coverage) with client validation, conformal intervals, and SHAP cards. | `POST /api/predict` |
| **4** | **AI Risk Analysis** | Actuarial decomposition: clinical risk drivers, age-BMI synergy effects, lifestyle penalty breakdowns, and actionable health guidance. | `/api/analytics/feature-importance` |
| **5** | **ML Analytics & Models** | Multi-model benchmarks (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost), $R^2$, RMSE, MAE, Residual charts. | `GET /api/analytics/benchmarks` |
| **6** | **What-if Simulator** | Interactive real-time sliders (Age, BMI, Smoking, Children, Region) that re-query the genuine model, compute exact cost deltas, and render scenario comparison charts. | `POST /api/predict/simulate` |
| **7** | **Plan Comparison** | Side-by-side comparison of 4 verified illustrative plan categories (Standard HMO, Comprehensive PPO, HDHP, Senior & Critical Care Gold) with deductibles, copay, and exclusions. | `GET /api/plans`, `POST /api/plans/recommend` |
| **8** | **AI Insurance Assistant** | Conversational insurance intelligence: clarifies deductibles, copays, waiting periods, SHAP risk drivers, and policy clauses with structured offline actuarial engine. | `POST /api/assistant/chat` |
| **9** | **Prediction History** | Searchable audit table with risk filters, date sorting, CSV export, PDF download, and record deletion. | `GET /api/history`, `GET /api/history/export/csv` |
| **10** | **Certified Reports** | Official PDF report generator with tamper-evident cryptographic hashes, actuarial disclaimers, and SHAP visual cards. | `GET /api/history/{id}/pdf` |
| **11** | **My Profile** | User account management, credential updating, role identification, and session activity tracking. | `GET /api/auth/me`, `PUT /api/auth/profile` |
| **12** | **Platform Settings** | Dark/Light theme toggles, notification preferences, data retention settings, and API configuration inspector. | Client-side preferences & `/api/config` |
| **13** | **Admin Overview** | Enterprise system overview: total users, database row counts, prediction throughput, audit events, and API uptime. | `GET /api/admin/system-stats` |
| **14** | **User Management** | Administrative role assignments (Admin, User, Actuary, Underwriter), account activation/deactivation, and user audit. | `GET /api/admin/users`, `PATCH /api/admin/users/{id}/role` |
| **15** | **Dataset Management** | Genuine Kaggle dataset metadata (1,338 records), SHA-256 validation checksum, missing value verification, and feature distribution summary. | `GET /api/admin/dataset` |
| **16** | **Model Registry** | MLOps Model Registry: inspect registered model versions, active champion selection, hyperparameters, and retraining status. | `GET /api/admin/models` |
| **17** | **System Monitoring** | Real-time health metrics, memory/disk status, database connection pool, API latency, and live audit event stream. | `GET /api/admin/audit-logs`, `GET /api/health` |

---

## 🚀 Quick Start Guide

### Option A: Using Bun (Recommended)
```powershell
# Start the full-stack server
bun run dev

# Run the 24 automated tests
bun run test
```

### Option B: Using Python
```powershell
# Start the server with hot-reload
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option C: Using Batch Script (1-Click)
Double-click `run_server.bat` in the project folder.

---

## 🌐 Application URLs

- **Live Application:** [http://localhost:8000/](http://localhost:8000/) (or `http://127.0.0.1:8000/`)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Redoc Docs:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Streamlit Dashboard:** [http://localhost:8501/](http://localhost:8501/)

---

## 🔑 Pre-Seeded Credentials

| Account | Email | Password | Quick Autofill Button |
|---|---|---|---|
| **Administrator** | `admin@health.ai` | `AdminPass123!` | Click **👑 Admin Demo** |
| **User Account** | `vishwaschinnu05@gmail.com` | `Vishwas123!` | Click **👤 User Demo** |

---

## 🧪 Automated Test Suite (24 Passed)

```powershell
python -m pytest tests -v
```
Verified 24/24 passing tests covering:
- Authentication, registration, auto-login, session hydration (`/api/auth/me`), and password reset tokens.
- Real machine learning predictions and batch CSV underwriting.
- Conformal prediction interval engine (90% coverage).
- Tree SHAP explainability waterfall decomposition.
- What-If scenario simulations with dollar delta calculations.
- Insurance plan comparisons and algorithmic recommendations.
- Conversational AI assistant with domain knowledge base.
- ReportLab binary PDF streaming and CSV audit export.
- Administrative user management and role updating.
