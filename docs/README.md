# Customer Sentiment Analysis & Reporting for E-Commerce

[![CI](https://github.com/organization/ecommerce-sentiment/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Coverage >=70%](https://img.shields.io/badge/coverage-74%25-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

An end-to-end machine learning system classifying e-commerce customer reviews into **negative / neutral / positive**, persisting reviews and inferences in an ANSI-compliant SQLite database, exposing a low-latency REST API, and surfacing executive operational insights in a multi-page Power BI dashboard.

---

## 1. Quick Start (Fresh Clone to Running API in 4 Commands)

As specified in **NFR-09**, you can go from a fresh clone to a serving API in $\le 5$ commands:

```bash
# 1. Install dependencies and NLTK corpora
make setup

# 2. Ingest, clean, split, and persist raw data to SQLite
make data

# 3. Train models, tune hyperparameters, select best, and evaluate
make train

# 4. Start the production Flask REST API server
make serve
```

*(On Windows systems without `make`, use the PowerShell helper: `.\make.ps1 setup`, `.\make.ps1 data`, `.\make.ps1 train`, `.\make.ps1 serve`)*

---

## 2. System Architecture

```
              ┌────────────┐   ┌───────────┐   ┌─────────────┐
raw CSV  ───► │ Ingestion  │──►│  SQLite   │◄──│  Flask API  │◄── Clients / Webhooks
              └────────────┘   │ reviews   │   │  /predict   │
                    │          │predictions│  └──────▲──────┘
                    ▼          │model_runs │          │ loads active
              ┌────────────┐   │ views     │   ┌──────┴──────┐
              │ Training   │──►└─────┬─────┘   │ models/<v>/ │
              │ Pipeline   │─────────┼────────►│ model.joblib│ (2.77 MB)
              └────────────┘         │         └─────────────┘
                                     ▼
                               ┌─────────────┐
                               │ CSV Exports │──► Power BI Dashboard
                               └─────────────┘    (reports/sentiment_dashboard.pbix)
```

---

## 3. Technology Stack

- **Core & Runtime**: Python 3.11, NumPy, pandas, SciPy
- **Machine Learning**: scikit-learn, NLTK (tokenization, negation preservation, lemmatization), joblib
- **Database**: SQLite with SQLAlchemy ORM and repository pattern, raw ANSI-SQL views
- **API Server**: Flask 3.0, Gunicorn, RESTful JSON interfaces, sub-20ms latency
- **Visualization & BI**: Matplotlib, Seaborn, Power BI Desktop (`.pbix` + custom theme)
- **Code Quality & CI**: pytest, pytest-cov (74%+ test coverage), ruff, black, GitHub Actions

---

## 4. Key Performance Benchmarks

All models are trained with fixed random seed `42` under stratified 70/15/15 cross-validation:

| Model Candidate | Feature Extraction | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Status |
|---|---|---|---|---|---|
| **Untuned MultinomialNB (Baseline)** | Bag-of-Words | 0.6472 | 0.6484 | 0.8248 | Benchmark |
| **Tuned MultinomialNB ($\alpha=0.5$)** | **Bag-of-Words** | **0.6523** | **0.6556** | **0.8280** | **ACTIVE (v1.0.0)** |
| Tuned Logistic Regression ($C=1.0$) | TF-IDF | 0.6515 | 0.6538 | 0.8265 | Candidate |
| Tuned Random Forest (150 trees) | TF-IDF | 0.5480 | 0.5420 | 0.8010 | Candidate |
| HistGradientBoosting (L2 reg=1.0) | TF-IDF (SVD 100) | 0.5960 | 0.5910 | 0.8075 | Candidate |

- **Macro-F1 Acceptance Gate (>= 0.65)**: Passed (0.6556).
- **Beats Baseline**: Passed (+0.0072 F1 improvement over baseline).
- **Inference Latency (p95 < 200 ms)**: Passed (p50 ~ 10 ms, p95 ~ 25 ms locally).
- **Model Artifact Size (< 100 MB)**: Passed (2.77 MB).
- **Class Probabilities Sum**: $1.000000 \pm 10^{-6}$.

---

## 5. REST API Endpoints

The API runs by default on `http://localhost:5000`:

### `POST /predict`
Score a single customer review. Every prediction is persisted to SQL (`source='api'`).
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The dress fits beautifully and the silk fabric is breathtaking!"}'
```
**Response (HTTP 200)**:
```json
{
  "predicted_label": "positive",
  "confidence": 0.9998,
  "probabilities": {
    "negative": 0.00001,
    "neutral": 0.00019,
    "positive": 0.99980
  },
  "model_version": "v1.0.0"
}
```

### `POST /predict/batch`
Score up to 100 customer reviews in a single request.
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Love it!", "Runs slightly small.", "Horrible seams ripped."]}'
```

### Other Endpoints
- `GET /health`: System status, active model version, and database connectivity.
- `GET /model/info`: Active model metadata, hyperparameter configuration, training timestamp, and metrics.
- `GET /reviews?limit=20&offset=0&label=negative`: Paginated database inspection of stored reviews.
- `GET /stats`: Real-time aggregated statistics from SQL views.

---

## 6. Automation & Command Reference

The project includes one-command automation via `Makefile` (and `make.ps1` for Windows):

| Target | Description |
|---|---|
| `make setup` | Install pip dependencies and NLTK corpora |
| `make data` | Load raw CSV, clean, generate data quality report, split, and ingest to SQLite |
| `make eda` | Generate standardized EDA visual figures in `reports/figures/` |
| `make train` | Run candidate comparison, hyperparameter tuning, evaluation, and serialization |
| `make evaluate` | Compute evaluation metrics for the active model |
| `make promote VERSION=v1.0.0` | Promote a registered model version to active |
| `make batch-score` | Score all stored reviews in SQLite and persist predictions (`source='batch'`) |
| `make export` | Export all 7 SQL views to `reports/exports/*.csv` for Power BI |
| `make serve` | Launch the Flask REST API server |
| `make test` | Execute pytest test suite with coverage gate ($\ge 70\%$) |
| `make lint` | Run ruff and black lint/formatting checks |
| `make format` | Automatically format all code using black and ruff |
| `make all` | Run complete end-to-end pipeline: data $\rightarrow$ eda $\rightarrow$ train $\rightarrow$ batch-score $\rightarrow$ export |

---

## 7. Power BI Dashboard (`reports/sentiment_dashboard.pbix`)

Four core reporting pages designed in accordance with `DASHBOARD.md`:
1. **Executive Overview**: Total Reviews KPI, Sentiment Share Donut, Department Distribution, and Synced Slicers.
2. **Category Drill-Down**: 100% Stacked Bar across product classes, Department $\times$ Class matrix with conditional formatting, and Scatter plot (Avg Rating vs. % Negative).
3. **Text Insights**: Top 10 high-frequency terms per sentiment, Rating (1–5) vs. Predicted Label confusion heatmap, and sortable lowest-confidence review inspector.
4. **Model Monitoring**: Active version telemetry, validation and test F1 history, API traffic volume by day, and average confidence trendline.

*One-click refresh*: Running `make export` updates the CSV files in `reports/exports/`, refreshing all Power BI visuals without modifying queries.

---

## 8. Ethics, Privacy & Fairness

- **No Review Text in API Logs (ETH-01)**: In accordance with `ETHICS.md`, request logging records timestamp, status code, latency, and predicted label, but **never** customer review text or customer identifiers.
- **Demographic Bias Auditing (ETH-03)**: Evaluation splits confirm consistent Macro-F1 across product departments and customer age brackets (<30, 30–45, >45).
- **Proxy Label Awareness (ETH-04)**: Star ratings serve as proxy ground truth; documented neutral review ambiguity is explicitly addressed in model confidence distributions.

---

## 9. Document Index

All eleven specification and requirements documents are maintained in root and mirrored in `docs/`:

| File | Purpose |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Functional and non-functional requirements, scope, acceptance criteria |
| [DATA.md](DATA.md) | Raw dataset schema, cleaning rules, label derivation, and split strategy |
| [ML_PIPELINE.md](ML_PIPELINE.md) | Preprocessing, feature extraction, candidate models, tuning, and evaluation |
| [DATABASE.md](DATABASE.md) | SQLite schema, relational constraints, indexing, and reporting views |
| [API.md](API.md) | REST API contract, request/response schemas, error handling, and latency |
| [DASHBOARD.md](DASHBOARD.md) | Power BI page specifications, DAX measures, theme, and acceptance rules |
| [MLOPS.md](MLOPS.md) | Semantic versioning, metadata tracking, reproducibility, and automation |
| [ETHICS.md](ETHICS.md) | Privacy safeguards, logging constraints, fairness auditing, and risk boundaries |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System components, end-to-end data flow, and repository layout |
| [TESTING.md](TESTING.md) | Comprehensive test plan, quality gates, and definition of done |
| [ROADMAP.md](ROADMAP.md) | Modular implementation breakdown and phased delivery milestones |

---

## 10. Deliverables Manifest

- **Source Code**: Fully modular code in `src/` (`data/`, `features/`, `models/`, `db/`, `api/`, `cli.py`).
- **Trained Model Artifact**: `models/v1.0.0/model.joblib` (2.77 MB) + `metadata.json` + `metrics.json`.
- **Database File**: `data/sentiment.db` (populated with 22,640 reviews, model run registry, and 22,640 predictions).
- **Power BI Dashboard**: `reports/sentiment_dashboard.pbix` + `reports/theme.json` + `reports/powerbi_guide.md`.
- **Reports & Artifacts**: `reports/evaluation.md`, `reports/data_quality.md`, `reports/themes.md`, `reports/errors.csv`, and EDA charts in `reports/figures/`.
- **Automated CI & Tests**: `.github/workflows/ci.yml`, `Makefile`, `make.ps1`, `tests/` ($\ge 74\%$ coverage).
