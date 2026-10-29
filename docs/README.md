# Customer Sentiment Analysis & Reporting for E-Commerce

[![CI](https://github.com/organization/ecommerce-sentiment/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Coverage >=70%](https://img.shields.io/badge/coverage-73%25-brightgreen.svg)](tests/)
[![Env: uv](https://img.shields.io/badge/env-uv-de5fe9.svg)](https://docs.astral.sh/uv/)
[![Lint & format: ruff](https://img.shields.io/badge/lint%20%26%20format-ruff-orange.svg)](https://github.com/astral-sh/ruff)

An end-to-end machine learning system classifying e-commerce customer reviews into **negative / neutral / positive**, persisting reviews and inferences in an ANSI-compliant SQLite database, exposing a low-latency REST API, and surfacing executive operational insights in a multi-page Power BI dashboard.

---

## 1. Quick Start (Fresh Clone to Running API in 4 Commands)

As specified in **NFR-09**, you can go from a fresh clone to a serving API in $\le 5$ commands.

**Prerequisites**: [uv](https://docs.astral.sh/uv/getting-started/installation/) (it downloads Python 3.11 itself, pinned in `.python-version`) and the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) CSV saved as `data/raw/reviews.csv`.

```bash
# 1. Create .venv from pyproject.toml + uv.lock and download NLTK corpora
make setup

# 2. Ingest, clean, split, and persist raw data to SQLite
make data

# 3. Train models, tune hyperparameters, select best, and evaluate
make train

# 4. Start the production Flask REST API server
make serve
```

*(On Windows systems without `make`, use the PowerShell helper: `.\make.ps1 setup`, `.\make.ps1 data`, `.\make.ps1 train`, `.\make.ps1 serve`)*

Without either helper, the same steps are plain uv commands:

```bash
uv sync                                                   # create .venv (dev tools included)
uv run python -m nltk.downloader stopwords wordnet punkt punkt_tab
uv run python -m src.cli all                              # data -> eda -> train -> batch-score -> export
uv run python -m src.cli serve
uv run pytest                                             # tests + coverage
```

Add a dependency with `uv add <package>` (or `uv add --dev <package>`); never edit `uv.lock` by hand. For the notebooks, select the `.venv` interpreter as the Jupyter kernel.

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
- **Environment**: uv (`pyproject.toml` + `uv.lock`, Python pinned in `.python-version`)
- **Code Quality & CI**: pytest, pytest-cov (73%+ test coverage), ruff (lint + format), GitHub Actions

---

## 4. Key Performance Benchmarks

All models use fixed random seed `42` and a stratified 70/15/15 train/validation/test split; tuning uses stratified k-fold CV on the train split. Every non-baseline model (standard and tuned, including the Random Forest and HistGradientBoosting ensembles) competes on **validation** macro-F1, and only the winner is scored on the test set. `make train` regenerates the full table in [`reports/evaluation.md`](reports/evaluation.md); the numbers below are from the committed v1.0.0 run:

| Model | Feature Extraction | Val Macro-F1 | Val Accuracy | Status |
|---|---|---|---|---|
| Untuned MultinomialNB (Baseline) | Bag-of-Words | 0.6447 | 0.8336 | Benchmark (test macro-F1 0.6484) |
| Untuned MultinomialNB | TF-IDF | 0.3013 | 0.7724 | Benchmark |
| Standard Logistic Regression ($C=1.0$) | TF-IDF | 0.6551 | 0.8218 | Candidate |
| Standard Random Forest (100 trees) | TF-IDF | 0.4072 | 0.7912 | Candidate |
| **Tuned MultinomialNB ($\alpha=0.5$)** | **Bag-of-Words** | **0.6523** | **0.8292** | **ACTIVE v1.0.0: test macro-F1 0.6556, test accuracy 0.8280** |
| Tuned Logistic Regression ($C=2.48$) | TF-IDF | 0.6467 | 0.8242 | Candidate |

> The committed v1.0.0 run predates two fixes: the Standard Logistic Regression and the tuned Random Forest / HistGradientBoosting models were not allowed to compete, and contractions such as *didn't* lost their negation in preprocessing. Re-run `make train` to refresh the model and this table.

- **Macro-F1 Acceptance Gate (>= 0.65)**: Passed (0.6556).
- **Beats Baseline**: Passed (+0.0072 F1 improvement over baseline).
- **Inference Latency (p95 < 200 ms)**: Passed (p50 ~ 10 ms, p95 ~ 25 ms locally).
- **Model Artifact Size (< 100 MB)**: Passed (2.77 MB).
- **Class Probabilities Sum**: $1.000000 \pm 10^{-6}$.

---

## 5. REST API Endpoints

The API runs by default on `http://localhost:5000`. Open that address in a browser for a one-page UI ([`src/api/static/index.html`](src/api/static/index.html)) that scores one review, or several (one per line) as a batch, through the endpoints below.

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
  "label": "positive",
  "confidence": 0.9998,
  "probabilities": {
    "negative": 0.0,
    "neutral": 0.0002,
    "positive": 0.9998
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
| `make setup` | `uv sync` the environment and download NLTK corpora |
| `make data` | Load raw CSV, clean, generate data quality report, split, and ingest to SQLite |
| `make eda` | Generate standardized EDA visual figures in `reports/figures/` |
| `make train` | Run candidate comparison, hyperparameter tuning, evaluation, and serialization |
| `make evaluate` | Compute evaluation metrics for the active model |
| `make promote VERSION=v1.0.0` | Promote a registered model version to active |
| `make batch-score` | Score all stored reviews in SQLite and persist predictions (`source='batch'`) |
| `make export` | Export all 7 SQL views to `reports/exports/*.csv` for Power BI |
| `make serve` | Launch the Flask REST API server |
| `make test` | Execute pytest test suite with coverage gate ($\ge 70\%$) |
| `make lint` | Run ruff lint and format checks |
| `make format` | Automatically format and fix all code with ruff |
| `make all` | Run complete end-to-end pipeline: data $\rightarrow$ eda $\rightarrow$ train $\rightarrow$ batch-score $\rightarrow$ export |

---

## 7. Power BI Dashboard (`reports/sentiment_dashboard.pbix`)

Four core reporting pages designed in accordance with `DASHBOARD.md`:
1. **Executive Overview**: Total Reviews KPI, Sentiment Share Donut, Department Distribution, and Synced Slicers.
2. **Category Drill-Down**: 100% Stacked Bar across product classes, Department $\times$ Class matrix with conditional formatting, and Scatter plot (Avg Rating vs. % Negative).
3. **Text Insights**: Top 10 high-frequency terms per sentiment, Rating (1–5) vs. Predicted Label confusion heatmap, and sortable lowest-confidence review inspector.
4. **Model Monitoring**: Active version telemetry, validation and test F1 history, API traffic volume by day, and average confidence trendline.

*One-click refresh*: Running `make export` updates the CSV files in `reports/exports/`, refreshing all Power BI visuals without modifying queries.

> **Note**: the committed `.pbix` holds only a page layout and theme. It has no data model, so Power BI Desktop is unlikely to open it. Build the dashboard in Power BI Desktop by following [`reports/powerbi_guide.md`](reports/powerbi_guide.md) (CSV imports, DAX measures, theme), then save over this file.

---

## 8. Ethics, Privacy & Fairness

- **No Review Text in API Logs (ETH-01)**: In accordance with `ETHICS.md`, request logging records timestamp, status code, latency, and predicted label, but **never** customer review text or customer identifiers.
- **Demographic Bias Auditing (ETH-03)**: Evaluation splits confirm consistent Macro-F1 across product departments and customer age brackets (<30, 30–45, >45).
- **Proxy Label Awareness (ETH-04)**: Star ratings serve as proxy ground truth; documented neutral review ambiguity is explicitly addressed in model confidence distributions.

---

## 9. Course Syllabus Coverage

Each course module maps to code you can run in this repository (week-by-week plan in [ROADMAP.md](ROADMAP.md)):

| # | Module | Where it is covered |
|---|---|---|
| 1 | Introduction to ML & the Data Science Ecosystem; Core Python Libraries: **NumPy** | Project layout, uv environment (`pyproject.toml`, `uv.lock`), `config.yaml` → `src/config.py`; NumPy index arrays and `np.isclose` in `src/data/split.py`, `argsort`/`np.max` over probability matrices in `src/models/evaluate.py` and `src/models/predict.py`; vectorised statistics in `notebooks/01_eda.ipynb` §2b |
| 2 | Core Python Libraries: **Pandas**; Data Visualization with **Matplotlib & Seaborn** | `src/data/load.py`, `src/data/clean.py` (null / duplicate / rating filters → `reports/data_quality.md`); `src/data/eda.py` → `reports/figures/*.png`; `notebooks/01_eda.ipynb` |
| 3 | Introduction to **Scikit-learn**, Supervised Learning, Feature Engineering & Preprocessing | Custom `TextPreprocessor` (`BaseEstimator` + `TransformerMixin`) in `src/features/text.py`; labels derived from ratings in `src/data/clean.py`; stratified 70/15/15 split in `src/data/split.py`; Logistic Regression and Naive Bayes classifiers in `src/models/train.py` |
| 4 | **Model Evaluation & Selection**; Unsupervised Learning: **Clustering** | `src/models/evaluate.py`: accuracy, macro/weighted F1, per-class precision/recall/F1, normalized confusion matrix, error analysis, subgroup fairness; selection on validation macro-F1 in `src/models/train.py`; K-Means + silhouette score on negative reviews → `reports/themes.md` |
| 5 | **Text Data Handling & Basic NLP**; Text Classification with **Naive Bayes** | NLTK tokenisation, stop-word removal that keeps negations, WordNet lemmatisation; Bag-of-Words vs TF-IDF with n-grams; MultinomialNB baseline and tuned model; comparison in `notebooks/02_modeling.ipynb` §1b |
| 6 | Introduction to **SQL** for Data Science; Integrating **Python with SQL** | `src/db/schema.sql` (tables, CHECK constraints, indexes, 7 reporting views), `src/db/migrations/`; `src/db/repository.py` (SQLAlchemy, parameterised queries, idempotent ingestion, prediction storage) |
| 7 | Model Deployment with **Flask**; Data Storytelling & Dashboards with **Power BI**; Advanced Scikit-learn: **Ensemble Methods, Hyperparameter Tuning & Model Optimization** | `src/api/app.py` + `src/api/schemas.py` (6 endpoints, validation, every prediction persisted); `reports/evaluation.md`, `reports/themes.md`, `reports/exports/*.csv`, `reports/powerbi_guide.md`, `DASHBOARD.md`; Random Forest and HistGradientBoosting (on TruncatedSVD) tuned with `GridSearchCV` / `RandomizedSearchCV` + `StratifiedKFold` in `src/models/tune.py`, `class_weight="balanced"` for imbalance |
| 8 | **ML Pipelines** & Best Practices; Introduction to **MLOps** Concepts | One `Pipeline` artefact (preprocess → vectorise → classify) to avoid train/serve skew; versioned `models/<version>/` with `metadata.json` (data hash, seed, sklearn version); `model_runs` registry with `promote`; `Makefile` / `make.ps1`, locked uv environment, ruff, pytest coverage gate, GitHub Actions CI; `ETHICS.md` |
| **Capstone** | **Customer Sentiment Analysis & Reporting for E-commerce** | This repository end to end: `make all` (ingest → EDA → train / tune / select → batch-score → export), then `make serve` for the API and the Power BI dashboard on the exported views |

---

## 10. Document Index

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

## 11. Deliverables Manifest

- **Source Code**: Fully modular code in `src/` (`data/`, `features/`, `models/`, `db/`, `api/`, `cli.py`).
- **Trained Model Artifact**: `models/v1.0.0/model.joblib` (2.77 MB) + `metadata.json` + `metrics.json`.
- **Database File**: `data/sentiment.db` (populated with 22,640 reviews, model run registry, and 22,640 predictions).
- **Power BI Dashboard**: `reports/sentiment_dashboard.pbix` + `reports/theme.json` + `reports/powerbi_guide.md`.
- **Reports & Artifacts**: `reports/evaluation.md`, `reports/data_quality.md`, `reports/themes.md`, `reports/errors.csv`, and EDA charts in `reports/figures/`.
- **Automated CI & Tests**: `.github/workflows/ci.yml`, `Makefile`, `make.ps1`, `tests/` ($\ge 70\%$ coverage gate, currently 73%).
