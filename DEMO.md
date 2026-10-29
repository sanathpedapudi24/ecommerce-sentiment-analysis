# Demo Script: Customer Sentiment Analysis & Reporting for E-commerce

A 15-minute walkthrough of the capstone, from raw reviews to a live prediction UI and dashboard. Each step lists what to run, what to show, what to say, and which course module it demonstrates (full map in [README §9](README.md#9-course-syllabus-coverage)).

Numbers quoted below come from the active **v1.1.0** model and reports (trained on all 22,640 reviews). If you retrain, read the fresh values off `reports/` on the day.

---

## Before the demo (do this the day before)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and save the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) CSV as `data/raw/reviews.csv`.
2. Build the environment and run the whole pipeline once. The full run took about 4½ minutes on the development laptop, so never do it live:

   ```bash
   make setup
   ```
   ```bash
   make all
   ```

   On Windows without `make`, use `.\make.ps1 setup` and `.\make.ps1 all`.
3. Check that the run ended with `=== All Pipeline Stages Completed Successfully! ===` and that `reports/evaluation.md` §1 shows all three acceptance criteria as PASSED.
4. Open these in tabs so you are not hunting for files live:
   - `reports/data_quality.md`, `reports/evaluation.md`, `reports/themes.md`
   - `reports/figures/label_distribution.png`, `models/v1.1.0/confusion_matrix.png`
   - `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb` (kernel: the project's `.venv`)
5. Run the tests once so you can show a green result: `make test`.

**No dataset?** Step 7 (API and UI) still works with the committed model. `/model/info` and `/stats` will show empty values until `make all` has run.

---

## 1. The problem (1 min)

**Say:** An online clothing store gets thousands of reviews. Star ratings say *how* happy customers are but not *why*. This project classifies review text as negative / neutral / positive, stores every prediction in SQL, serves a live API, and feeds a Power BI dashboard.

**Show:** the architecture diagram in [README §2](README.md#2-system-architecture).

---

## 2. Data and cleaning (2 min) — *Modules 1–2: NumPy, Pandas*

**Show:** `reports/data_quality.md`.

**Say:**
- 23,486 raw reviews → **22,640** after dropping 845 with empty text and 1 exact duplicate.
- Labels come from the rating: 1–2 negative, 3 neutral, 4–5 positive.
- The data is heavily imbalanced: **77.1% positive**, 12.5% neutral, 10.5% negative. A model that always says "positive" would score 77% accuracy, which is why we judge models on **macro-F1** instead.
- The split is stratified 70/15/15 (train/validation/test). The test set is touched exactly once.

---

## 3. Exploratory analysis (2 min) — *Module 2: Matplotlib & Seaborn*

**Show:** `notebooks/01_eda.ipynb`, run top to bottom.

**Point out:**
- §2b: NumPy vectorised statistics on review length vs rating.
- The four figures: label distribution, text length by sentiment, rating by department, and top terms per class.

---

## 4. Model pipeline and selection (3 min) — *Modules 3–5 and 7: scikit-learn, NLP, Naive Bayes, ensembles, tuning*

**Show:** `notebooks/02_modeling.ipynb` §1b (Bag-of-Words vs TF-IDF with Naive Bayes), then `reports/evaluation.md` §2.

**Say:**
- Preprocessing is one scikit-learn transformer: lowercase, strip URLs/HTML/digits, tokenise, remove stop words **but keep negations** ("didn't like" becomes "did not like" rather than just "like"), lemmatise.
- Candidates: Naive Bayes baselines, Logistic Regression, Random Forest, HistGradientBoosting. They're tuned with `GridSearchCV` / `RandomizedSearchCV` and stratified k-fold CV.
- Every candidate competes on **validation** macro-F1. Only the winner is scored on test.
- The v1.1.0 winner is a **tuned Multinomial Naive Bayes** (BoW 1–2 grams, α = 0.5): test **macro-F1 0.6602**, accuracy 0.8289. That beats the untuned baseline (0.6537) and clears the 0.65 acceptance gate. Logistic Regression, Random Forest and HistGradientBoosting all scored lower on validation.
- Versioning story: v1.0.0 scored 0.6556. Fixing negation handling (*didn't*, *dont*) produced v1.1.0. The same seed and split make the two runs directly comparable.

**Show:** `models/v1.1.0/confusion_matrix.png`.

**Say (be honest about weaknesses):**
- Positive is easy: F1 0.93, recall 91%.
- Neutral is hard: F1 0.46, recall 50%. A 3-star review often mixes praise and complaints.
- 30% of negatives are predicted neutral, but only 10% are predicted positive. Mistakes mostly land on the *neighbouring* class.

---

## 5. What customers complain about (1 min) — *Module 4: Clustering*

**Show:** `reports/themes.md`.

**Say:** K-Means on TF-IDF vectors of negative reviews, with k chosen by silhouette score (k = 5), surfaces themes like **sizing** ("size, small, runs, large"), **fabric quality** ("fabric, quality, material, cheap"), and boxy or itchy fits. The silhouette score is low (0.011), which is typical for short text. Treat the themes as a starting point for analysts, not ground truth.

Also worth showing: `reports/evaluation.md` §6, the most informative terms per class ("felt cheap", "hideous", "highly recommend", "great purchase").

---

## 6. SQL layer (1 min) — *Module 6: SQL, Python + SQL*

**Show:** `src/db/schema.sql`: four tables with CHECK constraints, plus seven reporting views.

**Run** a view query live:

```bash
uv run python -c "import pandas as pd, sqlite3; print(pd.read_sql('SELECT * FROM vw_sentiment_by_department ORDER BY review_count DESC LIMIT 8', sqlite3.connect('data/sentiment.db')))"
```

**Say:** Ingestion is idempotent (re-running `make data` inserts nothing new). Every batch and API prediction is stored with its model version, which makes the dashboard auditable.

---

## 7. Live API and UI (3 min) — *Module 7: Flask deployment*

**Run:**

```bash
make serve
```

**Show:** open **http://localhost:5000** in a browser.

The example buttons are real reviews from the held-out test set:

1. Click **Positive (5★)**: "Great, versatile piece. A great addition to any wardrobe…" → **positive, ~100%**.
2. Click **Neutral (3★)**: "Pretty but not for tall ladies. Loved the style and color but this dress was simply too short…" → **neutral, ~94%**. The model reads the "loved X **but** Y" pivot.
3. Click **Negative (1★)**: "Just a ginormous, shapeless shirt. Runs extremely large…" → **negative, ~100%**.
4. Click **Tricky neutral (3★)**: "Love the style! This top runs large… I love the style." → the model says **positive (~88%)**, a real miss. The review opens and closes with praise, so the mild complaint in the middle gets outweighed. This is the neutral-class weakness from step 4, live; use it rather than hiding it.
5. Click **Batch: one of each**: the first three reviews scored in one request, one result per line.

**Optional, for a technical audience:** the same API from the terminal. In Windows PowerShell, use `curl.exe`, because plain `curl` there is a different command.

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"Love it, wearing it every weekend!\"}"
```
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"   \"}"
```

The second call shows input validation: **HTTP 422** with a JSON error body.

**Say:** Local latency was p50 ≈ 24 ms and p95 ≈ 134 ms, inside the 200 ms target. Request logs record timing and status but **never review text** (see `ETHICS.md`).

---

## 8. Dashboard (1 min) — *Module 7: Data storytelling & Power BI*

**Run:** `make export`, which writes the seven views to `reports/exports/*.csv`.

**Show:** the dashboard built from [`reports/powerbi_guide.md`](reports/powerbi_guide.md), which has four pages: Executive Overview, Category Drill-Down, Text Insights, and Model Monitoring.

> The committed `reports/sentiment_dashboard.pbix` is a layout scaffold with no data model. Build the real dashboard in Power BI Desktop from the guide before the demo, and save it over that file.

---

## 9. MLOps and engineering practice (1 min) — *Module 8: Pipelines & MLOps*

**Show and say:**
- **One artefact:** preprocessing, vectoriser, and classifier are a single `Pipeline` in `models/v1.1.0/model.joblib`, so training and serving can't drift apart.
- **Versioning:** `models/v1.1.0/metadata.json` records the data hash, seed, scikit-learn version, and metrics. A preprocessing change meant a minor bump (v1.0.0 → v1.1.0). `make promote VERSION=<v>` switches the active model among versions registered in `model_runs`.
- **Reproducible environment:** `pyproject.toml` + `uv.lock` pin every package, and `.python-version` pins Python 3.11.
- **Quality gates:** `make test` runs 46 tests with a 70% coverage gate; ruff handles linting; GitHub Actions runs it all on every push.

---

## Likely questions

| Question | Answer |
|---|---|
| Why not just use the star rating? | New text arrives without a rating (support tickets, social posts), and the text explains *why* a customer is unhappy. |
| Why is neutral so weak? | 3-star reviews mix positive and negative statements, and there are fewer of them (12.5%). Ratings are also a proxy label: some 3-star texts read as clearly positive. |
| Why does a short complaint like "I dont like the phone" come out positive (~57%)? | Naive Bayes starts from the training prior (77% positive). Real reviews have a median of 62 words, enough to overcome it; five words aren't, and only 0.6% of training reviews are that short. "phone" also barely appears in clothing reviews. Tuning tried a uniform prior (`fit_prior=False`), but the learned prior scored better on real reviews. On real test reviews under 15 words the model holds up (macro-F1 0.62, negative recall 0.75). A clothing-only model will still be weak on other product types. |
| Why did Naive Bayes beat Random Forest? | Sparse, high-dimensional text suits linear and probabilistic models. Trees struggle with tens of thousands of n-gram features. |
| Is the model fair across groups? | `reports/evaluation.md` §5 compares macro-F1 by department and age band, and flags any group more than 10 points from overall. |
| What would you do next? | More neutral examples or a sentence-embedding model, aspect-level sentiment (fit vs fabric vs price), and drift monitoring on API traffic. |
