# ML Pipeline

## Stages

1. **Load and validate** (`src/data/load.py`)
2. **Clean and label** (`src/data/clean.py`)
3. **Split** (stratified 70/15/15, seed 42)
4. **Preprocess text** (`src/features/text.py`)
5. **Vectorise** (BoW or TF-IDF)
6. **Train candidates**
7. **Tune** (grid + randomised search)
8. **Evaluate** on validation, then once on test
9. **Persist** model + metadata + metrics

## Text preprocessing

Implemented as a scikit-learn-compatible transformer so it lives inside the `Pipeline`.

- Lowercase; remove URLs, HTML tags, digits, punctuation
- Tokenise (NLTK)
- Stop words: NLTK English list **minus** negations (`no`, `not`, `nor`, `never`, `n't`, `against`)
- Lemmatise (WordNet)
- Drop tokens shorter than 2 characters

## Features

| Variant | Settings |
|---|---|
| BoW | `CountVectorizer(ngram_range=(1,2), min_df=3, max_features=50000)` |
| TF-IDF | `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=50000)` |

Vectoriser is fitted on training data only.

## Candidate models

| Model | Notes |
|---|---|
| MultinomialNB | Baseline (Module 10) |
| LogisticRegression | `class_weight="balanced"`, `max_iter=1000` |
| RandomForestClassifier | `class_weight="balanced"`, `n_jobs=-1` |
| GradientBoostingClassifier | On TruncatedSVD-reduced TF-IDF (300 components) for speed |

## Hyperparameter search

Stratified 5-fold CV, scoring `f1_macro`, `random_state=42`.

| Model | Grid / distribution |
|---|---|
| MultinomialNB | `alpha` ∈ {0.01, 0.1, 0.5, 1.0}; vectoriser ∈ {BoW, TF-IDF}; `ngram_range` ∈ {(1,1),(1,2)} |
| LogisticRegression | `C` ∈ loguniform(1e-2, 1e2) |
| RandomForest | `n_estimators` ∈ [100, 500], `max_depth` ∈ [None, 10, 20, 40], `min_samples_leaf` ∈ [1, 2, 5] |
| GradientBoosting | `n_estimators` ∈ [100, 300], `learning_rate` ∈ [0.03, 0.3], `max_depth` ∈ [2, 5] |

Use `GridSearchCV` for MultinomialNB (small space) and `RandomizedSearchCV` (`n_iter=30`) for the others.

## Evaluation

Reported on validation (selection) and test (final):

- Accuracy, macro-F1, weighted-F1
- Per-class precision, recall, F1
- Confusion matrix (counts and row-normalised)
- Cross-validation mean ± std
- 10 most informative terms per class (NB log-probabilities or LR coefficients)
- Error analysis: 20 misclassified reviews per class, saved to `reports/errors.csv`

**Selection rule:** highest validation macro-F1; ties broken by smaller artefact size.

## Optional: theme discovery (Module 8)

K-Means on TF-IDF of negative reviews, k ∈ 3–10 chosen by silhouette score and inspection. Output: cluster id, top terms, example reviews, to `reports/themes.md`. Not part of the served model.

## Outputs

```
models/<version>/
  model.joblib
  metadata.json
  metrics.json
  confusion_matrix.png
reports/evaluation.md
```
