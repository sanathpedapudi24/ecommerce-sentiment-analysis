# Testing

Framework: `pytest`, `pytest-cov`. Run with `make test`. Coverage gate ≥ 70% on `src/`.

## Unit tests

| File | Cases |
|---|---|
| `test_clean.py` | Drops null/empty text; removes duplicates; rejects rating outside 1–5; label mapping 1,2→negative, 3→neutral, 4,5→positive |
| `test_text.py` | Lowercasing; URL/HTML/digit removal; negations preserved (`"not good"` keeps `not`); lemmatisation; empty string safe |
| `test_pipeline.py` | Pipeline fits and predicts on a toy set; output labels in allowed set; probabilities sum to 1; same seed gives identical predictions; no vectoriser fit on test data |
| `test_db.py` | Schema creates; CHECK constraints reject bad rating/label; ingestion is idempotent; only one active model; views return expected columns |

## API tests (Flask test client)

| Case | Expect |
|---|---|
| `GET /health` | 200, status ok |
| `POST /predict` valid | 200, label in set, probabilities sum to 1 |
| `POST /predict` empty text | 422 |
| `POST /predict` invalid JSON | 400 |
| `POST /predict` > 5000 chars | 422 |
| `POST /predict/batch` 100 items | 200, count 100 |
| `POST /predict/batch` 101 items | 422 |
| Prediction persisted | Row exists in `predictions` with `source='api'` |
| `GET /reviews?limit=1000` | limit capped at 500 |
| `GET /model/info` | version matches active `model_runs` row |

## Model quality gates (run by `make evaluate`)

- Test macro-F1 ≥ 0.65
- Tuned model macro-F1 > baseline Naive Bayes macro-F1
- No class with recall < 0.30

## Performance check

100 sequential `/predict` calls locally: p95 < 200 ms.

## Definition of done

- [ ] All functional requirements FR-01 to FR-20 implemented
- [ ] Acceptance criteria 1–7 in `REQUIREMENTS.md` met
- [ ] `make all` runs end to end from a clean clone
- [ ] CI green
- [ ] Dashboard matches database totals
- [ ] `reports/evaluation.md` includes baseline vs final comparison and error analysis
- [ ] Ethics limitations documented
