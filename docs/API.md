# API

Flask app, JSON over HTTP. Base URL `http://localhost:5000`. Served by Gunicorn in production mode.

## Startup

1. Load `config.yaml`.
2. Load the active model from `models/<version>/model.joblib` (version from `model_runs.is_active`).
3. Open DB connection pool.
4. Fail fast with a non-zero exit if the model or DB is unavailable.

## Endpoints

### `GET /health`
`200` `{"status":"ok","model_version":"v1.0.0","db":"ok"}`

### `GET /model/info`
`200` `{"model_version","algorithm","trained_at","metrics":{"accuracy","macro_f1"},"labels":["negative","neutral","positive"]}`

### `POST /predict`
Request:
```json
{"text": "The fabric feels cheap and it shrank after one wash."}
```
Response `200`:
```json
{
  "label": "negative",
  "confidence": 0.87,
  "probabilities": {"negative": 0.87, "neutral": 0.09, "positive": 0.04},
  "model_version": "v1.0.0"
}
```

### `POST /predict/batch`
Request: `{"texts": ["...", "..."]}`, 1–100 items.
Response `200`: `{"predictions": [ {same object as /predict}, ... ], "count": 2}`

### `GET /reviews`
Query: `label` (optional), `department` (optional), `limit` (default 50, max 500), `offset` (default 0).
Response `200`: `{"items":[{review fields + predicted_label}], "total": n, "limit": 50, "offset": 0}`

### `GET /stats`
`200` `{"total_reviews", "by_label":{"negative","neutral","positive"}, "by_department":[...]}`

## Validation

| Rule | Response |
|---|---|
| Body is not valid JSON | `400` |
| `text` missing, not a string, empty, or whitespace-only | `422` |
| `text` > 5000 characters | `422` |
| `texts` empty or > 100 items | `422` |
| Unknown route | `404` |
| Wrong HTTP method | `405` |
| Model/DB failure | `500` |

Error body: `{"error":{"code":"validation_error","message":"text must be a non-empty string"}}`

## Behaviour

- Every successful `/predict` and `/predict/batch` item is inserted into `predictions` with `source='api'`.
- Responses include header `X-Model-Version`.
- Logging: one structured line per request (method, path, status, duration ms), no review text in logs.
- CORS disabled by default.
- Request size limit 1 MB.

## Example

```bash
curl -X POST localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Love the fit, true to size."}'
```
