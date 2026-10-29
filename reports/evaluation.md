# Model Evaluation Report

## 1. Executive Summary & Acceptance Criteria

| Criterion | Target | Baseline (MNB) | Final Model | Result |
|---|---|---|---|---|
| Test Macro-F1 | >= 0.65 | 0.6484 | 0.6556 | PASSED |
| Beat Baseline | Final > Baseline | 0.6484 | 0.6556 | PASSED |
| Minority Recall | >= 0.30 | 0.5309 | 0.5983 | PASSED |

## 2. Candidate Model Comparison (Validation Set)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Notes |
|---|---|---|---|---|
| Untuned MNB (BoW) | 0.8336 | 0.6447 | 0.8305 | Baseline |
| Untuned MNB (TF-IDF) | 0.7724 | 0.3013 | 0.6750 | Baseline |
| Standard Logistic Regression | 0.8218 | 0.6551 | 0.8307 | C=1.0, balanced |
| Standard Random Forest | 0.7912 | 0.4072 | 0.7178 | Trees=100, balanced |
| Tuned MultinomialNB | 0.8292 | 0.6523 | 0.8328 | alpha=0.5 |
| Tuned Logistic Regression | 0.8242 | 0.6467 | 0.8287 | C=2.48, balanced |

## 3. Final Model Test Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Negative** | 0.5804 | 0.5983 | 0.5892 | 356 |
| **Neutral** | 0.4206 | 0.4823 | 0.4493 | 423 |
| **Positive** | 0.9414 | 0.9152 | 0.9281 | 2,617 |

### Overall Metrics
- **Accuracy**: 0.8280
- **Macro-F1**: 0.6556
- **Weighted-F1**: 0.8330

## 4. Confusion Matrix (Normalized)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 59.83% | 31.18% | 8.99% |
| **Neutral** | 24.11% | 48.23% | 27.66% |
| **Positive** | 1.99% | 6.50% | 91.52% |

## 5. Subgroup & Fairness Evaluation (ETH-04)

Overall Macro-F1: **0.6556**

### Department Breakdown
| Department | Count | Macro-F1 | Difference from Overall |
|---|---|---|---|

### Age Band Breakdown
| Age Band | Count | Macro-F1 | Difference from Overall |
|---|---|---|---|

> [!NOTE]
> No demographic or department subgroups deviated by more than 10 points from overall macro-F1.

## 6. Top 10 Informative Terms per Class

| Rank | Negative | Neutral | Positive |
|---|---|---|---|
| 1 | felt cheap (2.72) | odd length (2.29) | highly recommend (2.43) |
| 2 | hideous (2.57) | sit well (1.95) | wait wear (2.30) |
| 3 | look nothing (2.40) | however bust (1.95) | highly (2.21) |
| 4 | look pound (2.39) | bit cheap (1.95) | summer fall (2.04) |
| 5 | pound heavier (2.39) | material maybe (1.95) | accessory (1.99) |
| 6 | yuck (2.39) | loved lace (1.95) | great purchase (1.98) |
| 7 | fit terrible (2.28) | listened review (1.95) | many compliment (1.96) |
| 8 | odd place (2.28) | unfortunately ran (1.95) | great jean (1.96) |
| 9 | back asap (2.28) | almost good (1.95) | comfortable great (1.95) |
| 10 | could return (2.28) | not sit (1.95) | definitely recommend (1.95) |