# Model Evaluation Report

## 1. Executive Summary & Acceptance Criteria

| Criterion | Target | Baseline (MNB) | Final Model | Result |
|---|---|---|---|---|
| Test Macro-F1 | >= 0.65 | 0.6537 | 0.6602 | PASSED |
| Beat Baseline | Final > Baseline | 0.6537 | 0.6602 | PASSED |
| Minority Recall | >= 0.30 | 0.5337 | 0.6039 | PASSED |

## 2. Candidate Model Comparison (Validation Set)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Notes |
|---|---|---|---|---|
| Untuned MNB (BoW) | 0.8363 | 0.6513 | 0.8344 | Baseline |
| Untuned MNB (TF-IDF) | 0.7721 | 0.2994 | 0.6744 | Baseline |
| Standard Logistic Regression | 0.8171 | 0.6424 | 0.8261 | C=1, class_weight=balanced |
| Standard Random Forest | 0.8283 | 0.5911 | 0.8061 | n_estimators=100, class_weight=balanced |
| Tuned MultinomialNB | 0.8313 | 0.6595 | 0.8356 | vectorizer=BoW, ngram_range=[1, 2], alpha=0.5, fit_prior=True |
| Tuned Logistic Regression | 0.8242 | 0.6437 | 0.8292 | C=2.48, solver=lbfgs |
| Tuned Random Forest | 0.8009 | 0.6020 | 0.8015 | n_estimators=150, max_depth=30, min_samples_leaf=1 |
| Tuned HistGradientBoosting | 0.8071 | 0.5345 | 0.7764 | learning_rate=0.1, max_leaf_nodes=31 |

## 3. Final Model Test Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Negative** | 0.5764 | 0.6039 | 0.5898 | 356 |
| **Neutral** | 0.4300 | 0.5012 | 0.4629 | 423 |
| **Positive** | 0.9439 | 0.9125 | 0.9279 | 2,617 |

### Overall Metrics
- **Accuracy**: 0.8289
- **Macro-F1**: 0.6602
- **Weighted-F1**: 0.8346

## 4. Confusion Matrix (Normalized)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 60.39% | 30.06% | 9.55% |
| **Neutral** | 24.35% | 50.12% | 25.53% |
| **Positive** | 2.10% | 6.65% | 91.25% |

## 5. Subgroup & Fairness Evaluation (ETH-04)

Overall Macro-F1: **0.6602**

### Department Breakdown
| Department | Count | Macro-F1 | Difference from Overall |
|---|---|---|---|
| Bottoms | 541 | 0.6009 | -0.0594 |
| Dresses | 936 | 0.6795 | +0.0193 |
| Intimate | 254 | 0.6896 | +0.0294 |
| Jackets | 125 | 0.6027 | -0.0575 |
| Tops | 1520 | 0.6639 | +0.0037 |

### Age Band Breakdown
| Age Band | Count | Macro-F1 | Difference from Overall |
|---|---|---|---|
| 30–45 | 1673 | 0.6733 | +0.0131 |
| <30 | 454 | 0.6737 | +0.0135 |
| >45 | 1269 | 0.6356 | -0.0246 |

> [!NOTE]
> No demographic or department subgroups deviated by more than 10 points from overall macro-F1.

## 6. Top 10 Informative Terms per Class

| Rank | Negative | Neutral | Positive |
|---|---|---|---|
| 1 | felt cheap (2.72) | odd length (2.29) | highly recommend (2.44) |
| 2 | hideous (2.57) | bit cheap (1.95) | wait wear (2.30) |
| 3 | look nothing (2.40) | listened review (1.95) | highly (2.22) |
| 4 | pound heavier (2.39) | sit well (1.95) | summer fall (2.04) |
| 5 | look pound (2.39) | however bust (1.95) | accessory (1.99) |
| 6 | yuck (2.39) | material maybe (1.95) | great purchase (1.98) |
| 7 | fit terrible (2.28) | almost good (1.95) | many compliment (1.96) |
| 8 | could return (2.28) | unfortunately ran (1.95) | great jean (1.96) |
| 9 | odd place (2.28) | loved lace (1.95) | comfortable great (1.96) |
| 10 | back asap (2.28) | coarse (1.92) | definitely recommend (1.96) |