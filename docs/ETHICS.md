# Ethics and Responsible Use

## Principles

Transparency, privacy, fairness, human oversight.

## Requirements

| ID | Requirement |
|---|---|
| ETH-01 | No personally identifying fields stored beyond what the dataset provides; no names, emails, or order IDs |
| ETH-02 | Review text is not written to application logs |
| ETH-03 | Report per-class metrics, not accuracy alone, so minority-class failure is visible |
| ETH-04 | Evaluate performance by `Department Name` and by `Age` band (<30, 30–45, >45); document any group whose macro-F1 differs from overall by more than 10 points |
| ETH-05 | Document the label proxy: rating is a stand-in for sentiment, and ratings can disagree with text (sarcasm, mixed reviews) |
| ETH-06 | State the domain limit: trained on clothing reviews, performance on other product types is unverified |
| ETH-07 | `/model/info` exposes version and metrics so users can see model quality |
| ETH-08 | Predictions are advisory; the dashboard labels them "model-predicted", not "true sentiment" |
| ETH-09 | Dataset licence and source recorded in `DATA.md` and `README.md` |

## Known limitations to document in the final report

- Neutral (3-star) reviews are inherently ambiguous and expected to have the lowest F1
- English only; code-mixed or transliterated text will be misclassified
- Bag-of-words models cannot capture sarcasm or long-range negation
- Class imbalance can hide errors on negative reviews, which are the most business-critical
