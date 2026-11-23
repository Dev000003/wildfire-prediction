# Integration test report

Executed on Linux, Python 3.12.14, Streamlit 1.50.0 and scikit-learn 1.8.0. All seven unittest cases passed. All eight direct dependency pins matched the imported runtime packages.

- Model loading verified the fixed local file's SHA-256 before deserialization, fitted pipeline steps, four ordered feature names, numeric preprocessing output, and class labels. Missing and modified models were rejected before deserialization.
- Prediction validation exercised missing, invalid, boolean, nonfinite, fractional integer-field and out-of-range values. All observed bounds matched the workbook. Both range extremes were also exercised in valid combinations.
- Probability checks compared with the fitted model directly, tested reversed class ordering, and rejected a malformed probability sum.
- Streamlit AppTest navigated all four pages, checked the overview counts, validated the three-row comparison table against its CSV, and found 1 overview chart, 7 EDA charts and 5 performance charts. The 13 underlying PNGs decoded successfully.
- The prediction form submitted four observed dataset rows, displayed both probabilities and progress bars, and cleared a prior result when a new submission had a missing field.
- Live Streamlit startup was checked separately using its HTTP homepage and health endpoint. Widget interactions were executed using Streamlit AppTest. Mac browser rendering was not tested.

## Actual form submissions

| Temperature | Wind | Rain | Relative humidity | Observed label | Prediction | Fire | No Fire |
|---:|---:|---:|---:|---|---|---:|---:|
| 29 | 18 | 0.0 | 57 | No Fire | No Fire | 40.7% | 59.3% |
| 29 | 13 | 1.3 | 61 | No Fire | No Fire | 0.0% | 100.0% |
| 31 | 14 | 0.0 | 67 | Fire | Fire | 61.3% | 38.7% |
| 33 | 13 | 0.0 | 54 | Fire | Fire | 60.8% | 39.2% |

Percentages are rounded for display. These are workflow examples from the workbook; predictive performance is reported separately on the Model Performance page.
