# Three-minute project demonstration

| Time | On screen | What to say |
|---|---|---|
| 0:00–0:25 | Overview Dashboard | “Our question is whether four environmental measurements can classify the Fire label for the same region and day. The workbook has 1,215 observations across nine regions. Of these, 633 are Fire.” |
| 0:25–0:55 | Exploratory Data Analysis → Weather distributions | “Compare the humidity boxes. The median reading is 39 for Fire and 67 for No Fire, but the distributions overlap. Wind medians differ by only one recorded unit. Units are unverified.” |
| 0:55–1:25 | README → Feature selection, preprocessing and training | “We used the four measurements a user can enter. All three classifiers trained on the same 793 earlier rows. Grid search moved forward through dates, fitting preprocessing inside each fold. Five records with inferred date corrections were excluded.” |
| 1:25–2:05 | ML Model Performance → Validation comparison, then Final test | “Random Forest led August Fire F1 at .871. Logistic Regression caught three more fires but produced nine more false alarms. After selection we refitted the forest and tested on September. It caught 70 of 73 Fire labels and gave 39 false alarms.” |
| 2:05–2:40 | Wildfire Prediction | “Enter temperature 31, wind 14, rain 0 and relative humidity 67. Click Predict. This gives Fire, 61.3% Fire and 38.7% No Fire. Change the inputs to 29, 18, 0 and 57. This returns No Fire with 59.3%.” |
| 2:40–3:00 | ML Model Performance → Final test, reliability plot | “The model can rank cases well and still overestimate probability in some ranges. Our evidence supports classification of recorded same-day labels. Forecasting tomorrow would require a different target and independent validation.” |

The two demonstrations above are recorded workbook examples exercised by the app tests. The held-out September results provide the separate performance estimate.

## Suggested viva questions and answers

1. **What is one observation?** One region on one date, with environmental readings and a recorded binary Fire label.
2. **Why only four inputs?** The interface can collect those four readings. The workbook supplies six indices, but their computation, timing and relationship to label creation have not been independently established.
3. **What would cause direct target leakage?** The text column class_label encodes exactly the same answer as fire. Including it in X would reveal the label.
4. **Why split by date?** Nearby dates can share fire-weather conditions, and several regions share a calendar date. Whole-date chronological blocks evaluate performance on later observations.
5. **Why scale only Logistic Regression?** Regularization depends on coefficient size, which depends on the input scale. Tree splits compare thresholds and preserve their ordering under positive linear scaling.
6. **How did you control tree overfitting?** Tune maximum depth and minimum leaf size inside forward-validation folds. The selected single tree allowed depth 3 and required at least eight samples per leaf.
7. **Why select the forest if Logistic Regression had better recall?** We prioritized Fire F1, which balances precision and recall. On August the forest reduced false alarms from 39 to 30 while missed fires rose from four to seven.
8. **What is the difference between recall and precision for the final model?** Recall was 70/73 among the September Fire records. Precision was 70/109 among the records the model classified as Fire.
9. **Does 80% mean eight out of ten similar days really burn?** The number is an estimated label probability. Assess that interpretation with observed frequencies and calibration; the mid-range September scores were too high on this sample.
10. **What did you do with missing dates and extreme values?** No synthetic region-days were inserted. Five inferred dates were excluded from time-sensitive analyses. Extreme numeric readings were retained and flagged for review.
11. **How does the app avoid swapping class probabilities?** It uses the fitted classifier’s classes_ to map probability columns, validates their total and displays a percentage for each label.
12. **What would make this a forecasting project?** Define a future horizon, link today’s available inputs to the appropriate future label for each region, and test on independently collected later seasons.
