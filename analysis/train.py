"""Reproduce the three baseline and three tuned candidates using four weather inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocessing import FEATURES, load_dataset, make_pipelines, primary_split

GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [3, 5, 8], "min_samples_leaf": [3, 8, 16]},
    "random_forest": {"n_estimators": [100, 200], "max_depth": [5, 10],
                      "min_samples_leaf": [2, 8], "max_features": ["sqrt", 1.0]},
}


def date_folds(dates):
    windows = [("2017-06-01", "2017-10-01"), ("2021-06-01", "2021-07-01"),
               ("2021-07-01", "2021-08-01")]
    folds, rows = [], []
    for start, end in windows:
        train = np.flatnonzero(dates < start)
        valid = np.flatnonzero((dates >= start) & (dates < end))
        assert dates.iloc[train].max() < dates.iloc[valid].min()
        folds.append((train, valid))
        rows.append(dict(valid_start=start, valid_end_exclusive=end, train_rows=len(train),
                         valid_rows=len(valid), last_training_date=dates.iloc[train].max()))
    return folds, pd.DataFrame(rows)


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    metrics = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
                   recall=recall_score(y, pred), fire_f1=f1_score(y, pred), roc_auc=roc_auc_score(y, prob))
    predictions = pd.DataFrame(dict(actual=y, predicted=pred, fire_probability=prob), index=y.index)
    return metrics, predictions


def train(workbook, output):
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    models.mkdir(exist_ok=True)
    data = load_dataset(workbook)
    role = primary_split(data.meta)
    X = data.X.loc[role.eq("train")]
    y = data.y.loc[X.index]
    valid = data.X.loc[role.eq("validation")]
    truth = data.y.loc[valid.index]
    folds, fold_summary = date_folds(data.meta.loc[X.index, "date"])
    fold_summary.to_csv(output / "cv_folds.csv", index=False)
    data.meta.join(role).to_csv(output / "split_manifest.csv")
    results, predictions, searches = [], [], []
    pipelines = make_pipelines()

    def record(name, stage, model):
        metrics, rows = evaluate(model, valid, truth)
        params = model["classifier"].get_params()
        settings = {k: params[k] for k in GRIDS[name]}
        results.append(dict(model=name, stage=stage, parameters=json.dumps(settings),
                            train_fire_f1=f1_score(y, model.predict(X)), **metrics))
        predictions.append(rows.assign(model=name, stage=stage))
        path = models / f"{name}_{stage}.joblib"
        joblib.dump(model, path, compress=3)
        restored = joblib.load(path)
        assert tuple(restored.feature_names_in_) == FEATURES
        np.testing.assert_array_equal(restored.predict_proba(valid), model.predict_proba(valid))

    for name, pipeline in pipelines.items():
        record(name, "baseline", clone(pipeline).fit(X, y))

    for name, pipeline in pipelines.items():
        grid = {"classifier__" + key: vals for key, vals in GRIDS[name].items()}
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=folds, n_jobs=2, error_score="raise",
                              return_train_score=True, refit=True)
        search.fit(X, y)
        scores = pd.DataFrame(search.cv_results_)
        cols = ["params", "rank_test_score", "mean_test_score", "std_test_score", "mean_train_score",
                *[f"split{i}_test_score" for i in range(len(folds))]]
        searches.append(scores[cols].assign(model=name, selected=scores.index == search.best_index_))
        record(name, "tuned", search.best_estimator_)

    table = pd.DataFrame(results)
    table.to_csv(output / "validation_results.csv", index=False)
    pd.concat(predictions).to_csv(output / "august_predictions.csv")
    pd.concat(searches).to_csv(output / "grid_results.csv", index=False)
    (output / "workbook_sha256.txt").write_text(hashlib.sha256(workbook.read_bytes()).hexdigest()+"\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    train(args.workbook, args.output)
