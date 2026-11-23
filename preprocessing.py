"""Four-weather-input data contract, unfitted model pipelines, and temporal splits."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ("temperature", "wind_speed", "rain", "relative_humidity")
INDICES = ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")
SEED = 42
VALID_FROM = pd.Timestamp("2021-08-01")
TEST_FROM = pd.Timestamp("2021-09-01")


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def numeric_weather(frame):
    """Nulls in historical inputs are handled later by training-fold imputation."""
    values = frame.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("Weather inputs must be finite or missing.")
    return values


def form_row(*, temperature, wind_speed, rain, relative_humidity):
    """Call with four named Streamlit number_input values, in recorded units."""
    row = numeric_weather(pd.DataFrame([dict(temperature=temperature, wind_speed=wind_speed,
                                             rain=rain, relative_humidity=relative_humidity)]))
    if row.isna().any().any():
        raise ValueError("Enter all four weather readings.")
    return row


def load_dataset(workbook: Path):
    frame = pd.read_excel(workbook, sheet_name="Clean data").set_index("source_row", verify_integrity=True)
    X = numeric_weather(frame)
    y = frame["fire"]
    labels = frame.class_label.str.strip().str.lower().map({"fire": 1, "not fire": 0})
    if not y.isin([0, 1]).all() or not labels.eq(y).all():
        raise ValueError("Target or label mismatch.")
    meta = frame.loc[:, ["date", "region", "date_status"]].copy()
    meta["date"] = pd.to_datetime(meta.date, errors="raise").dt.normalize()
    if meta.isna().any().any() or meta.duplicated(["date", "region"]).any():
        raise ValueError("Missing metadata or repeated region-day key requires review.")
    if not meta.date_status.isin(["as_recorded", "excel_serial_converted", "year_corrected_inferred"]).all():
        raise ValueError("Review unexpected date status before splitting.")
    return Dataset(X, y.astype(int), meta)


def primary_split(meta):
    """Series of roles indexed by original source_row, with whole calendar dates together."""
    role = pd.Series("train", index=meta.index, name="split")
    role.loc[meta.date.ge(VALID_FROM)] = "validation"
    role.loc[meta.date.ge(TEST_FROM)] = "test"
    role.loc[meta.date_status.eq("year_corrected_inferred")] = "excluded"
    blocks = [meta.loc[role.eq(name), "date"] for name in ("train", "validation", "test")]
    if any(block.empty for block in blocks) or not (blocks[0].max() < blocks[1].min()
                                                     and blocks[1].max() < blocks[2].min()):
        raise ValueError("Workbook does not support the planned chronological blocks.")
    return role


def region_transfer_splits(meta, role):
    """August transfer sensitivity. Yield source IDs for .loc, never positional indices."""
    for region in sorted(meta.loc[role.eq("validation"), "region"].unique()):
        train = meta.index[role.eq("train") & meta.region.ne(region)]
        valid = meta.index[role.eq("validation") & meta.region.eq(region)]
        yield region, train, valid


def make_pipelines():
    """Return fresh unfitted pipelines; each preprocessing object belongs to one model."""
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    pipelines = {}
    for name, estimator in classifiers.items():
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if name == "logistic_regression":
            steps.append(("scale", StandardScaler()))
        weather = ColumnTransformer([("weather", Pipeline(steps), list(FEATURES))], remainder="drop")
        pipelines[name] = Pipeline([("preprocess", weather), ("classifier", estimator)])
    return pipelines
