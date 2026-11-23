"""Trusted local model loading and the four-input application contract."""
from io import BytesIO
from hashlib import sha256
from pathlib import Path
from numbers import Real

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from preprocessing import FEATURES, form_row

MODEL_FILE = Path(__file__).resolve().parent / "model" / "best_model.joblib"
MODEL_HASH = "09a3c8d9cbd41023bbee165b8fc22d47eb3e19ce4c31c41a2af7190f8e435ee4"
# Clean data minima and maxima, checked against the workbook in the test suite.
BOUNDS = dict(temperature=(21,43), wind_speed=(5,50), rain=(0,20.2), relative_humidity=(11,96))


def load_local_model():
    try:
        artifact = MODEL_FILE.read_bytes()
    except OSError as exc:
        raise RuntimeError("The bundled model is missing or unreadable. Extract the complete project ZIP.") from exc
    if sha256(artifact).hexdigest() != MODEL_HASH:
        raise RuntimeError("The bundled model failed its integrity check. Restore the original model file.")
    model = joblib.load(BytesIO(artifact))
    if not isinstance(model, Pipeline) or tuple(model.feature_names_in_) != FEATURES or set(model.classes_) != {0,1}:
        raise RuntimeError("Unexpected model pipeline, feature schema or class labels.")
    check_is_fitted(model)
    return model


def predict(model, *, temperature, wind_speed, rain, relative_humidity):
    values = dict(temperature=temperature, wind_speed=wind_speed, rain=rain, relative_humidity=relative_humidity)
    for name, value in values.items():
        low, high = BOUNDS[name]
        if isinstance(value,bool) or not isinstance(value,Real) or not low <= value <= high:
            raise ValueError(f"Enter a number from {low:g} to {high:g} for {name.replace('_',' ')}.")
        if name != "rain" and not float(value).is_integer():
            raise ValueError(f"Enter a whole number for {name.replace('_',' ')}.")
    row = form_row(**values)
    p = model.predict_proba(row)[0]
    if (set(model.classes_) != {0,1} or len(p) != 2 or not np.isfinite(p).all()
            or np.any(p<0) or np.any(p>1) or not np.isclose(np.sum(p),1,rtol=0,atol=1e-8)):
        raise RuntimeError("The model returned invalid class probabilities.")
    probabilities = dict(zip(model.classes_, p))
    label = int(model.predict(row)[0])
    if label not in probabilities or probabilities[label] < max(probabilities.values()):
        raise RuntimeError("The class and probability outputs disagree.")
    return {"class": "Fire" if label else "No Fire", "fire_probability":float(probabilities[1]),
            "no_fire_probability":float(probabilities[0])}
