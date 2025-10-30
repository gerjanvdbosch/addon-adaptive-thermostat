# trainer.py
import os
import datetime
import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

FEATURES = [
    "huidige_temp","temp_verandering",
    "min_temp_vandaag","max_temp_vandaag",
    "min_temp_morgen","max_temp_morgen",
    "zon_kwh","zon_kans","terugleveren",
    "huidige_setpoint",
    "hour_sin","hour_cos",
    "windrichting_vandaag","windkracht_vandaag",
    "windrichting_morgen","windkracht_morgen",
    "verwarmen","buiten_temp",
    "thermostaat_vraag","weekday"
]

def load_feedback_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f) or []
    except Exception:
        return []

def prepare_dataset_from_list(rows):
    # rows: list of dicts
    # filter effective records
    eff = []
    for r in rows:
        try:
            v = float(r.get("feedback_value", 0.0) or 0.0)
        except Exception:
            v = 0.0
        if abs(v) > 1e-6 or r.get("reason") == "setpoint":
            eff.append(r)
    n = len(eff)
    if n == 0:
        return np.zeros((0, len(FEATURES))), np.zeros((0,)), []
    X = np.zeros((n, len(FEATURES)), dtype=float)
    y = np.zeros((n,), dtype=float)
    # fill arrays
    for i, r in enumerate(eff):
        for j, f in enumerate(FEATURES):
            try:
                X[i, j] = float(r.get(f, 0.0) or 0.0)
            except Exception:
                X[i, j] = 0.0
        # if hour_sin/hour_cos missing but half_hour_index present, derive
        if abs(X[i,10]) < 1e-9 and abs(X[i,11]) < 1e-9 and ("half_hour_index" in r or "hour" in r):
            hh = int(r.get("half_hour_index", 0))
            theta = 2.0 * np.pi * (float(hh) / 48.0)
            X[i,10] = np.sin(theta); X[i,11] = np.cos(theta)
        try:
            y[i] = float(r.get("feedback_value", 0.0) or 0.0)
        except Exception:
            y[i] = 0.0
    # sort by timestamp if possible to respect chronology
    try:
        ts = [r.get("timestamp") for r in eff]
        if any(ts):
            idx = np.argsort([np.datetime64(t) if t is not None else np.datetime64('1970-01-01') for t in ts])
            X = X[idx]; y = y[idx]
    except Exception:
        pass
    return X, y, eff

class Trainer:
    def __init__(self, feedback_path):
        self.feedback_path = feedback_path

    def train_sgd(self, sgd_params, min_samples=40, n_splits=5):
        rows = load_feedback_json(self.feedback_path)
        X, y, _ = prepare_dataset_from_list(rows)
        n = X.shape[0]
        if n < min_samples:
            return None, {"reason": "not_enough_samples", "n": n}

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SGDRegressor(**sgd_params))
        ])

        # Fit on full data
        pipeline.fit(X, y)

        # TimeSeriesSplit validation (respect order)
        n_splits = max(2, min(n_splits, max(2, n//10)))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        val_mses = []
        try:
            for train_idx, test_idx in tscv.split(X):
                if len(train_idx) < 2 or len(test_idx) < 1:
                    continue
                pipeline.fit(X[train_idx], y[train_idx])
                y_pred = pipeline.predict(X[test_idx])
                val_mses.append(float(mean_squared_error(y[test_idx], y_pred)))
        except Exception:
            val_mses = []

        mean_val_mse = float(np.mean(val_mses)) if val_mses else None

        info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "n_samples": int(n),
            "mean_val_mse": mean_val_mse,
            "val_mses": val_mses,
            "sgd_params": sgd_params
        }
        return pipeline, info
