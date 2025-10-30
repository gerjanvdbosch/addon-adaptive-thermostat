#!/usr/bin/env python3
# offline_train_sgd.py
# Script executed by OfflineTrainerRunner. Reads FEEDBACK_PATH, trains pipeline (StandardScaler + SGDRegressor),
# writes MODEL_PATH via joblib, and appends diag to DIAG_PATH. Exits with 0 on success, >0 on failure.

import os
import sys
import json
import datetime
import traceback
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib

# Read environment variables set by runner
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model_pipeline_sgd.pkl")
FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "/data/historical_feedback.json")
DIAG_PATH = os.environ.get("DIAG_PATH", "/data/training_diagnostics.json")
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "40"))

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

def append_diag(entry):
    diagnostics = []
    try:
        if os.path.exists(DIAG_PATH):
            with open(DIAG_PATH, "r") as f:
                diagnostics = json.load(f) or []
    except Exception:
        diagnostics = []
    diagnostics.append(entry)
    tmpd = DIAG_PATH + ".tmp"
    try:
        with open(tmpd, "w") as f:
            json.dump(diagnostics, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmpd, DIAG_PATH)
    except Exception:
        pass

def load_feedback(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f) or []
    except Exception:
        return []

def prepare_dataset(rows):
    eff = []
    for r in rows:
        try:
            v = float(r.get("feedback_value", 0.0) or 0.0)
        except Exception:
            v = 0.0
        if abs(v) > 1e-9 or r.get("reason") == "setpoint":
            eff.append(r)
    n = len(eff)
    if n == 0:
        return np.zeros((0, len(FEATURES))), np.zeros((0,)), []
    X = np.zeros((n, len(FEATURES)), dtype=float)
    y = np.zeros((n,), dtype=float)
    for i, r in enumerate(eff):
        for j, f in enumerate(FEATURES):
            try:
                X[i, j] = float(r.get(f, 0.0) or 0.0)
            except Exception:
                X[i, j] = 0.0
        if abs(X[i,10]) < 1e-9 and abs(X[i,11]) < 1e-9 and ("half_hour_index" in r or "hour" in r):
            hh = int(r.get("half_hour_index", 0))
            theta = 2.0 * np.pi * (float(hh) / 48.0)
            X[i,10] = np.sin(theta); X[i,11] = np.cos(theta)
        try:
            y[i] = float(r.get("feedback_value", 0.0) or 0.0)
        except Exception:
            y[i] = 0.0
    # try ordering by timestamp
    try:
        ts = [r.get("timestamp") for r in eff]
        if any(ts):
            idx = np.argsort([np.datetime64(t) if t is not None else np.datetime64('1970-01-01') for t in ts])
            X = X[idx]; y = y[idx]
    except Exception:
        pass
    return X, y, eff

def main():
    start_ts = datetime.datetime.now().isoformat()
    pid = os.getpid()
    append_diag({"timestamp": start_ts, "type": "offline_train_started", "pid": pid})
    try:
        rows = load_feedback(FEEDBACK_PATH)
        X, y, eff = prepare_dataset(rows)
        n = X.shape[0]
        append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "offline_train_info", "n_samples_effective": int(n), "min_required": MIN_SAMPLES})
        if n < MIN_SAMPLES:
            append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "offline_train_aborted", "reason": "not_enough_samples", "n": int(n)})
            print(f"Not enough samples: {n} < {MIN_SAMPLES}", file=sys.stderr)
            return 2

        sgd_params = {
            "loss": "squared_loss",
            "penalty": "l2",
            "alpha": float(1e-4),
            "max_iter": int(1000),
            "tol": float(1e-4),
            "learning_rate": "invscaling"
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SGDRegressor(**sgd_params))
        ])

        # Fit pipeline on full data
        pipeline.fit(X, y)

        # TimeSeriesSplit validation
        n_splits = max(2, min(5, max(2, n // 10)))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        val_mses = []
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < 2 or len(test_idx) < 1:
                continue
            pipeline.fit(X[train_idx], y[train_idx])
            y_pred = pipeline.predict(X[test_idx])
            val_mses.append(float(mean_squared_error(y[test_idx], y_pred)))
        mean_val_mse = float(np.mean(val_mses)) if val_mses else None

        # Save model atomically
        tmp_model = MODEL_PATH + ".tmp"
        try:
            joblib.dump(pipeline, tmp_model)
            if os.path.exists(tmp_model):
                os.replace(tmp_model, MODEL_PATH)
            append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "offline_train_finished", "n_samples": int(n), "mean_val_mse": mean_val_mse})
            print(f"Training complete. n={n}, mean_val_mse={mean_val_mse}", file=sys.stdout)
            return 0
        except Exception as e:
            append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "offline_train_save_error", "error": str(e)})
            print(f"Failed saving model: {e}", file=sys.stderr)
            return 3

    except Exception as e:
        tb = traceback.format_exc()
        append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "offline_train_exception", "error": str(e), "trace": tb})
        print(f"Exception during training: {e}\n{tb}", file=sys.stderr)
        return 4

if __name__ == "__main__":
    code = main()
    # ensure exit code is int and not None
    try:
        sys.exit(int(code))
    except Exception:
        sys.exit(1)
