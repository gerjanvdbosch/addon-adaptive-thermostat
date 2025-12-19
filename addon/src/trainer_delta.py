import os
import logging
import joblib
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from db import fetch_training_setpoints
from collector import FEATURE_ORDER

logger = logging.getLogger(__name__)


def _atomic_dump(obj, path: str):
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _assemble_matrix_delta(rows, feature_order):
    """
    Build X and y where y = setpoint - baseline_current_setpoint (delta).
    Prefer Setpoint.observed_current_setpoint as baseline; fallback to feat["current_setpoint"].

    UPDATE: Uses np.nan for missing features instead of 0.0.
    """
    X = []
    y = []
    used_rows = []

    for r in rows:
        # rows are Setpoint objects (from fetch_training_setpoints)
        feat = r.data if r.data and isinstance(r.data, dict) else {}

        try:
            # label is the logged setpoint (the user override)
            label = _safe_float(getattr(r, "setpoint", None), None)
            if label is None:
                continue

            # sanity bounds on absolute setpoint
            if not (5 <= label <= 30):
                continue

            # baseline: prefer explicit observed_current_setpoint field on Setpoint row
            curr_raw = getattr(r, "observed_current_setpoint", None)
            if curr_raw is None:
                curr_raw = feat.get("current_setpoint")

            current = _safe_float(curr_raw, None)
            if current is None:
                continue

            delta = label - current

            # Construct feature vector
            vec = []
            for k in feature_order:
                v = feat.get(k)
                if v is None:
                    # Use NaN for missing values so the model learns it's missing
                    # instead of treating it as 0.0
                    vec.append(np.nan)
                else:
                    try:
                        vec.append(float(v))
                    except Exception:
                        vec.append(np.nan)

            X.append(vec)
            y.append(delta)
            used_rows.append(r)
        except Exception:
            logger.exception(
                "Skipping corrupt setpoint row %s in assemble_matrix_delta",
                getattr(r, "id", None),
            )

    if not X:
        return None, None, []
    return np.array(X, dtype=float), np.array(y, dtype=float), used_rows


def _clean_train_arrays(Xa: np.ndarray, ya: np.ndarray):
    """
    Ensure numeric arrays.
    Drops rows where y is non-finite.
    Allows X to contain NaNs (handled natively by HistGradientBoostingRegressor).
    """
    Xa = np.asarray(Xa, dtype=float)
    ya = np.asarray(ya, dtype=float)

    if Xa.ndim != 2:
        raise ValueError(f"X must be 2D array, got ndim={Xa.ndim}")
    if ya.ndim != 1:
        raise ValueError(f"y must be 1D array, got ndim={ya.ndim}")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"Row count mismatch X ({Xa.shape[0]}) vs y ({ya.shape[0]})")

    # Only drop if Target (y) is broken. NaNs in X are allowed now.
    finite_y_mask = np.isfinite(ya)
    dropped = np.count_nonzero(~finite_y_mask)

    if dropped:
        logger.warning("Dropping %d rows with non-finite targets", int(dropped))

    if not np.any(finite_y_mask):
        raise ValueError("No valid target rows after cleaning")

    Xa_clean = Xa[finite_y_mask]
    ya_clean = ya[finite_y_mask]
    return Xa_clean, ya_clean


class TrainerDelta:
    """
    Trainer that learns delta = setpoint - baseline_current_setpoint.
    Optimized with Early Stopping and MAE loss.
    """

    def __init__(self, ha_client=None, opts=None):
        self.ha = ha_client
        self.opts = opts or {}
        self.model_path = self.opts.get(
            "model_path", "/config/models/delta_model.joblib"
        )
        self.feature_order = FEATURE_ORDER
        self.random_state = int(self.opts.get("random_state", 42))

        # safe defaults
        if "warning_std_threshold" not in self.opts:
            self.opts["warning_std_threshold"] = 1e-3

    def _fetch_data(self):
        rows = fetch_training_setpoints(days=int(self.opts.get("buffer_days", 30)))
        labeled_rows = [r for r in rows if getattr(r, "setpoint", None) is not None]
        X_lab, y_lab, used_rows = _assemble_matrix_delta(
            labeled_rows, self.feature_order
        )

        if X_lab is None:
            return None, None, [], 0

        X = X_lab
        y = y_lab

        # diagnostics
        try:
            if y is not None and len(y):
                std = float(np.std(y))
                logger.debug(
                    "Training deltas: n=%d mean=%.4f std=%.4f",
                    len(y),
                    float(np.mean(y)),
                    std,
                )
                if std < float(self.opts.get("warning_std_threshold", 1e-3)):
                    logger.warning("Training deltas near-constant (std=%.6f)", std)
        except Exception:
            pass

        return X, y, used_rows, 0

    def train_job(self, force: bool = False):
        start = time.time()
        X, y_delta, used_rows, _ = self._fetch_data()

        if X is None or len(X) < 10:
            logger.info("TrainerDelta: Not enough training data (<10 samples)")
            return

        # Prepare data: Drop invalid targets, keep X NaNs
        try:
            X, y_delta = _clean_train_arrays(X, y_delta)
        except Exception as e:
            logger.error("Data cleaning failed: %s", e)
            return

        n_total = len(y_delta)
        self.opts["last_n_labeled"] = n_total

        # Split:
        # We reserve the last 15% purely for reporting "Holdout MAE" to the user.
        # The model *also* uses an internal split (validation_fraction) for Early Stopping.
        val_frac = 0.15
        val_size = max(1, int(n_total * val_frac))

        if n_total < 50:
            # Very small dataset: use everything for training to get at least something working
            X_train, y_train = X, y_delta
            X_val = None
            logger.info(
                "Dataset too small for holdout split, using all data for training."
            )
        else:
            X_train = X[:-val_size]
            y_train = y_delta[:-val_size]
            X_val = X[-val_size:]
        #             y_val = y_delta[-val_size:]

        logger.info(
            "TrainerDelta: n_train=%d n_holdout=%d",
            len(X_train),
            len(X_val) if X_val is not None else 0,
        )

        # --- OPTIMIZED MODEL CONFIGURATION ---
        # No SearchCV needed. HistGradientBoostingRegressor is smart enough with these settings.
        model = HistGradientBoostingRegressor(
            loss="absolute_error",  # Minimize MAE directly (better for setpoints)
            learning_rate=0.05,  # Good balance speed/precision
            max_iter=2000,  # High limit, rely on early_stopping
            max_leaf_nodes=31,  # Standard, prevents overfitting
            min_samples_leaf=20,  # Robustness against outliers
            l2_regularization=1.0,  # Regularization
            early_stopping=True,  # Stop when validation score plateaus
            validation_fraction=0.15,  # Internal split for early stopping
            n_iter_no_change=20,  # Patience
            random_state=self.random_state,
        )

        try:
            model.fit(X_train, y_train)
            logger.debug(
                "Model fit complete. n_iter_=%d score=%.4f",
                model.n_iter_,
                model.train_score_[-1],
            )
        except Exception:
            logger.exception("Model fitting failed")
            return

        # ---------------------------------------------------------
        # METRICS & REPORTING (Reconstruct absolute temperatures)
        # ---------------------------------------------------------

        def reconstruct_abs(X_subset, y_pred_subset, rows_subset):
            # Reconstruct baseline array
            baselines = []
            targets = []
            valid_indices = []

            for i, pred in enumerate(y_pred_subset):
                r = rows_subset[i]
                # Try to get baseline from row object or feature dict
                b_val = getattr(r, "observed_current_setpoint", None)
                if b_val is None:
                    # Fallback to the feature used during training (index varies, retrieve from dict)
                    if r.data:
                        b_val = r.data.get("current_setpoint")

                t_val = getattr(r, "setpoint", None)

                b = _safe_float(b_val)
                t = _safe_float(t_val)

                if b is not None and t is not None:
                    baselines.append(b)
                    targets.append(t)
                    valid_indices.append(i)

            if not baselines:
                return None, None

            preds_abs = np.array(baselines) + y_pred_subset[valid_indices]
            true_abs = np.array(targets)
            return preds_abs, true_abs

        # 1. Training/Full Set Performance
        mae_abs = None
        try:
            # Predict on the portion used for training (OOF-like analysis)
            preds_train_delta = model.predict(X_train)
            p_abs, t_abs = reconstruct_abs(
                X_train, preds_train_delta, used_rows[: len(X_train)]
            )

            if p_abs is not None:
                mae_abs = float(mean_absolute_error(t_abs, p_abs))
                self._report_household_drift(used_rows[: len(X_train)], p_abs, t_abs)
        except Exception:
            logger.exception("Failed computing training MAE")

        # 2. Holdout Validation Performance
        val_mae_abs = None
        try:
            if X_val is not None and len(X_val) > 0:
                preds_val_delta = model.predict(X_val)
                # Map back to the correct rows (tail of used_rows)
                val_rows = used_rows[-len(X_val) :]
                p_val_abs, t_val_abs = reconstruct_abs(X_val, preds_val_delta, val_rows)

                if p_val_abs is not None:
                    val_mae_abs = float(mean_absolute_error(t_val_abs, p_val_abs))
        except Exception:
            logger.exception("Failed computing validation MAE")

        runtime = time.time() - start
        logger.info(
            "Training finished. Runtime=%.2fs. Train MAE=%.3f. Holdout MAE=%.3f. Trees=%d",
            runtime,
            mae_abs if mae_abs else 0.0,
            val_mae_abs if val_mae_abs else 0.0,
            model.n_iter_,
        )

        # Save
        try:
            meta = {
                "feature_order": self.feature_order,
                "backend": "sklearn_histgb_optimized",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "mae": mae_abs,
                "val_mae": val_mae_abs,
                "n_samples": n_total,
                "model_params": model.get_params(),
                "runtime_seconds": runtime,
                "target": "delta",
            }

            payload = {"model": model, "meta": meta}
            _atomic_dump(payload, self.model_path)
            logger.debug("Saved model to %s", self.model_path)

        except Exception:
            logger.exception("Failed saving model")

    def _report_household_drift(self, used_rows, preds_abs, y_true_abs):
        try:
            hh_map = {}
            # We need to match the filtered predictions back to rows.
            # Note: reconstruct_abs filters rows if baseline missing, so lengths might differ slightly
            # if data is messy, but usually they match.
            limit = min(len(used_rows), len(preds_abs))

            for i in range(limit):
                r = used_rows[i]
                hh = getattr(r, "household_id", None) or "unknown"
                hh_map.setdefault(hh, []).append((preds_abs[i], y_true_abs[i]))

            for hh, vals in hh_map.items():
                errors = [
                    abs(p - t) for p, t in vals if np.isfinite(p) and np.isfinite(t)
                ]
                if not errors:
                    continue
                hh_mae = float(np.mean(errors))
                if len(errors) > 5:  # Only report if statistically somewhat relevant
                    logger.debug(
                        "Household %s MAE=%.4f (n=%d)", hh, hh_mae, len(errors)
                    )
        except Exception:
            logger.warning("Could not report household drift details")
