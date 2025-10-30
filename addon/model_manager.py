# model_manager.py
import os
import shutil
import joblib
import numpy as np
import datetime
from model_utils import atomic_save, append_diag

class ModelManager:
    def __init__(self, model_path, diag_path, clamp_action=3.0):
        self.model_path = model_path
        self.diag_path = diag_path
        self.clamp_action = clamp_action
        self.pipeline = None
        self.load()

    def load(self):
        try:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
                return True
        except Exception:
            self.pipeline = None
        return False

    def save(self, make_backup=True):
        try:
            if make_backup and os.path.exists(self.model_path):
                try:
                    shutil.copy2(self.model_path, self.model_path + ".bak")
                except Exception:
                    pass
            atomic_save(self.pipeline, self.model_path)
            return True
        except Exception:
            return False

    def predict(self, features):
        if self.pipeline is None:
            return 0.0
        X = np.asarray(features, dtype=float).reshape(1, -1)
        try:
            corr = float(self.pipeline.predict(X)[0])
        except Exception:
            corr = 0.0
        return max(-self.clamp_action, min(self.clamp_action, corr))

    def safe_partial_fit(self, features, target_feedback, feedback_count_getter, save=True, max_delta=None, min_abs_update=None):
        if self.pipeline is None:
            return {"ok": False, "reason": "no_pipeline"}
        try:
            n = feedback_count_getter()
            from config import get_adaptive_config
            cfg = get_adaptive_config(n)
            max_delta_eff = max_delta if max_delta is not None else cfg["max_delta"]
            min_abs_update_eff = min_abs_update if min_abs_update is not None else cfg["min_abs_update"]

            scaler = self.pipeline.named_steps.get("scaler")
            model = self.pipeline.named_steps.get("model")
            if model is None or not hasattr(model, "partial_fit"):
                return {"ok": False, "reason": "model_no_partial_fit"}

            X = np.asarray(features, dtype=float).reshape(1, -1)
            Xs = scaler.transform(X) if scaler is not None else X

            try:
                before = float(model.predict(Xs)[0])
            except Exception:
                before = 0.0

            # perform partial_fit
            model.partial_fit(Xs, [float(target_feedback)])

            try:
                after = float(model.predict(Xs)[0])
            except Exception:
                after = before

            delta = after - before

            # --- Smart normalization of delta ---
            eps = 1e-9
            pred_std = None
            try:
                coef = None
                if hasattr(model, "coef_"):
                    coef = np.asarray(getattr(model, "coef_")).flatten()
                if coef is not None and scaler is not None and hasattr(scaler, "scale_"):
                    feature_scale = np.asarray(getattr(scaler, "scale_"), dtype=float)
                    denom = np.where(feature_scale <= 0, 1.0, feature_scale)
                    pred_std_est = np.sqrt(np.sum((coef / denom) ** 2))
                    pred_std = float(pred_std_est)
                elif coef is not None:
                    pred_std = float(np.linalg.norm(coef))
                else:
                    pred_std = float(np.linalg.norm(X) + eps)
            except Exception:
                pred_std = float(np.linalg.norm(X) + eps)

            norm_delta = delta / (pred_std + eps)
            max_delta_norm = float(max_delta_eff) / (pred_std + eps)

            status = "accepted"
            accepted = True
            if abs(delta) < min_abs_update_eff:
                status = "tiny_update"
                accepted = True
            elif abs(norm_delta) > max_delta_norm:
                status = "rejected_delta"
                accepted = False

            if not accepted:
                if os.path.exists(self.model_path + ".bak"):
                    try:
                        self.pipeline = joblib.load(self.model_path + ".bak")
                    except Exception:
                        pass
                append_diag(self.diag_path, {"timestamp": datetime.datetime.now().isoformat(), "type": "partial_fit", "status": status, "delta": float(delta), "norm_delta": float(norm_delta), "pred_std": float(pred_std)})
                return {"ok": False, "status": status, "delta": float(delta), "norm_delta": float(norm_delta)}

            if save:
                self.save(make_backup=True)

            append_diag(self.diag_path, {"timestamp": datetime.datetime.now().isoformat(), "type": "partial_fit", "status": status, "delta": float(delta), "norm_delta": float(norm_delta)})
            return {"ok": True, "status": status, "delta": float(delta), "norm_delta": float(norm_delta)}
        except Exception as e:
            append_diag(self.diag_path, {"timestamp": datetime.datetime.now().isoformat(), "type": "partial_fit_error", "error": str(e)})
            return {"ok": False, "reason": str(e)}
