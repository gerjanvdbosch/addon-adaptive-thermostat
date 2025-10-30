# api_service.py
import os
import json
import time
import threading
import datetime
import requests
import yaml
from flask import Flask, request, jsonify
from model_utils import append_diag, atomic_save_json
from trainer import Trainer, FEATURES
from store import FeedbackStore
from model_manager import ModelManager
from config import get_adaptive_config
from offline_trainer import OfflineTrainerRunner

here = os.path.dirname(__file__)
with open(os.path.join(here, "config_default.yaml"), "r") as f:
    cfg_default = yaml.safe_load(f)

MODEL_DIR = os.environ.get("MODEL_DIR", cfg_default.get("model_dir", "/data"))
MODEL_PATH = os.path.join(MODEL_DIR, cfg_default.get("model_filename", "model_pipeline_sgd.pkl"))
DIAG_PATH = os.path.join(MODEL_DIR, cfg_default.get("diag_filename", "training_diagnostics.json"))
FEEDBACK_PATH = os.path.join(MODEL_DIR, cfg_default.get("feedback_filename", "historical_feedback.json"))

os.makedirs(MODEL_DIR, exist_ok=True)

class APIService:
    def __init__(self):
        self.store = FeedbackStore(FEEDBACK_PATH)
        self.trainer = Trainer(FEEDBACK_PATH)
        self.model = ModelManager(MODEL_PATH, DIAG_PATH, clamp_action=cfg_default.get("safe", {}).get("clamp_action", 3.0))
        self.trainer_runner = OfflineTrainerRunner(here, MODEL_PATH, FEEDBACK_PATH, DIAG_PATH, min_samples=cfg_default.get("min_train_samples", 40))
        self.app = Flask(__name__)
        self._lock = threading.Lock()
        self._last_feedback = {"ts": 0, "value": None}
        self._register_routes()

    def _append_diag(self, entry):
        append_diag(DIAG_PATH, entry)

    def _should_debounce(self, value, debounce_seconds):
        try:
            with self._lock:
                now = time.time()
                last_ts = self._last_feedback.get("ts", 0)
                last_val = self._last_feedback.get("value", 0.0) or 0.0
                if now - last_ts < float(debounce_seconds):
                    if (value * last_val >= 0) and abs(value - last_val) <= 0.5:
                        return True
                self._last_feedback["ts"] = now
                self._last_feedback["value"] = value
                return False
        except Exception:
            return False

    def _ha_setpoint_via_rest(self, value, entity, ha_url, ha_token):
        headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
        payload = {"entity_id": entity, "value": float(value)}
        try:
            r = requests.post(f"{ha_url}/api/services/number/set_value", json=payload, headers=headers, timeout=5)
            if r.ok:
                return {"ok": True, "method": "number.set_value"}
        except Exception:
            pass
        try:
            payload = {"entity_id": entity, "temperature": float(value)}
            r = requests.post(f"{ha_url}/api/services/climate/set_temperature", json=payload, headers=headers, timeout=5)
            if r.ok:
                return {"ok": True, "method": "climate.set_temperature"}
        except Exception:
            pass
        return {"ok": False, "reason": "ha_call_failed"}

    def _register_routes(self):
        @self.app.route("/predict", methods=["POST"])
        def predict_route():
            payload = request.get_json(force=True)
            features = payload.get("features")
            if features is None:
                return jsonify({"error": "missing features"}), 400
            corr = self.model.predict(features)
            return jsonify({"correction": float(corr)})

        @self.app.route("/feedback", methods=["POST"])
        def feedback_route():
            payload = request.get_json(force=True)
            features = payload.get("features")
            feedback_value = payload.get("feedback_value")
            reason = payload.get("reason", "user")
            if features is None or feedback_value is None:
                return jsonify({"error": "missing features or feedback_value"}), 400
            try:
                feedback_value = float(feedback_value)
                feedback_value = max(-3.0, min(3.0, feedback_value))
            except Exception:
                return jsonify({"error": "invalid feedback_value"}), 400

            debounce_seconds = cfg_default.get("correction", {}).get("debounce_seconds", 30)
            if self._should_debounce(feedback_value, debounce_seconds):
                self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "debounced_feedback", "value": feedback_value})
                return jsonify({"ok": True, "status": "debounced"})

            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "reason": reason,
                "feedback_value": float(feedback_value),
                **{k: float(v) for k, v in zip(FEATURES, features)}
            }
            self.store.append(entry)

            res = self.model.safe_partial_fit(features, feedback_value, feedback_count_getter=self.store.count_effective, save=True)

            suggestion = None
            try:
                suggestion = self.model.predict(features)
            except Exception:
                suggestion = None

            self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "feedback_received", "reason": reason, "suggestion": suggestion})
            return jsonify({"ok": True, "partial_fit": res, "suggestion": suggestion})

        @self.app.route("/apply", methods=["POST"])
        def apply_route():
            if not cfg_default.get("correction", {}).get("allow_direct_apply_via_api", True):
                return jsonify({"ok": False, "reason": "direct_apply_disabled"}), 403

            payload = request.get_json(force=True)
            features = payload.get("features")
            feedback_value = payload.get("feedback_value")
            reason = payload.get("reason", "apply")
            if features is None or feedback_value is None:
                return jsonify({"error": "missing features or feedback_value"}), 400
            try:
                feedback_value = float(feedback_value)
                feedback_value = max(-3.0, min(3.0, feedback_value))
            except Exception:
                return jsonify({"error": "invalid feedback_value"}), 400

            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "reason": reason,
                "feedback_value": float(feedback_value),
                **{k: float(v) for k, v in zip(FEATURES, features)}
            }
            self.store.append(entry)

            corr_cfg = cfg_default.get("correction", {})
            ha_url = os.environ.get("HA_URL")
            ha_token = os.environ.get("HA_TOKEN")
            entity = corr_cfg.get("ha_setpoint_entity")
            try:
                idx = FEATURES.index("huidige_setpoint")
                current_setpoint = float(features[idx])
            except Exception:
                current_setpoint = None
            target_value = current_setpoint + feedback_value if current_setpoint is not None else feedback_value

            apply_result = {"ok": False, "reason": "not_attempted"}
            if corr_cfg.get("apply_via") == "ha_api" and ha_url and ha_token and entity:
                apply_result = self._ha_setpoint_via_rest(target_value, entity, ha_url, ha_token)
            elif corr_cfg.get("apply_via") == "mqtt":
                HA_URL = ha_url; HA_TOKEN = ha_token
                if HA_URL and HA_TOKEN:
                    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
                    payload_pub = {"topic": corr_cfg.get("mqtt_setpoint_topic"), "payload": f"{float(target_value):.1f}"}
                    try:
                        r = requests.post(f"{HA_URL}/api/services/mqtt/publish", json=payload_pub, headers=headers, timeout=5)
                        if r.ok:
                            apply_result = {"ok": True, "method": "mqtt_publish"}
                        else:
                            apply_result = {"ok": False, "reason": "mqtt_publish_failed"}
                    except Exception as e:
                        apply_result = {"ok": False, "reason": f"exception:{e}"}
                else:
                    apply_result = {"ok": False, "reason": "mqtt_no_ha_config"}
            else:
                apply_result = {"ok": False, "reason": "apply_method_not_configured"}

            self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "apply_attempt", "apply_result": apply_result, "target_value": target_value})

            n = self.store.count_effective()
            cfg = get_adaptive_config(n)
            stricter_max_delta = max(0.1, cfg.get("max_delta", 0.5) * 0.5)
            res = self.model.safe_partial_fit(features, feedback_value, feedback_count_getter=self.store.count_effective, save=True, max_delta=stricter_max_delta)

            return jsonify({"ok": True, "apply_result": apply_result, "partial_fit": res})

        @self.app.route("/train", methods=["POST"])
        def train_route():
            job_id = self.trainer_runner.start_job()
            return jsonify({"ok": True, "job_id": job_id}), 202

        @self.app.route("/train/status/<job_id>", methods=["GET"])
        def train_status_route(job_id):
            info = self.trainer_runner.job_status(job_id)
            if not info:
                return jsonify({"ok": False, "reason": "not_found"}), 404
            return jsonify({"ok": True, "job": info})

        @self.app.route("/health", methods=["GET"])
        def health_route():
            return jsonify({"ok": True, "pipeline_loaded": self.model.pipeline is not None})

# instantiate and expose WSGI app
_service = APIService()
app = _service.app
