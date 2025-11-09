import os
import json
from typing import Dict, Any, Optional

def load_options() -> Dict[str, Any]:
    # Support passing sensors mapping as JSON string in SENSORS env var or via mapped config (opts injected by Supervisor)
    try:
        sensors = json.loads(os.getenv("SENSORS", None))
    except Exception:
        sensors = None

    return {
        "climate_entity": os.getenv("CLIMATE_ENTITY", "climate.woonkamer"),
        "shadow_mode": bool(os.getenv("SHADOW_MODE")),
        "shadow_setpoint": os.getenv("SHADOW_SETPOINT"),
        "sample_interval_seconds": int(os.getenv("SAMPLE_INTERVAL_SECONDS", 300)),
        "partial_fit_interval_seconds": int(
            os.getenv("PARTIAL_FIT_INTERVAL_SECONDS", 3600)
        ),
        "full_retrain_time": os.getenv("FULL_RETRAIN_TIME", "03:00"),
        "min_setpoint": float(os.getenv("MIN_SETPOINT", 15.0)),
        "max_setpoint": float(os.getenv("MAX_SETPOINT", 24.0)),
        "min_change_threshold": float(os.getenv("MIN_CHANGE_THRESHOLD", 0.3)),
        "buffer_days": int(os.getenv("BUFFER_DAYS", 30)),
        "partial_learning_rate": os.getenv("PARTIAL_LEARNING_RATE", "constant"),
        "partial_eta0": float(os.getenv("PARTIAL_ETA0", 0.01)),
        "partial_alpha": float(os.getenv("PARTIAL_ALPHA", 0.0001)),
        "addon_api_token": os.getenv("ADDON_API_TOKEN", None),
        "webapi_host": os.getenv("WEBAPI_HOST", "0.0.0.0"),
        "webapi_port": int(os.getenv("WEBAPI_PORT", os.getenv("WEBAPI_PORT", 8000))),
        "model_path_partial": os.getenv("MODEL_PATH_PARTIAL"),
        "model_path_full": os.getenv("MODEL_PATH_FULL"),
        "use_unlabeled": bool(os.getenv("USE_UNLABELED")),
        "pseudo_limit": int(os.getenv("PSEUDO_LIMIT", 1000)),
        "weight_label": float(os.getenv("WEIGHT_LABEL", 1.0)),
        "weight_pseudo": float(os.getenv("WEIGHT_PSEUDO", 0.25)),
        "sensors": sensors,
    }
