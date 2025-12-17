import os
import json


def load_options():
    try:
        sensors = json.loads(os.getenv("SENSORS", None))
    except Exception:
        sensors = None

    return {
        "climate_entity": os.getenv("CLIMATE_ENTITY", "climate.living_room"),
        "shadow_mode": bool(os.getenv("SHADOW_MODE")),
        "shadow_setpoint": os.getenv("SHADOW_SETPOINT"),
        "inferencer_interval_seconds": int(
            os.getenv("INFERENCER_INTERVAL_SECONDS", 60)
        ),
        "solar_interval_seconds": int(os.getenv("SOLAR_INTERVAL_SECONDS", 15)),
        "sample_interval_seconds": int(os.getenv("SAMPLE_INTERVAL_SECONDS", 300)),
        "cooldown_hours": float(os.getenv("COOLDOWN_HOURS", 1)),
        "full_retrain_time": os.getenv("FULL_RETRAIN_TIME", "03:00"),
        "stability_hours": float(os.getenv("STABILITY_HOURS", 6.0)),
        "min_setpoint": float(os.getenv("MIN_SETPOINT", 15.0)),
        "max_setpoint": float(os.getenv("MAX_SETPOINT", 24.0)),
        "min_change_threshold": float(os.getenv("MIN_CHANGE_THRESHOLD", 0.3)),
        "buffer_days": int(os.getenv("BUFFER_DAYS", 30)),
        "webapi_host": os.getenv("WEBAPI_HOST", "0.0.0.0"),
        "webapi_port": int(os.getenv("WEBAPI_PORT", 8000)),
        "model_path": os.getenv("MODEL_PATH"),
        "model_path_full": os.getenv("MODEL_PATH_FULL"),
        "use_unlabeled": bool(os.getenv("USE_UNLABELED")),
        "pseudo_limit": int(os.getenv("PSEUDO_LIMIT", 1000)),
        "weight_label": float(os.getenv("WEIGHT_LABEL", 1.0)),
        "weight_pseudo": float(os.getenv("WEIGHT_PSEUDO", 0.25)),
        "sensors": sensors,
    }
