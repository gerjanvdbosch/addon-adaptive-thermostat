import os
import json


def load_options():
    try:
        sensors = json.loads(os.getenv("SENSORS", None))
    except Exception:
        sensors = None

    return {
        "thermostat_entity": os.getenv("THERMOSTAT_ENTITY"),
        "solar_entity": os.getenv("SOLAR_ENTITY"),
        "presence_entity": os.getenv("PRESENCE_ENTITY"),
        "thermal_entity": os.getenv("THERMAL_ENTITY"),
        "thermostat_interval_seconds": int(
            os.getenv("THERMOSTAT_INTERVAL_SECONDS", 60)
        ),
        "solar_interval_seconds": int(os.getenv("SOLAR_INTERVAL_SECONDS", 15)),
        "cooldown_hours": float(os.getenv("COOLDOWN_HOURS", 1)),
        "full_retrain_time": os.getenv("FULL_RETRAIN_TIME", "03:00"),
        "stability_hours": float(os.getenv("STABILITY_HOURS", 6.0)),
        "min_setpoint": float(os.getenv("MIN_SETPOINT", 15.0)),
        "max_setpoint": float(os.getenv("MAX_SETPOINT", 24.0)),
        "min_change_threshold": float(os.getenv("MIN_CHANGE_THRESHOLD", 0.3)),
        "buffer_days": int(os.getenv("BUFFER_DAYS", 730)),
        "webapi_host": os.getenv("WEBAPI_HOST", "0.0.0.0"),
        "webapi_port": int(os.getenv("WEBAPI_PORT", 8000)),
        "thermostat_model_path": os.getenv("THERMOSTAT_MODEL_PATH"),
        "solar_model_path": os.getenv("SOLAR_MODEL_PATH"),
        "presence_model_path": os.getenv("PRESENCE_MODEL_PATH"),
        "thermal_model_path": os.getenv("THERMAL_MODEL_PATH"),
        "sensors": sensors,
    }
