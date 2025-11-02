# config.py
import os
import yaml

_here = os.path.dirname(__file__)
_cfg_path = os.path.join(os.path.abspath(os.path.join(_here, "..")), "config_default.yaml")

def _load_defaults():
    try:
        if os.path.exists(_cfg_path):
            with open(_cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

_cfg_default = _load_defaults()

def _get_from_env_or_cfg(key_path, default=None):
    env_name = "_".join(k.upper() for k in key_path)
    val = os.environ.get(env_name)
    if val is not None:
        if isinstance(val, str) and val.lower() in ("true", "false"):
            return val.lower() == "true"
        try:
            if "." in val:
                return float(val)
            return int(val)
        except Exception:
            return val
    cur = _cfg_default
    try:
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
    except Exception:
        return default

def get_adaptive_config(n_feedback=0):
    max_delta = _get_from_env_or_cfg(["correction", "max_delta"], default=0.5)
    min_abs_update = _get_from_env_or_cfg(["correction", "min_abs_update"], default=0.01)
    min_train_samples = _get_from_env_or_cfg(["min_train_samples"], default=40)
    try:
        n = int(n_feedback or 0)
    except Exception:
        n = 0
    factor = 0.5 if n < max(1, int(min_train_samples)) else 1.0
    return {
        "max_delta": float(max_delta) * factor,
        "min_abs_update": float(min_abs_update),
        "min_train_samples": int(min_train_samples)
    }
