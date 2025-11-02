# config.py
import os
import yaml

_here = os.path.dirname(__file__)
CFG_PATH = os.path.join(os.path.abspath(os.path.join(_here, "..")), "config_default.yaml")

def _load_defaults():
    try:
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

cfg_default = _load_defaults()

def _get_from_cfg(key_path, default=None):
    cur = cfg_default
    try:
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
    except Exception:
        return default

def get_adaptive_config(n_feedback=0):
    max_delta = _get_from_cfg(["correction", "max_delta"]) or 0.5
    min_abs_update = _get_from_cfg(["correction", "min_abs_update"]) or 0.01
    min_train_samples = _get_from_cfg(["min_train_samples"]) or 40
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
