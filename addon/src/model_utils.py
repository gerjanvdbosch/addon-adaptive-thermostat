# model_utils.py
import os
import json
import tempfile

def atomic_save_json(obj, path, indent=2):
    """
    Schrijf JSON atomisch naar path.
    """
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False

def atomic_save(obj, path, indent=2):
    """
    Alias voor atomic_save_json (voor generieke objecten).
    """
    return atomic_save_json(obj, path, indent=indent)

def _read_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []

def append_diag(diag_path, entry):
    """
    Lees bestaande diagnostics array, append entry en sla atomisch op.
    Als bestand niet bestaat, wordt een nieuwe lijst aangemaakt.
    """
    try:
        diagnostics = _read_json(diag_path)
        diagnostics.append(entry)
        atomic_save_json(diagnostics, diag_path)
        return True
    except Exception:
        return False
