# store.py
import os
import json
import threading
import tempfile
import datetime

class FeedbackStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        # ensure directory exists
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)

    def _read_all(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f) or []
        except Exception:
            return []

    def _atomic_write(self, obj):
        d = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False

    def append(self, entry):
        """
        Append a feedback entry (dict). Adds timestamp if missing.
        Returns True on success.
        """
        if not isinstance(entry, dict):
            return False
        entry = dict(entry)
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.datetime.now().isoformat()
        with self._lock:
            data = self._read_all()
            data.append(entry)
            return self._atomic_write(data)

    def all(self):
        """Return list of all feedback entries (new list)."""
        with self._lock:
            return list(self._read_all())

    def count(self):
        """Total number of stored entries."""
        with self._lock:
            return len(self._read_all())

    def count_effective(self):
        """
        Count 'effective' feedback rows suitable for training:
        nonzero feedback_value or reason == 'setpoint'.
        """
        with self._lock:
            rows = self._read_all()
        n = 0
        for r in rows:
            try:
                v = float(r.get("feedback_value", 0.0) or 0.0)
            except Exception:
                v = 0.0
            if abs(v) > 1e-9 or r.get("reason") == "setpoint":
                n += 1
        return n

    def replace_all(self, new_list):
        """Overwrite entire store (useful for tests)."""
        if not isinstance(new_list, list):
            return False
        with self._lock:
            return self._atomic_write(new_list)
