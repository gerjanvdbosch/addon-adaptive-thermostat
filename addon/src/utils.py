import math

from datetime import datetime


def round_half(x):
    return round(x * 2) / 2


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_round(v, digits=1):
    try:
        return round(float(v), digits)
    except Exception:
        return None


def cyclical_hour(ts):
    h = ts.hour + ts.minute / 60.0
    return math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0)


def cyclical_day(ts):
    d = ts.weekday()
    return math.sin(2 * math.pi * d / 7.0), math.cos(2 * math.pi * d / 7.0)


def cyclical_month(ts):
    m = ts.month - 1  # 0..11
    return math.sin(2 * math.pi * m / 12.0), math.cos(2 * math.pi * m / 12.0)


def day_or_night(ts=None):
    if ts is None:
        ts = datetime.now()
    h = ts.hour
    if 7 <= h < 22:
        return 0  # day
    return 1  # night


def month_to_season(ts):
    m = ts.month
    if m in (12, 1, 2):
        return 0  # Dec-Feb=winter
    if m in (3, 4, 5):
        return 1  # Mar-May=spring
    if m in (6, 7, 8):
        return 2  # Jun-Aug=summer
    return 3  # Sep-Nov=autumn


def encode_wind(degrees):
    if degrees is None:
        return 0.0, 0.0
    try:
        rad = math.radians(float(degrees) % 360)
        return math.sin(rad), math.cos(rad)
    except Exception:
        return 0.0, 0.0


def encode_binary_onoff(val):
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        f = float(val)
        return 1.0 if f != 0.0 else 0.0
    except Exception:
        s = str(val).strip().lower()
        if s in ("on", "aan", "true", "1", "yes", "y"):
            return 1.0
        if s in ("off", "uit", "false", "0", "no", "n"):
            return 0.0
    return 0.0
