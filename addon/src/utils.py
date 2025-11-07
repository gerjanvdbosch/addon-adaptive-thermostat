def round_half(x):
    return round(x * 2) / 2

def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def cyclical_hour(ts):
    h = ts.hour + ts.minute / 60.0
    return math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0)

def cyclical_day(ts):
    d = ts.weekday()
    return math.sin(2 * math.pi * d / 7.0), math.cos(2 * math.pi * d / 7.0)

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
        return None
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
    return None
