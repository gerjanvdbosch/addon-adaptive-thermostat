def round_half(x):
    return round(x * 2) / 2

def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None
