import math
import pandas as pd
import numpy as np


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


def safe_bool(val):
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    try:
        f = float(val)
        return f != 0.0
    except Exception:
        s = str(val).strip().lower()
        if s in ("true", "1", "yes", "y", "on", "aan"):
            return True
    return False


def safe_bool_to_float(val):
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        f = float(val)
        return 1.0 if f != 0.0 else 0.0
    except Exception:
        s = str(val).strip().lower()
        if s in ("true", "1", "yes", "y", "on", "aan"):
            return 1.0
    return 0.0


def cyclical_hour(ts):
    h = ts.hour + ts.minute / 60.0
    return math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0)


def cyclical_day(ts):
    d = ts.weekday()
    return math.sin(2 * math.pi * d / 7.0), math.cos(2 * math.pi * d / 7.0)


def cyclical_doy(ts):
    doy = ts.timetuple().tm_yday
    doy_sin = math.sin(2 * math.pi * doy / 366.0)
    doy_cos = math.cos(2 * math.pi * doy / 366.0)
    return doy_sin, doy_cos


def encode_wind(degrees):
    if degrees is None:
        return 0.0, 0.0
    try:
        rad = math.radians(float(degrees) % 360)
        return math.sin(rad), math.cos(rad)
    except Exception:
        return 0.0, 0.0


def add_cyclic_time_features(df: pd.DataFrame, col_name="timestamp") -> pd.DataFrame:
    """
    Voegt cyclische tijd-features toe (hour, day, doy) als sin/cos paren.
    Neemt minuten mee voor hogere precisie.
    """
    if df is None or col_name not in df.columns:
        return df

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        df[col_name] = pd.to_datetime(df[col_name])

    dt = df[col_name].dt

    # 1. Tijd van de dag (0..24 uur)
    # We voegen minuten toe voor precisie (bv. 14:30 wordt 14.5)
    precise_hour = dt.hour + (dt.minute / 60.0)

    df["hour_sin"] = np.sin(2 * np.pi * precise_hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * precise_hour / 24.0)

    # 2. Dag van de week (0..6, Maandag=0)
    df["day_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7.0)

    # 3. Dag van het jaar (1..366) - Seizoenen
    df["doy_sin"] = np.sin(2 * np.pi * dt.dayofyear / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dayofyear / 366.0)

    return df
