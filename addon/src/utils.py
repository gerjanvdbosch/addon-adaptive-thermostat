import numpy as np
import pandas as pd

from datetime import datetime


def add_cyclic_time_features(df: pd.DataFrame, col_name="timestamp") -> pd.DataFrame:
    """
    Voegt cyclische tijd-features toe (hour, day, doy) als sin/cos paren.
    Neemt minuten mee voor hogere precisie.
    """
    if df is None or col_name not in df.columns:
        return df

    tz = datetime.now().astimezone().tzinfo

    df = df.copy()
    df[col_name] = pd.to_datetime(df[col_name])
    dt = df[col_name].dt.tz_convert(tz)

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
