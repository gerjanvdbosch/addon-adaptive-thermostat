import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as real_datetime
import json

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from freezegun import freeze_time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- CONFIGURATIE ---
os.environ["TZ"] = "UTC"
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("solar")
logger.setLevel(logging.WARNING)

# --- STAP 1: MOCK DB & MODULES ---
mock_db = MagicMock()
mock_db.fetch_solar_training_data_orm.return_value = pd.DataFrame() # Geen historische data voor deze test
sys.modules["db"] = mock_db
sys.modules["ha_client"] = MagicMock()

import solar
from solar import SolarAI, SolarStatus

# --- STAP 2: TIJD PATCHES (Bulletproof) ---
class MockDateTime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        t = real_datetime.datetime.now()
        if t.tzinfo is None: t = t.replace(tzinfo=real_datetime.timezone.utc)
        return t.astimezone(tz) if tz else t

solar.datetime = MockDateTime

def patched_timestamp_now(tz=None):
    t = real_datetime.datetime.now()
    ts = pd.Timestamp(t)
    if ts.tzinfo is None: ts = ts.tz_localize('UTC')
    return ts.tz_convert(tz) if tz else ts

solar.pd.Timestamp.now = patched_timestamp_now

# --- STAP 3: SCENARIO GENERATOR ---
def generate_scenario():
    simulation_date = "2025-12-28"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "solar.json")

    with open(json_path, "r") as f:
        solar_data = json.load(f)

    # 1. Inladen in DataFrame en index zetten op timestamp
    df_source = pd.DataFrame(solar_data)
    df_source['timestamp'] = pd.to_datetime(df_source['timestamp'], utc=True)
    df_source = df_source.set_index('timestamp').sort_index()

    # 2. Definieer de nieuwe tijdreeks (1 minuut interval)
    start_time = pd.to_datetime(f"{simulation_date} 07:00").tz_localize(timezone.utc)
    end_time = pd.to_datetime(f"{simulation_date} 17:00").tz_localize(timezone.utc)
    new_times = pd.date_range(start=start_time, end=end_time, freq="1min", tz=timezone.utc)

    # 3. Reindex en Interpoleren
    # We combineren de oude index met de nieuwe index om gaten op te vullen
    combined_index = df_source.index.union(new_times)
    df_resampled = df_source.reindex(combined_index)

    # Lineaire interpolatie om de minuten tussen de bekende meetpunten te vullen
    df_resampled = df_resampled.interpolate(method='linear')

    # Selecteer nu alleen de minuten die we daadwerkelijk nodig hebben
    df_final = df_resampled.loc[new_times]

    # 4. Data omzetten naar de gewenste formaten
    times = df_final.index
    actual_pv = df_final['actual_pv_yield'].tolist()
    forecast_pv = df_final['pv_estimate'].tolist()

    # Payload voor SolarAI
    forecast_payload = []
    for ts, row in df_final.iterrows():
        forecast_payload.append({
            "period_start": ts.isoformat(),
            "pv_estimate": row['pv_estimate'],
            "pv_estimate10": row.get('pv_estimate10', row['pv_estimate'] * 0.9),
            "pv_estimate90": row.get('pv_estimate90', row['pv_estimate'] * 1.1)
        })

    return times, forecast_payload, actual_pv, forecast_pv

# --- STAP 4: SIMULATIE RUNNER ---
def run_simulation():
    print(f"\n=== START SIMULATIE: Fixed User Scenario ===")
    mock_ha = MagicMock()
    states = {}
    mock_ha.get_state.side_effect = lambda e: states.get(e, "0.0")

    opts = {"system_max_kw": 2.0, "duration_hours": 1.0, "min_viable_kw": 0.3, "state_length": 1}
    ai = SolarAI(mock_ha, opts)

    times, forecast_payload, actual_pv_values, forecast_values = generate_scenario()

    # Zet de initiele forecast in de mock
    mock_ha.get_payload.return_value = {"attributes": {"detailedForecast": forecast_payload}}
    states[ai.entity_solcast_poll] = "2025-12-28T05:00:00Z"

    results = []

    # Header uitlijning met alle kolommen
    print(f"{'Tijd':<8} | {'PV Act':<8} | {'Forecast':<8} | {'Drempel':<8} | {'Bias':<4} | {'Status':<12} | {'Drempel %':<8}")
    print("-" * 75)

    for i, t in enumerate(times):
        with freeze_time(t):
            states[ai.entity_pv] = str(actual_pv_values[i] * 1000)
            ai.run_cycle()

            res = ai.last_stable_advice
            ctx = res["context"]

            # Data opslaan voor plot en logging
            row = {
                "time": t,
                "pv": actual_pv_values[i],
                "forecast": forecast_values[i],
                "status": res["action"].value,
                "threshold": ctx.trigger_threshold_kw if ctx else 0.0,
                "bias": ctx.bias_factor if ctx else 1.0,
                "pct": (ctx.threshold_percentage * 100) if ctx else 0.0,
            }
            results.append(row)

            # Log elke 10 minuten
            if t.minute % 10 == 0:
                print(f"{t.strftime('%H:%M'):<8} | {row['pv']:>6.2f}kW | {row['forecast']:>6.2f}kW | {row['threshold']:>6.2f}kW | {row['bias']:>4.2f} | {row['status']:<12} | {row['pct']:>6.1f}%")

    # --- PLOT ---
    p10 = [d['pv_estimate10'] for d in forecast_payload]
    p90 = [d['pv_estimate90'] for d in forecast_payload]

    df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))

    # Lijnen
    plt.plot(df['time'], p10, label='Solcast Forecast p10 (kW)', color='#B3D9FF', linestyle=':', alpha=0.6)
    plt.plot(df['time'], p90, label='Solcast Forecast p90 (kW)', color='#3399FF', linestyle=':', alpha=0.6)

    plt.plot(df['time'], df['pv'], label='Actueel PV (kW)', color='orange', lw=2)
    plt.plot(df['time'], df['forecast'], label='Solcast Forecast (kW)', color='#004080', linestyle='--', alpha=0.7, lw=1.5)
    plt.plot(df['time'], df['threshold'], label='Trigger Drempel (kW)', color='red', linestyle='--')

    # START zones inkleuren
    plt.fill_between(df['time'], 0, df['forecast'].max(), where=(df['status'] == 'START'), color='green', alpha=0.1, label='START Signaal')
    plt.fill_between(times, p10, p90, color='#3399FF', alpha=0.1, label='Onzekerheidsmarge')

    plt.title("Simulatie: Forecast vs Werkelijkheid")
    plt.ylabel("kW")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()