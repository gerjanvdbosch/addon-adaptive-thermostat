import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as real_datetime
import json
import matplotlib.dates as mdates
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
logger = logging.getLogger("solar2")
logger.setLevel(logging.WARNING)

# --- STAP 1: MOCK DB & MODULES ---
mock_db = MagicMock()
mock_db.fetch_solar_training_data_orm.return_value = pd.DataFrame()  # Geen historische data
mock_db.upsert_solar_record = MagicMock()
sys.modules["db"] = mock_db
sys.modules["ha_client"] = MagicMock()

import solar
from solar import Solar, SolarStatus, NowCaster

# --- STAP 2: TIJD PATCHES ---
class MockDateTime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        t = real_datetime.datetime.now()
        if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(tz) if tz else t

solar2.datetime = MockDateTime

def patched_timestamp_now(tz=None):
    t = real_datetime.datetime.now()
    ts = pd.Timestamp(t)
    if ts.tzinfo is None: ts = ts.tz_localize('UTC')
    return ts.tz_convert(tz) if tz else ts

solar2.pd.Timestamp.now = patched_timestamp_now

# --- STAP 3: SCENARIO GENERATOR ---
def generate_scenario(simulation_date, solar_instance):
    json_path = os.path.join(current_dir, "solar.json")
    with open(json_path, "r") as f:
        solar_data = json.load(f)

    df_source = pd.DataFrame(solar_data)
    df_source['timestamp'] = pd.to_datetime(df_source['timestamp'], utc=True)
    target_day = pd.to_datetime(simulation_date).date()
    df_source = df_source[df_source['timestamp'].dt.date == target_day].copy()
    df_source['actual_pv_yield'] = df_source['actual_pv_yield'].fillna(0.0)
    df_source = df_source.set_index('timestamp').sort_index()

    start_time = pd.to_datetime(f"{simulation_date} 07:00").tz_localize(timezone.utc)
    end_time = pd.to_datetime(f"{simulation_date} 17:00").tz_localize(timezone.utc)
    new_times = pd.date_range(start=start_time, end=end_time, freq="1min", tz=timezone.utc)

    # --- AI LOGICA SIMULATIE ---
    df_source["ai_power"] = df_source["pv_estimate90"]  # fallback
    if solar_instance.model.is_fitted:
        X_pred = solar_instance.model._prepare_features(df_source.reset_index())
        pred_ai = solar_instance.model.predict(df_source)
        df_source["ai_power"] = (0.4*df_source["pv_estimate90"] + 0.6*pred_ai).clip(0, solar_instance.system_max)

    df_resampled = df_source.reindex(df_source.index.union(new_times)).interpolate(method='linear')
    df_final = df_resampled.loc[new_times]
    df_final["power_corrected"] = df_final["ai_power"]

    times = df_final.index
    actual_pv = df_final['actual_pv_yield'].tolist()
    forecast_pv = df_final['pv_estimate'].tolist()
    p10 = df_final['pv_estimate10'].tolist()
    p90 = df_final['pv_estimate90'].tolist()
    ai_prediction = df_final['ai_power'].tolist()

    return times, actual_pv, forecast_pv, p10, p90, ai_prediction

# --- STAP 4: SIMULATIE RUNNER ---
def run_simulation():
    print(f"\n=== START SIMULATIE: Fixed User Scenario ===")
    mock_ha = MagicMock()
    states = {}
    mock_ha.get_state.side_effect = lambda e: states.get(e, "0.0")

    opts = {"system_max_kw": 4.0, "duration_hours": 1.0}
    ai = Solar(mock_ha, opts)

    times, actual_pv_values, forecast_values, p10_values, p90_values, ai_pred_values = generate_scenario("2025-12-28", ai)

    forecast_payload = []
    for i, t in enumerate(times):
        forecast_payload.append({
            "period_start": t.isoformat(),
            "pv_estimate": forecast_values[i],
            "pv_estimate10": p10_values[i],
            "pv_estimate90": p90_values[i],
            "power_corrected": ai_pred_values[i]  # deze kolom toevoegen
        })
    mock_ha.get_payload.return_value = {"attributes": {"detailedForecast": forecast_payload}}
    states[ai.opts.get("sensor_solcast_poll", "sensor_solcast_pv_forecast_api_last_polled")] = "2025-12-28T05:00:00Z"

    results = []

    print(f"{'Tijd':<8} | {'PV Act':<8} | {'AI Pred':<8} | {'Status':<12}")
    print("-" * 50)

    for i, t in enumerate(times):
        with freeze_time(t):
            states[ai.opts.get("sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual")] = str(actual_pv_values[i] * 1000)
            ai.run_cycle()

            # context ophalen uit optimizer
            status = SolarStatus.DONE
            ctx = None
            if hasattr(ai, "optimizer") and ai.cached_forecast is not None:
                status, ctx = ai.optimizer.calculate_optimal_window(ai.cached_forecast, t, 100)

            row = {
                "time": t,
                "pv": actual_pv_values[i],
                "forecast": forecast_values[i],
                "ai_pred": ai_pred_values[i],
                "status": status.value,
            }
            results.append(row)

            if t.minute % 10 == 0:
                print(f"{t.strftime('%H:%M'):<8} | {row['pv']:>6.2f}kW | {row['ai_pred']:>6.2f}kW | {row['status']:<12}")

    # --- PLOT ---
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(times, p10_values, label='Solcast P10', color='#3399FF', linestyle=':', alpha=0.6)
    ax.plot(times, p90_values, label='Solcast P90', color='#3399FF', linestyle=':', alpha=0.6)
    ax.fill_between(times, p10_values, p90_values, color='#3399FF', alpha=0.05, label='Onzekerheidsmarge')

    ax.plot(df['time'], df['pv'], label='Actueel PV (kW)', color='orange', lw=2)
    ax.plot(df['time'], df['forecast'], label='Solcast Forecast (kW)', color='#004080', linestyle='--', alpha=0.7, lw=1.5)
    ax.plot(df['time'], df['ai_pred'], label='Solar Prediction', color='#004080', lw=1.5)

    ax.fill_between(df['time'], 0, df['pv'].max(), where=(df['status'] == 'START'), color='green', alpha=0.1, label='Signaal: START')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.title(f"Simulatie Solar: {times[0].strftime('%Y-%m-%d')}")
    plt.ylabel("Vermogen (kW)")
    plt.xlabel("Tijd")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
