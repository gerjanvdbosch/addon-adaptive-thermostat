import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as real_datetime
import json
import matplotlib.dates as mdates

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
def generate_scenario(simulation_date, solar_ai_instance):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "solar.json")

    with open(json_path, "r") as f:
        solar_data = json.load(f)

    df_source = pd.DataFrame(solar_data)
    df_source['timestamp'] = pd.to_datetime(df_source['timestamp'], utc=True)
    df_source = df_source.set_index('timestamp').sort_index()

    # 1-minuut range maken
    start_time = pd.to_datetime(f"{simulation_date} 07:00").tz_localize(timezone.utc)
    end_time = pd.to_datetime(f"{simulation_date} 17:00").tz_localize(timezone.utc)
    new_times = pd.date_range(start=start_time, end=end_time, freq="1min", tz=timezone.utc)

    # --- AI LOGICA SIMULATIE ---
    # We berekenen de AI power voordat we resamplen (zoals in de SolarAI class)
    if solar_ai_instance.is_fitted:
        X_pred = solar_ai_instance._create_features(df_source.reset_index())
        pred_ai = solar_ai_instance.model.predict(X_pred)
        # Blending: 60% AI, 40% Solcast
        df_source["ai_power_raw"] = (solar_ai_instance.ml_weight * pred_ai) + (solar_ai_instance.solcast_weight * df_source["pv_estimate"])
    else:
        df_source["ai_power_raw"] = df_source["pv_estimate"]

    # Pas de huidige bias van de AI-klasse toe
    df_source["ai_power"] = (df_source["ai_power_raw"] * solar_ai_instance.smoothed_bias).clip(0, solar_ai_instance.system_max_kw)

    # Resample en Interpoleren naar 1 minuut
    df_numeric = df_source.select_dtypes(include=[np.number])
    df_resampled = df_numeric.reindex(df_source.index.union(new_times)).interpolate(method='linear')
    df_final = df_resampled.loc[new_times]

    # Data klaarmaken voor plot
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

    opts = {"state_length": 1}
    ai = SolarAI(mock_ha, opts)

    # FIX 1: Unpack alle 6 waarden die de generator nu teruggeeft
    times, actual_pv_values, forecast_values, p10_values, p90_values, ai_pred_values = generate_scenario("2025-12-28", ai)

    # FIX 2: Bouw de forecast_payload op die de SolarAI klasse verwacht (lijst van dicts)
    # Dit is nodig omdat ai.run_cycle() intern _update_solcast_cache aanroept
    forecast_payload = []
    for i, t in enumerate(times):
        forecast_payload.append({
            "period_start": t.isoformat(),
            "pv_estimate": forecast_values[i],
            "pv_estimate10": p10_values[i],
            "pv_estimate90": p90_values[i]
        })

    # Zet de initiele forecast in de mock
    mock_ha.get_payload.return_value = {"attributes": {"detailedForecast": forecast_payload}}
    states[ai.entity_solcast_poll] = "2025-12-28T05:00:00Z"

    results = []

    # Header uitlijning
    print(f"{'Tijd':<8} | {'PV Act':<8} | {'AI Pred':<8} | {'Drempel':<8} | {'Bias':<4} | {'Status':<12} | {'Drempel %':<8}")
    print("-" * 85)

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
                "ai_pred": ai_pred_values[i], # Nieuwe AI lijn
                "status": res["action"].value,
                "threshold": ctx.trigger_threshold_kw if ctx else 0.0,
                "bias": ctx.bias_factor if ctx else 1.0,
                "pct": (ctx.threshold_percentage * 100) if ctx else 0.0,
            }
            results.append(row)

            # Log elke 10 minuten
            if t.minute % 10 == 0:
                print(f"{t.strftime('%H:%M'):<8} | {row['pv']:>6.2f}kW | {row['ai_pred']:>6.2f}kW | {row['threshold']:>6.2f}kW | {row['bias']:>4.2f} | {row['status']:<12} | {row['pct']:>6.1f}%")

    # --- PLOT ---
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Solcast Onzekerheidsbanden (Blauw)
    ax.plot(times, p10_values, label='Solcast Forecast P10 (kW)', color='#3399FF', linestyle=':', alpha=0.6)
    ax.plot(times, p90_values, label='Solcast Forecast P90 (kW)', color='#3399FF', linestyle=':', alpha=0.6)
    ax.fill_between(times, p10_values, p90_values, color='#3399FF', alpha=0.05, label='Onzekerheidsmarge')

    # Hoofdlijnen
    ax.plot(df['time'], df['pv'], label='Actueel PV (kW)', color='orange', lw=2)
    ax.plot(df['time'], df['forecast'], label='Solcast Forecast (kW)', color='#004080', linestyle='--', alpha=0.7, lw=1.5)

    # De SolarAI Lijn
    ax.plot(df['time'], df['ai_pred'], label='SolarAI Prediction', color='#004080', lw=1.5)

    # Drempel
    ax.plot(df['time'], df['threshold'], label='Trigger Drempel', color='red', linestyle='--')

    # START zones inkleuren
    ax.fill_between(df['time'], 0, df['pv'].max(), where=(df['status'] == 'START'), color='green', alpha=0.1, label='Signaal: START')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Toon HH:MM
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Zet een streepje om het uur

    plt.title(f"Simulatie SolarAI: {times[0].strftime('%Y-%m-%d')}")
    plt.ylabel("Vermogen (kW)")
    plt.xlabel("Tijd")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()