import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as real_datetime
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
    # De volledige Solcast dataset van de gebruiker
    raw_solcast_input = [
        ('00:00', 0, 0, 0), ('00:30', 0, 0, 0), ('01:00', 0, 0, 0), ('01:30', 0, 0, 0),
        ('02:00', 0, 0, 0), ('02:30', 0, 0, 0), ('03:00', 0, 0, 0), ('03:30', 0, 0, 0),
        ('04:00', 0, 0, 0), ('04:30', 0, 0, 0), ('05:00', 0, 0, 0), ('05:30', 0, 0, 0),
        ('06:00', 0, 0, 0), ('06:30', 0, 0, 0), ('07:00', 0, 0, 0), ('07:30', 0, 0, 0),
        ('08:00', 0, 0, 0), ('08:30', 0, 0, 0),
        ('09:00', 0.1656, 0.0665, 0.2336), ('09:30', 0.5298, 0.4606, 0.5551),
        ('10:00', 1.1404, 1.1404, 1.1404), ('10:30', 1.2884, 1.263, 1.3102),
        ('11:00', 1.359, 1.359, 1.359), ('11:30', 1.4001, 1.4001, 1.4001),
        ('12:00', 1.3473, 1.3299, 1.3609), ('12:30', 1.2953, 1.2305, 1.2953),
        ('13:00', 1.1715, 1.1715, 1.1715), ('13:30', 1.0324, 1.0236, 1.0356),
        ('14:00', 0.9023, 0.9023, 0.9023), ('14:30', 0.7277, 0.7277, 0.7277),
        ('15:00', 0.4729, 0.4555, 0.4768), ('15:30', 0.2565, 0.2565, 0.2565),
        ('16:00', 0.02, 0.02, 0.02), ('16:30', 0, 0, 0), ('17:00', 0, 0, 0),
        ('17:30', 0, 0, 0), ('18:00', 0, 0, 0), ('18:30', 0, 0, 0), ('19:00', 0, 0, 0),
        ('19:30', 0, 0, 0), ('20:00', 0, 0, 0), ('20:30', 0, 0, 0), ('21:00', 0, 0, 0),
        ('21:30', 0, 0, 0), ('22:00', 0, 0, 0), ('22:30', 0, 0, 0), ('23:00', 0, 0, 0),
        ('23:30', 0, 0, 0)
    ]

    forecast_payload = []
    simulation_date = "2025-12-28"

    for time_str, est, p10, p90 in raw_solcast_input:
        ts = datetime.strptime(f"{simulation_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        forecast_payload.append({
            "period_start": ts.isoformat(),
            "pv_estimate": est,
            "pv_estimate10": p10,
            "pv_estimate90": p90
        })


    # 2. Maak 1-minuut resolutie DataFrame (ZONDER waarschuwingen)
    df_fc = pd.DataFrame(forecast_payload)
    df_fc['timestamp'] = pd.to_datetime(df_fc['period_start'], utc=True)
    df_fc = df_fc.set_index('timestamp')

    # CRUCIALE FIX: Verwijder tekstkolommen voor interpolatie
    df_fc = df_fc.select_dtypes(include=[np.number])
    df_fc = df_fc.resample('1min').interpolate(method='linear')

    # 3. Genereer actuele PV waarden (Simulatie van de werkelijkheid)
    times = pd.date_range(f"{simulation_date} 07:00", f"{simulation_date} 17:00", freq="1min", tz=timezone.utc)
    actual_pv = []


    actual_pv = []
    for t in times:
        theoretical = df_fc.loc[t, 'pv_estimate'] if t in df_fc.index else 0.0
        # Bias curve: 0.35 -> 0.95
        if t.hour < 9: bias = 0.35
        elif 9 <= t.hour < 11: bias = 0.35 + ((t.hour-9)*60 + t.minute)/120 * 0.60
        else: bias = 0.95
        actual_pv.append(max(0, theoretical * bias + np.random.normal(0, 0.002)))



    return times, forecast_payload, actual_pv

# --- STAP 4: SIMULATIE RUNNER ---
def run_simulation():
    print(f"\n=== START SIMULATIE: Fixed User Scenario ===")
    mock_ha = MagicMock()
    states = {}
    mock_ha.get_state.side_effect = lambda e: states.get(e, "0.0")

    opts = {"system_max_kw": 2.0, "duration_hours": 1.0, "min_viable_kw": 0.3, "state_length": 1}
    ai = SolarAI(mock_ha, opts)

    times, forecast_payload, actual_pv_values = generate_scenario()

    # Zet de initiele forecast in de mock
    mock_ha.get_payload.return_value = {"attributes": {"detailedForecast": forecast_payload}}
    states[ai.entity_solcast_poll] = "2025-12-28T05:00:00Z"

    results = []

    print(f"{'Tijd':<8} | {'PV Act':<8} | {'Drempel':<8} | {'Bias':<5} | {'Status':<12}")
    print("-" * 55)

    for i, t in enumerate(times):
        with freeze_time(t):
            states[ai.entity_pv] = str(actual_pv_values[i] * 1000)
            ai.run_cycle()

            res = ai.last_stable_advice
            ctx = res["context"]

            results.append({
                "time": t, "pv": actual_pv_values[i], "status": res["action"].value,
                "threshold": ctx.trigger_threshold_kw if ctx else 0,
                "bias": ctx.bias_factor if ctx else 1.0
            })

            if t.minute % 10 == 0:
                print(f"{t.strftime('%H:%M'):<8} | {actual_pv_values[i]:>6.2f}k | {results[-1]['threshold']:>6.2f}k | {results[-1]['bias']:>4.2f} | {results[-1]['status']:<12}")

    # --- PLOT ---
    df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['pv'], label='Actueel PV (kW)', color='orange', lw=2)
    plt.plot(df['time'], df['threshold'], label='Trigger Drempel (kW)', color='red', linestyle='--')

    # Highlight de START zones
    plt.fill_between(df['time'], 0, df['pv'].max(), where=(df['status'] == 'START'), color='green', alpha=0.1, label='START Signaal')

    plt.title("Simulatie met Jouw Solcast Data")
    plt.ylabel("kW")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_simulation()