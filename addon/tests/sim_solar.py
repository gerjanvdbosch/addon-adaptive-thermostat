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

os.environ["TZ"] = "UTC"

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- STAP 2: MOCK MODULES ---
mock_db = MagicMock()
mock_db.fetch_solar_training_data_orm.return_value = pd.DataFrame()
mock_db.upsert_solar_record.return_value = None
sys.modules["db"] = mock_db
sys.modules["ha_client"] = MagicMock()

# Importeer nu pas de AI klasse en de module
import solar
from solar import SolarAI, SolarStatus

# --- STAP 3: DE ULTIEME TIJD-MACHINE ---
# We maken een echte subclass van datetime zodat .astimezone() werkt
class MockDateTime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        # Freezegun geeft een naive tijd. We maken deze geforceerd UTC aware.
        t = real_datetime.datetime.now()
        if t.tzinfo is None:
            t = t.replace(tzinfo=real_datetime.timezone.utc)

        if tz:
            return t.astimezone(tz)
        # Voor de regel 'local_tz = datetime.now().astimezone().tzinfo'
        # moeten we een object teruggeven dat .astimezone() snapt.
        return t

# Overschrijf de datetime klasse in de solar module
solar.datetime = MockDateTime

# Overschrijf pd.Timestamp.now om freeze_time te volgen en ALTIJD aware te zijn
def patched_timestamp_now(tz=None):
    # 1. Pak de bevroren tijd
    t = real_datetime.datetime.now()
    # 2. Maak Pandas object
    ts = pd.Timestamp(t)
    # 3. Als hij naive is (geen TZ), dwing hem naar UTC
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')

    # 4. Converteer naar gevraagde TZ (zoals "UTC" in jouw AI code)
    if tz:
        # Als er een tijdzone gevraagd wordt (zoals tz="UTC" in de AI), converteer dan
        return ts.tz_convert(tz)
    return ts

solar.pd.Timestamp.now = patched_timestamp_now

# --- STAP 4: MOCK HOME ASSISTANT ---
class MockHA:
    def __init__(self):
        self.states = {}
        self.payloads = {}
        self.last_prediction = None

    def get_state(self, entity):
        return self.states.get(entity, "0.0")

    def get_payload(self, entity):
        return self.payloads.get(entity, {})

    def get_state_attributes(self, entity):
        return {"elevation": 25.0}

    def set_solar_prediction(self, state, attrs):
        self.last_prediction = {"state": state, "attrs": attrs}

# --- STAP 5: SCENARIO GENERATOR ---
def generate_scenario():
    # Jouw data
    raw_solcast = {
        "09:00": 0.1656, "09:30": 0.5298, "10:00": 0.6602, "10:30": 0.8152,
        "11:00": 0.8640, "11:30": 0.8648, "12:00": 0.8125, "12:30": 0.7445,
        "13:00": 0.6474, "13:30": 0.5488, "14:00": 0.4338, "14:30": 0.3014,
        "15:00": 0.1540, "15:30": 0.0580, "16:00": 0.0072
    }

    forecast_payload = []
    for h in range(24):
        for m in [0, 30]:
            # We maken de forecast timestamps in UTC
            ts = datetime(2025, 12, 28, h, m, tzinfo=timezone.utc)
            key = f"{(h+1)%24:02d}:{m:02d}"
            val = raw_solcast.get(key, 0.0)
            forecast_payload.append({
                "period_start": ts.isoformat(),
                "pv_estimate": val,
                "pv_estimate10": val * 0.8,
                "pv_estimate90": val * 1.2
            })

    times = pd.date_range("2025-12-28 06:00", "2025-12-28 16:00", freq="1min", tz=timezone.utc)

    df_fc = pd.DataFrame(forecast_payload)
    df_fc['timestamp'] = pd.to_datetime(df_fc['period_start'], utc=True)
    df_fc = df_fc.set_index('timestamp').select_dtypes(include=[np.number]).resample('1min').interpolate(method='linear')

    actual_pv = []
    for t in times:
        theoretical = df_fc.loc[t, 'pv_estimate'] if t in df_fc.index else 0.0
        # Bias curve: 0.35 -> 0.95
        if t.hour < 9: bias = 0.35
        elif 9 <= t.hour < 11: bias = 0.35 + ((t.hour-9)*60 + t.minute)/120 * 0.60
        else: bias = 0.95
        actual_pv.append(max(0, theoretical * bias + np.random.normal(0, 0.002)))

    return times, forecast_payload, actual_pv

# --- STAP 6: RUNNER ---
def run_simulation():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    mock_ha = MockHA()
    opts = {"system_max_kw": 2.0, "duration_hours": 1.0, "min_viable_kw": 0.2}

    ai = SolarAI(mock_ha, opts)
    ai.state_len = 1

    times, forecast_payload, actual_pv_values = generate_scenario()
    mock_ha.payloads[ai.entity_solcast] = {"attributes": {"detailedForecast": forecast_payload}}
    mock_ha.states[ai.entity_solcast_poll] = "2025-12-28T05:00:00Z"

    results = []
    print(f"\n{'Tijd (UTC)':<10} | {'PV Act':<8} | {'Drempel':<8} | {'Bias':<5} | {'Rest':<5} | {'Status':<12}")
    print("-" * 75)

    for i, t in enumerate(times):
        # Freeze_time dwingt datetime.now() naar de simulatietijd
        with freeze_time(t):
            mock_ha.states[ai.entity_pv] = str(actual_pv_values[i] * 1000)
            ai.run_cycle()

            pred = mock_ha.last_prediction
            if pred:
                attrs = pred["attrs"]
                res = {
                    "time": t, "pv": actual_pv_values[i], "status": attrs.get("status"),
                    "threshold": attrs.get("trigger_threshold_kw", 0.0),
                    "bias": attrs.get("bias_factor", 1.0),
                    "rest": attrs.get("remaining_day_pct", 0)
                }
                results.append(res)
                if t.minute % 30 == 0:
                    print(f"{t.strftime('%H:%M'):<10} | {res['pv']:>6.2f}k | {res['threshold']:>6.2f}k | {res['bias']:>4.2f} | {res['rest']:>3}% | {res['status']:<12}")

    # Plot
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df_res['time'], df_res['pv'], label='PV Actueel', color='orange')
        plt.plot(df_res['time'], df_res['threshold'], label='Drempel', color='red', linestyle='--')
        plt.fill_between(df_res['time'], 0, 1.5, where=(df_res['status'] == 'START'), color='green', alpha=0.1)
        plt.title("SolarAI Simulatie (Final Fix)")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    run_simulation()