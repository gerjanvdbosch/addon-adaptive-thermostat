import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from solar import (  # noqa: E402
    SolarForecaster,
    SolarModel,
    NowCaster,
    SolarOptimizer,
)


def mock_module(name, attributes):
    mock = ModuleType(name)
    for key, val in attributes.items():
        setattr(mock, key, val)
    sys.modules[name] = mock
    return mock


# --- MOCKS VOOR ONTBREKENDE IMPORTS ---
class MockLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")

    def warning(self, msg):
        print(f"[WARN] {msg}")


logger = MockLogger()
mock_module("logger", {"logger": MockLogger()})
mock_module("config", {"Config": object})
mock_module("context", {"Context": object})


def add_cyclic_time_features(df, col_name="timestamp"):
    df = df.copy()
    hours = df[col_name].dt.hour + df[col_name].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    doy = df[col_name].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return df


class MockConfig:
    def __init__(self):
        self.pv_max_kw = 2.0
        self.dhw_duration_hours = 1.0
        self.min_kwh_threshold = 0.3
        self.avg_baseload_kw = 0.15
        self.solar_model_path = "solar_model.joblib"
        self.solar_model_ratio = 1


class MockContext:
    def __init__(self):
        self.now = None
        self.forecast_df = None
        self.stable_pv = 0
        self.stable_load = 0
        self.pv_max_kw = 0


# --- DATA GENERATOR ---


def generate_synthetic_forecast(start_time):
    """Genereert een 24-uurs forecast dataframe."""
    periods = 96  # 15 min intervals
    times = [start_time + timedelta(minutes=i * 15) for i in range(periods)]

    df = pd.DataFrame({"timestamp": times})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Simuleer een zonne-curve (sinus)
    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    # Zon tussen 08:00 en 20:00
    solar_curve = np.sin(np.pi * (hours - 8) / 12).clip(lower=0)

    df["pv_estimate"] = solar_curve * 2.0  # Max 2kW
    df["pv_estimate10"] = df["pv_estimate"] * 0.8
    df["pv_estimate90"] = df["pv_estimate"] * 1.2

    # Weather features
    df["temp"] = 15 + solar_curve * 10
    df["wind"] = 3.0
    df["cloud"] = 20.0
    df["radiation"] = solar_curve * 800
    df["diffuse"] = 100.0
    df["irradiance"] = df["radiation"]

    return df


# --- DE SIMULATIE ---


def plot_results(ctx, forecaster, decision):
    df = ctx.forecast_df.copy()

    # 1. Bereken de gecorrigeerde solar data
    # FIX: Pak expliciet de 'prediction' kolom uit het resultaat
    preds = forecaster.model.predict(df)
    df["power_ml"] = preds["prediction"]

    # Pas nowcaster toe
    df["power_corrected"] = forecaster.nowcaster.apply(df, ctx.now, "power_ml")

    # 2. Bereken de Load Projectie (gelijk aan de optimizer logica)
    baseload = forecaster.optimizer.avg_baseload
    df["consumption"] = baseload

    # Alleen voor de punten vanaf 'nu' passen we de decay toe
    future_mask = df["timestamp"] >= ctx.now
    future_indices = df.index[future_mask]

    decay_steps = 2
    for i, idx in enumerate(future_indices[: decay_steps + 1]):
        factor = 1.0 - (i / decay_steps)
        blended_load = (ctx.stable_load * factor) + (baseload * (1 - factor))
        df.at[idx, "consumption"] = max(blended_load, baseload)

    # Voor de historie (vóór nu) zetten we de load even op de laatst bekende stable_load
    df.loc[df["timestamp"] < ctx.now, "consumption"] = ctx.stable_load

    # 3. Bereken Netto Power
    df["net_power"] = (df["power_corrected"] - df["consumption"]).clip(lower=0)

    # --- PLOTTEN ---
    plt.figure(figsize=(14, 7))

    # Solar lijnen
    plt.plot(
        df["timestamp"],
        df["pv_estimate"],
        "--",
        label="Raw Solcast (Forecast)",
        color="gray",
        alpha=0.4,
    )
    plt.plot(
        df["timestamp"],
        df["power_ml"],
        ":",
        label="ML Model Output",
        color="blue",
        alpha=0.6,
    )
    plt.plot(
        df["timestamp"],
        df["power_corrected"],
        "g-",
        linewidth=2,
        label="Corrected Solar (Nowcast)",
    )

    # Huidige PV Meting
    x_min_plot = df["timestamp"].min()
    plt.hlines(
        y=ctx.stable_pv,
        xmin=x_min_plot,
        xmax=ctx.now,
        color="darkgreen",
        linestyle=":",
        alpha=0.4,
        linewidth=1,
    )
    plt.scatter(
        ctx.now,
        ctx.stable_pv,
        color="darkgreen",
        s=120,
        edgecolors="white",
        linewidths=1.5,
        zorder=15,
        label=f"Actuele PV ({ctx.stable_pv:.2f} kW)",
    )

    # Load lijn
    plt.step(
        df["timestamp"],
        df["consumption"],
        where="post",
        color="red",
        linewidth=2,
        label="Load Projectie",
    )

    # Netto area
    plt.fill_between(
        df["timestamp"],
        0,
        df["net_power"],
        color="green",
        alpha=0.15,
        label="Netto Beschikbaar",
    )

    # Markeringen
    plt.axvline(ctx.now, color="black", linestyle="-", alpha=0.5)
    plt.text(ctx.now, plt.ylim()[1] * 0.95, " NU", color="black", fontweight="bold")

    if decision and decision.planned_start:
        plt.axvline(
            decision.planned_start,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Geplande Start",
        )
        duration_end = decision.planned_start + timedelta(
            hours=forecaster.optimizer.duration
        )
        plt.axvspan(
            decision.planned_start,
            duration_end,
            color="orange",
            alpha=0.1,
            label="DHW Run Window",
        )

    plt.title(
        f"Solar & Load Simulatie\nActie: {decision.action.value} - {decision.reason}",
        fontsize=12,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()


def run_simulation():
    # 1. Setup
    cfg = MockConfig()
    ctx = MockContext()

    sim_now = datetime.now(timezone.utc).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    ctx.now = sim_now
    ctx.forecast_df = generate_synthetic_forecast(sim_now.replace(hour=0))

    ctx.stable_pv = 0.1
    ctx.stable_load = 2.0

    # 2. Initialiseer klassen (MET CORRECTE MOCK)
    class SimulatedSolarModel(SolarModel):
        def _load(self):
            self.is_fitted = True
            self.mae = 0.15

        def predict(self, df, model_ratio: float = 0.7):
            val = df["pv_estimate"] * 0.95
            # FIX: Return DataFrame zoals de echte code verwacht
            return pd.DataFrame({"prediction": val, "prediction_raw": val})

    forecaster = SolarForecaster(cfg, ctx)
    forecaster.model = SimulatedSolarModel(Path("mock.joblib"))
    forecaster.nowcaster = NowCaster(0.15, cfg.pv_max_kw)

    # 3. Voer de analyse uit
    print("\n" + "=" * 50)
    print(f"SOLAR SIMULATIE - TIJD: {sim_now.strftime('%H:%M')} UTC")
    print("=" * 50)

    status, decision = forecaster.analyze(ctx.now, ctx.stable_load)

    # 4. Rapportage en Plot
    if decision:
        print(f"ACTIE:          {decision.action.value}")
        print(f"REDEN:          {decision.reason}")
        try:
            plot_results(ctx, forecaster, decision)
        except Exception as e:
            print(f"Kon grafiek niet tekenen: {e}")
            import traceback

            traceback.print_exc()


def run_time_travel_simulation():
    cfg = MockConfig()
    ctx = MockContext()

    # 1. Genereer forecast voor de hele dag (startend vanaf middernacht UTC)
    start_of_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    ctx.forecast_df = generate_synthetic_forecast(start_of_day)
    ctx.pv_max_kw = cfg.pv_max_kw

    # Definieer de Mock Model klasse één keer
    class SimulatedSolarModel(SolarModel):
        def _load(self):
            self.is_fitted = True
            self.mae = 0.15

        def predict(self, df, model_ratio: float = 0.7):
            val = df["pv_estimate"] * 0.95
            return pd.DataFrame({"prediction": val, "prediction_raw": val})

    print(
        f"\n{'Tijd':<10} | {'PV Nu':<8} | {'Actie':<12} | {'Geplande Start':<15} | {'Reden'}"
    )
    print("-" * 100)

    # 2. Simuleer elk half uur van 08:00 tot 17:00
    for hour in range(8, 18):
        for minute in [0, 30]:
            sim_now = start_of_day.replace(hour=hour, minute=minute)
            ctx.now = sim_now

            # Simuleer realiteit: Er is bewolking (80% van forecast)
            predicted_row = ctx.forecast_df[ctx.forecast_df["timestamp"] == sim_now]
            if predicted_row.empty:
                continue

            predicted_val = predicted_row["pv_estimate"].values[0]
            ctx.stable_pv = predicted_val * 0.8
            ctx.stable_load = cfg.avg_baseload_kw  # Rustig huisverbruik

            # --- CRUCIAL: Initialiseer en PATCH bij elke stap ---
            forecaster = SolarForecaster(cfg, ctx)
            forecaster.model = SimulatedSolarModel(Path("mock.joblib"))
            forecaster.nowcaster = NowCaster(0.15, cfg.pv_max_kw)

            # Voer analyse uit
            decision = forecaster.analyze(ctx.now, ctx.stable_load)

            if decision:
                start_str = (
                    decision.planned_start.strftime("%H:%M")
                    if decision.planned_start
                    else "N/A"
                )
                print(
                    f"{sim_now.strftime('%H:%M'):<10} | {ctx.stable_pv:<8.2f} | {decision.action.value:<12} | {start_str:<15} | {decision.reason}"
                )

                if decision.action.value == "START":
                    print("-" * 100)
                    print(
                        f">>> TRIGGER: Boiler gaat AAN om {sim_now.strftime('%H:%M')}!"
                    )
                    # Optioneel: stop de simulatie als het doel bereikt is
                    # return


if __name__ == "__main__":
    # Overschrijf SolarForecaster __init__ even omdat de originele cfg.solar.model_path gebruikt
    # wat in onze Mock niet zit.
    def manual_init(self, config, context):
        self.config = config
        self.context = context
        self.model = None  # Wordt gepatcht in run_simulation
        self.nowcaster = None
        self.optimizer = SolarOptimizer(
            pv_max_kw=config.pv_max_kw,
            duration_hours=config.dhw_duration_hours,
            min_kwh_threshold=config.min_kwh_threshold,
            avg_baseload_kw=config.avg_baseload_kw,
        )

    SolarForecaster.__init__ = manual_init
    run_simulation()
#     run_time_travel_simulation()
