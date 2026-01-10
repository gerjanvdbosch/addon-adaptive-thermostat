import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Zorg dat we bij de source code kunnen
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from solar import SolarModel, NowCaster  # noqa: E402
from database import Database  # noqa: E402
from config import Config  # noqa: E402

DB_PATH = "database.sqlite"
MODEL_PATH = "solar_model.joblib"


def evaluate_static_model(df_history, system_max_kw):
    print("\n=== 1. STATIC MODEL EVALUATIE ===")

    # 1. Splits data (laatste 20% is test)
    if len(df_history) < 50:
        print("Te weinig data om te splitsen.")
        return

    split_idx = int(len(df_history) * 0.8)
    train_df = df_history.iloc[:split_idx].copy()
    test_df = df_history.iloc[split_idx:].copy()

    print(f"Training set: {len(train_df)} rijen, Test set: {len(test_df)} rijen")

    # 2. Train een tijdelijk model
    # We slaan hem even tijdelijk op om je echte model niet te overschrijven
    temp_path = Path("temp_test_model.joblib")
    model = SolarModel(temp_path)
    model.train(train_df, system_max_kw)

    # 3. Voorspel op test set
    print("Predicting...")

    # LET OP: Jouw model geeft nu een DataFrame terug, we pakken de 'prediction_raw'
    preds_df = model.predict(test_df)
    predictions = preds_df["prediction_raw"]  # De ruwe ML output

    # 4. Vergelijk met werkelijkheid
    # LET OP: Jouw kolom heet 'pv_actual', niet 'actual_pv_yield'
    actuals = test_df["pv_actual"]

    # Metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    bias = np.mean(predictions - actuals)

    print("--- Resultaten Static Model ---")
    print(f"MAE (Gem. afwijking): {mae:.3f} kW")
    print(f"Bias (Positief = overschatting): {bias:.3f} kW")
    print(f"RMSE: {rmse:.3f} kW")

    # 5. Plotten (Opslaan als bestand, werkt ook zonder scherm)
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["timestamp"], actuals, label="Werkelijk", color="black", alpha=0.6)
    plt.plot(
        test_df["timestamp"],
        predictions,
        label="Model (Raw)",
        color="purple",
        linestyle="--",
    )
    plt.title(f"Model Validatie (MAE: {mae:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig("validatie_static.png")
    print("Grafiek opgeslagen als 'validatie_static.png'")

    # Opruimen
    if temp_path.exists():
        temp_path.unlink()


def simulate_realtime_correction(df_test, model_path, system_max_kw):
    print("\n=== 2. REALTIME NOWCASTER SIMULATIE ===")

    # Laad het ECHTE getrainde model
    model = SolarModel(Path(model_path))
    if not model.is_fitted:
        model.train(df_test, system_max_kw)
        print(
            "Je model is nog niet getraind! Het is nu getraind op de testdata voor deze simulatie."
        )

    # Initialiseer NowCaster
    nc = NowCaster(model_mae=model.mae, pv_max_kw=system_max_kw)

    corrected_history = []
    ratios = []

    # Voorspelling voor de hele set (Blend)
    preds_df = model.predict(df_test)
    df_test = df_test.copy()
    df_test["prediction"] = preds_df["prediction"]  # Blend gebruiken voor NowCaster

    print("Simulating NowCaster loop...")

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        timestamp = row["timestamp"]
        actual = row["pv_actual"]
        predicted_base = row["prediction"]

        # 1. UPDATE: Leren van het verleden (net zoals in het echt)
        nc.update(actual_kw=actual, forecasted_kw=predicted_base)
        ratios.append(nc.current_ratio)

        # 2. APPLY: Correctie toepassen op het huidige punt
        # We doen alsof dit punt 'de toekomst' is van 0 uur ver
        # (Apply verwacht een dataframe, dus we maken een mini-dataframe van 1 rij)
        mini_df = df_test.iloc[[i]].copy()
        corrected_series = nc.apply(mini_df, timestamp, "prediction", actual_pv=actual)

        corrected_history.append(corrected_series.iloc[0])

    # Evaluatie
    mae_base = mean_absolute_error(df_test["pv_actual"], df_test["prediction"])
    mae_corrected = mean_absolute_error(df_test["pv_actual"], corrected_history)

    print("--- NowCaster Effect ---")
    print(f"MAE Basis (Blend):   {mae_base:.3f} kW")
    print(f"MAE Met NowCaster:   {mae_corrected:.3f} kW")
    if mae_base > 0:
        print(
            f"Verbetering:         {((mae_base - mae_corrected) / mae_base) * 100:.1f}%"
        )

    # Plot
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(
        df_test["timestamp"],
        df_test["pv_actual"],
        label="Werkelijk",
        color="black",
        lw=1.5,
    )
    plt.plot(
        df_test["timestamp"],
        df_test["prediction"],
        label="Basis Blend",
        color="orange",
        ls="--",
    )
    plt.plot(
        df_test["timestamp"],
        corrected_history,
        label="Met NowCaster",
        color="green",
        lw=1.5,
    )
    plt.title("Realtime Simulatie")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(df_test["timestamp"], ratios, label="Correctie Ratio", color="blue")
    plt.axhline(1.0, color="gray", ls="--")
    plt.title("Correctie Factor")
    plt.grid()

    plt.tight_layout()
    plt.savefig("validatie_nowcaster.png")
    print("Grafiek opgeslagen als 'validatie_nowcaster.png'")


if __name__ == "__main__":
    # 1. Haal data uit je ECHTE database
    # Dit is belangrijk: we willen weten hoe het model het doet op jouw data
    try:
        # Hacky manier om Database te initen zonder de hele app
        # Pas Config() aan als die parameters nodig heeft
        config = Config()
        config.database_path = DB_PATH
        db = Database(config)

        # Haal alles op wat je hebt
        print("Data ophalen uit database...")
        # We pakken een datum ver in het verleden om alles te krijgen
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
        df = db.get_forecast_history(cutoff)

        if df.empty:
            print("FOUT: Database is leeg!")
        else:
            print(f"Geladen: {len(df)} rijen.")

            # Stap 1: Test of het model uberhaupt iets kan leren
            evaluate_static_model(df, system_max_kw=config.pv_max_kw)

            # Stap 2: Test hoe goed de live correctie werkt
            # We gebruiken dezelfde dataset even als test
            simulate_realtime_correction(
                df, str(MODEL_PATH), system_max_kw=config.pv_max_kw
            )

    except Exception as e:
        print(f"Er ging iets mis: {e}")
        import traceback

        traceback.print_exc()
