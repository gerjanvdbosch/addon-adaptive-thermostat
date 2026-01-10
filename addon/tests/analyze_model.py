import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from pathlib import Path
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from solar import SolarModel  # noqa: E402
from config import Config  # noqa: E402
from database import Database  # noqa: E402

# Instellingen
DB_PATH = "database.sqlite"
MODEL_PATH = "solar_model.joblib"
CUTOFF_DATE = "2025-01-08"  # Datum vanaf waar je data wilt analyseren


def analyze():
    print("1. Model en Data laden...")
    model_wrapper = SolarModel(Path(MODEL_PATH))

    config = Config()
    config.database_path = DB_PATH
    db = Database(config)
    df = db.get_forecast_history(pd.to_datetime(CUTOFF_DATE).tz_localize("UTC"))

    model_wrapper.train(df, config.pv_max_kw)
    if not model_wrapper.is_fitted:
        print("Model is nog niet getraind!")
        return

    # Filteren en voorbereiden (net zoals in je train functie)
    is_daytime = (df["pv_estimate"] > 0.0) | (df["pv_actual"] > 0.0)
    df_day = df[is_daytime].copy()

    if df_day.empty:
        print("Geen dag-data gevonden! Check je database.")
        return

    X = model_wrapper._prepare_features(df_day)
    y = df_day["pv_actual"]

    print("\n--- DIAGNOSE ---")
    test_preds = model_wrapper.model.predict(X)
    print(f"Gemiddelde voorspelling: {test_preds.mean():.4f} kW")
    print(f"Maximale voorspelling:   {test_preds.max():.4f} kW")

    if test_preds.max() < 0.01:
        print("!!! ALARM: Je model voorspelt ALLEEN MAAR 0. !!!")
        print(
            "Oorzaak: Je traint lokaal op een lege database of laadt een corrupt bestand."
        )
        return  # Stop hier, want plotten heeft geen zin

    print(f"Data geladen: {len(X)} rijen (gefilterd op daglicht).")

    # --- ANALYSE 1: FEATURE IMPORTANCE ---
    print("3. Belangrijkste factoren berekenen...")
    result = permutation_importance(
        model_wrapper.model, X, y, n_repeats=5, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()
    top_features = X.columns[sorted_idx]

    print("\n--- Top 5 Belangrijkste Features ---")
    for name in top_features[-5:]:
        print(f"- {name}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        tick_labels=X.columns[sorted_idx],  # <--- AANPASSING: labels -> tick_labels
    )
    plt.title("Welke factoren bepalen de opbrengst het meest?")
    plt.tight_layout()
    plt.savefig("analyse_importance.png")
    print("Opgeslagen: analyse_importance.png")

    # --- ANALYSE 2: PARTIAL DEPENDENCE ---
    print("4. Gedrag analyseren (Partial Dependence)...")

    # Pak de 3 belangrijkste features
    features_to_plot = top_features[-3:]
    print(f"Plotten van gedrag voor: {list(features_to_plot)}")

    fig, ax = plt.subplots(figsize=(12, 4))

    # Hier zat de warning: Als het model overal 0 voorspelt, crasht de schaal
    try:
        PartialDependenceDisplay.from_estimator(
            model_wrapper.model, X, features_to_plot, kind="average", ax=ax
        )
        plt.suptitle("Hoe reageert het model op veranderingen?", y=1.02)
        plt.tight_layout()
        plt.savefig("analyse_behavior.png")
        print("Opgeslagen: analyse_behavior.png")
    except Exception as e:
        print(
            f"Kon behavior plot niet maken (waarschijnlijk is model een platte lijn): {e}"
        )


if __name__ == "__main__":
    analyze()
