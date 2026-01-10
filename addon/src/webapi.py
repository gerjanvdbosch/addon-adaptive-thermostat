import logging
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

from plotly.subplots import make_subplots
from sklearn.inspection import partial_dependence, permutation_importance
from operator import itemgetter
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import timedelta, datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@api.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Haal de HTML div string op in plaats van een plaatje
    plot_html = _get_solar_forecast_plot(request)
    importance_html = _get_importance_plot_plotly(request)
    behavior_html = _get_behavior_plot_plotly(request)

    coordinator = request.app.state.coordinator
    context = coordinator.context
    forecast = context.forecast

    details = {}
    explanation = {}

    if hasattr(context, "forecast") and context.forecast is not None:
        # Helper voor veilige datum weergave
        start_str = "-"
        if forecast.planned_start:
            # Converteer naar lokale tijd voor weergave
            local_tz = datetime.now().astimezone().tzinfo
            local_start = forecast.planned_start.astimezone(local_tz)
            start_str = local_start.strftime("%H:%M")

        details = {
            "Status": forecast.action.value,
            "Reden": forecast.reason,
            "PV Huidig": f"{forecast.actual_pv:.2f} kW",
            "Load Huidig": f"{forecast.load_now:.2f} kW",
            "Prognose Nu": f"{forecast.energy_now:.2f} kWh",
            "Prognose Beste": f"{forecast.energy_best:.2f} kWh",
            "Opp. Kosten": f"{forecast.opportunity_cost * 100:.3f} %",
            "Betrouwbaarheid": f"{forecast.confidence * 100:.3f} %",
            "Bias": f"{forecast.current_bias:.2f}",
            "Geplande Start": start_str,
        }

        # 3. Explain Genereren (Gedeelde Functie)
        explanation = _get_explanation_data(coordinator)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "forecast_plot": plot_html,
            "importance_plot": importance_html,
            "behavior_plot": behavior_html,
            "details": details,
            "explanation": explanation,
        },
    )


@api.post("/solar/train", response_class=JSONResponse)
def trigger_training(request: Request):
    """
    Forceer een hertraining van het model op basis van de laatste 60 dagen.
    """
    coordinator = request.app.state.coordinator
    forecaster = coordinator.planner.forecaster
    database = coordinator.collector.database
    config = coordinator.config

    # 1. Data ophalen
    # Zorg dat we genoeg data hebben voor een goed model
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=730)

    # We gebruiken de bestaande functie die ook voor de grafiek wordt gebruikt,
    # maar dan met een datum ver in het verleden.
    df_hist = database.get_forecast_history(cutoff_date)

    if df_hist.empty or len(df_hist) < 48:  # Minimaal ~12 uur aan kwartier-data
        return JSONResponse(
            {"error": "Te weinig data beschikbaar in database (minimaal 1 dag nodig)."},
            status_code=400,
        )

    # 2. Train het model
    try:
        # Dit blokkeert heel even de server (paar seconden), dat is prima voor thuisgebruik.
        forecaster.model.train(df_hist, config.pv_max_kw)
        new_mae = forecaster.model.mae

        return {
            "status": "success",
            "message": f"Model getraind op {len(df_hist)} datapunten.",
            "new_mae": round(new_mae, 3),
        }

    except Exception as e:
        logger.error(f"[API] Training mislukt: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def _get_solar_forecast_plot(request: Request) -> str:
    """
    Genereert een interactieve Plotly grafiek en retourneert deze als HTML string (div).
    """
    coordinator = request.app.state.coordinator
    context = coordinator.context
    forecaster = coordinator.planner.forecaster
    forecast = context.forecast
    database = coordinator.collector.database

    # Check of er data is
    if not hasattr(context, "forecast") or context.forecast is None:
        return ""

    local_tz = datetime.now().astimezone().tzinfo
    local_now = context.now.astimezone(local_tz).replace(tzinfo=None)

    # --- 1. DATA VOORBEREIDING (Identiek aan origineel) ---
    df = context.forecast_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    for col in ["pv_estimate", "power_ml", "power_ml_raw", "power_corrected"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Load & Net Power projectie
    baseload = forecaster.optimizer.avg_baseload
    df["consumption"] = baseload

    future_mask = df["timestamp_local"] >= local_now
    future_indices = df.index[future_mask].copy()
    decay_steps = 2
    for i, idx in enumerate(future_indices[: decay_steps + 1]):
        factor = 1.0 - (i / decay_steps)
        blended_load = (context.stable_load * factor) + (baseload * (1 - factor))
        df.at[idx, "consumption"] = max(blended_load, baseload)

    df.loc[df["timestamp_local"] < local_now, "consumption"] = context.stable_load
    df["net_power"] = (df["power_corrected"] - df["consumption"]).clip(lower=0)

    if df.empty:
        return ""

    # --- 2. PLOT GENERATIE (PLOTLY) ---
    cutoff_date = (
        local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        .replace(tzinfo=local_tz)
        .astimezone(timezone.utc)
    )

    df_hist = database.get_forecast_history(cutoff_date)
    df_hist_plot = pd.DataFrame()

    if not df_hist.empty:
        df_hist["timestamp_local"] = (
            df_hist["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
        )
        df_hist["pv_actual"] = df_hist["pv_actual"].round(2)
        df_hist_plot = df_hist.copy()

    zon_uren = df[df["power_corrected"] > 0]
    if not zon_uren.empty:
        x_start = zon_uren["timestamp_local"].min() - timedelta(hours=2)
        x_end = max(
            zon_uren["timestamp_local"].max(), df_hist_plot["timestamp_local"].max()
        ) + timedelta(hours=2)
    else:
        x_start = df["timestamp_local"].min()
        x_end = df["timestamp_local"].max()

    fig = go.Figure()

    # A. Raw Solcast (Grijs, dashed)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_local"],
            y=df["pv_estimate"],
            mode="lines",
            name="Solcast",
            line=dict(color="#888888", dash="dash", width=1),
            opacity=0.7,
        )
    )

    # B. Model Correction (Blauw, dot)
    if "power_ml" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_local"],
                y=df["power_ml"],
                mode="lines",
                name="Blended",
                line=dict(color="#4fa8ff", dash="dot", width=1),
                opacity=0.8,
            )
        )

    if "power_ml_raw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_local"],
                y=df["power_ml_raw"],
                mode="lines",
                name="Model",
                line=dict(color="#9467bd", dash="dot", width=1.5),  # Paars stippel
                opacity=0.6,
            )
        )

    if not df_hist_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=df_hist_plot["timestamp_local"],
                y=df_hist_plot["pv_actual"],
                mode="lines",
                name="PV history",
                legendgroup="history",
                line=dict(color="#ffa500", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.1)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[df_hist_plot["timestamp_local"].iloc[-1], local_now],
                y=[df_hist_plot["pv_actual"].iloc[-1], context.stable_pv],
                mode="lines",
                line=dict(color="#ffa500", dash="dash", width=1.5),  # Wit en gestippeld
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.1)",
                showlegend=False,  # We hoeven deze niet apart in de legenda
                hoverinfo="skip",  # Geen popup als je over het lijntje muist
                legendgroup="history",
            )
        )

    # C. Actuele PV Meting (Stip)
    # We tekenen alleen de stip, de horizontale stippellijn doen we via shapes of een losse trace als je wilt
    fig.add_trace(
        go.Scatter(
            x=[local_now],
            y=[context.stable_pv],
            mode="markers",
            name="Now",
            showlegend=False,
            marker=dict(color="#ffa500", size=12, line=dict(color="white", width=2)),
            zorder=10,
        )
    )

    # D. Corrected Solar
    df_future = df[df["timestamp_local"] >= local_now]

    # Om de lijn visueel aan het 'Huidig PV' bolletje te knopen,
    # plakken we het huidige punt vooraan de lijst.
    x_future = [local_now] + df_future["timestamp_local"].tolist()
    y_future = [context.stable_pv] + df_future["power_corrected"].tolist()

    fig.add_trace(
        go.Scatter(
            x=x_future,
            y=y_future,
            mode="lines",
            name="Forecast",
            line=dict(color="#ffffff", dash="dash", width=2),
            fill="tozeroy",  # Vul tot aan de X-as (0)
            fillcolor="rgba(255, 255, 255, 0.05)",
            opacity=0.8,
        )
    )

    # E. Load Projection (Rood, Step)
    #     fig.add_trace(
    #         go.Scatter(
    #             x=df["timestamp_local"],
    #             y=df["consumption"],
    #             mode="lines",
    #             name="Load Projection",
    #             line=dict(color="#ff5555", width=2, shape="hv"),  # shape='hv' is step-post
    #         )
    #     )

    # F. Netto Solar (Filled Area)
    # In Plotly is fill='tozeroy' makkelijk, maar om specifiek netto te kleuren gebruiken we de berekende kolom
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_local"],
            y=df["net_power"],
            mode="lines",  # Geen markers
            name="Netto Solar",
            showlegend=False,
            line=dict(width=0),  # Geen rand
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.3)",
            hoverinfo="skip",  # Maakt de grafiek rustiger bij hoveren
        )
    )

    # --- LAYOUT & SHAPES ---

    # Verticale lijn voor NU
    fig.add_vline(
        x=local_now, line_width=1, line_dash="solid", line_color="white", opacity=0.6
    )

    # Start Window Logic
    if forecast and forecast.planned_start:
        local_start = forecast.planned_start.astimezone(local_tz).replace(tzinfo=None)

        if local_start >= df["timestamp_local"].min():
            y_top = max(
                df[["power_corrected", "pv_estimate"]].max().max(),
                df_hist_plot["pv_actual"].max() if not df_hist_plot.empty else 0,
            )

            duration_end = local_start + timedelta(hours=forecaster.optimizer.duration)

            # 2. TRACE A: De verticale stippellijn + Tekst
            fig.add_trace(
                go.Scatter(
                    x=[local_start, local_start],
                    y=[0, y_top],
                    mode="lines+text",
                    name="Start",  # Dit komt in de legenda
                    legendgroup="start",  # Koppelnaam
                    line=dict(color="#2ca02c", width=2),
                    text=["", "Start"],  # Tekst alleen bij het bovenste punt
                    textposition="top right",  # Rechts van de lijn
                    textfont=dict(color="#2ca02c"),
                )
            )

            # 3. TRACE B: Het gearceerde vlak
            # We tekenen een onzichtbare lijn rondom het gebied en vullen die in ("toself")
            fig.add_trace(
                go.Scatter(
                    x=[local_start, duration_end, duration_end, local_start],
                    y=[0, 0, y_top, y_top],
                    fill="toself",
                    fillcolor="rgba(44, 160, 44, 0.15)",
                    mode="lines",
                    line=dict(width=0),  # Geen rand
                    name="Start",  # Zelfde naam
                    legendgroup="start",  # Zelfde groep!
                    showlegend=False,  # Niet dubbel in de lijst tonen
                    hoverinfo="skip",
                )
            )

    # Algemene Layout
    # y_max = max(df["pv_estimate"].max(), df["pv_actual"].max()) * 1.25

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        xaxis=dict(
            title="Tijd",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[x_start, x_end],
        ),
        yaxis=dict(
            title="Vermogen (kW)",
            # range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemdoubleclick=False,
        ),
        margin=dict(l=40, r=20, t=80, b=40),
        height=500,
        hovermode="x unified",  # Laat alle waardes zien op 1 verticale lijn
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )


def _get_explanation_data(coordinator) -> dict:
    """
    CENTRALE FUNCTIE: Bereidt de SHAP data voor.
    Wordt gebruikt door zowel de HTML template als de JSON API.
    """
    context = coordinator.context
    forecaster = coordinator.planner.forecaster

    # 1. Validatie
    if (
        not hasattr(context, "forecast_df")
        or context.forecast_df is None
        or context.forecast_df.empty
    ):
        return None

    if not forecaster.model.is_fitted:
        return None

    try:
        local_tz = datetime.now().astimezone().tzinfo
        current_time = datetime.now(local_tz)

        # Werk op een kopie
        df = context.forecast_df.copy()

        # 2. Bepaal kolom: 'power_ml_raw' heeft voorkeur (want dat legt SHAP uit)
        target_col = "power_ml_raw"

        # 3. Zoek 'Nu'
        target_time = current_time.astimezone(df["timestamp"].dt.tz)
        idx_now = df["timestamp"].searchsorted(target_time)
        idx_now = min(idx_now, len(df) - 1)

        # 4. Slimme Logica: Nu of Piek?
        prediction_now = df.iloc[idx_now][target_col]

        row = None
        time_label = ""

        # Als het donker is (< 0.05 kW), zoek de piek van de dag
        if prediction_now < 0.05:
            idx_max = df[target_col].idxmax()
            peak_val = df.iloc[idx_max][target_col]

            if peak_val > 0.1:
                row = df.iloc[[idx_max]].copy()
                ts = row["timestamp"].dt.tz_convert(local_tz).iloc[0]
                time_label = f"Piek om {ts.strftime('%H:%M')}"
            else:
                row = df.iloc[[idx_now]].copy()
                time_label = "Nu (Nacht)"
        else:
            row = df.iloc[[idx_now]].copy()
            time_label = "Nu"

        # 5. Vraag SHAP waardes op
        shap_data = forecaster.model.explain(row)

        # 6. Formatteren en Opschonen
        # Haal base en prediction eruit (kleine letters want SolarModel geeft lowercase terug)
        base_val = shap_data.pop("base", "0.00")
        pred_val = shap_data.pop("prediction", "0.00")

        # Verwijder rommel
        shap_data.pop("Info", None)
        shap_data.pop("_meta_label", None)

        factors = []
        for key, val_str in shap_data.items():
            try:
                val = float(val_str)
                # Filter verwaarloosbare waardes (< 10 Watt)
                if abs(val) < 0.01:
                    continue

                # Labels vertalen naar leesbaar Nederlands
                label = key.replace("_sin", "").replace("_cos", "").replace("pv_", "")
                if label == "estimate":
                    label = "Solcast Forecast"
                if label == "radiation":
                    label = "Straling"
                if label == "uncertainty":
                    label = "Onzekerheid"
                if label == "diffuse":
                    label = "Diffuus licht"
                if label == "tilted":
                    label = "Dakhelling"
                if label == "temp":
                    label = "Temperatuur"
                if label == "cloud":
                    label = "Bewolking"

                factors.append(
                    {
                        "label": label,
                        "value": val_str,
                        "abs_value": abs(val),
                        "css_class": "val-pos" if val >= 0 else "val-neg",
                    }
                )
            except Exception:
                continue

        # Sorteer op absolute impact (grootste bovenaan)
        factors.sort(key=itemgetter("abs_value"), reverse=True)

        return {
            "time_label": time_label,
            "base": base_val,
            "prediction": pred_val,
            "factors": factors,
        }

    except Exception as e:
        logger.error(f"Error preparing explanation: {e}")
        return None


def _get_importance_plot_plotly(request: Request) -> str:
    """
    Genereert een Plotly Bar Chart met de feature importance.
    Let op: Dit is rekenintensief, dus we doen dit op een beperkte set.
    """
    coordinator = request.app.state.coordinator
    forecaster = coordinator.planner.forecaster
    database = coordinator.collector.database

    # 1. Check: Is het model getraind?
    if not forecaster.model.is_fitted:
        return "<div class='p-4 text-muted'>Model nog niet getraind.</div>"

    # 2. Data ophalen (Beperk tot laatste 14 dagen voor snelheid)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)
    df_hist = database.get_forecast_history(cutoff_date)

    if df_hist.empty:
        return "<div class='p-4 text-muted'>Onvoldoende data voor analyse.</div>"

    # 3. Filter op daglicht (cruciaal, anders klopt de importance niet)
    # We gebruiken dezelfde logica als bij het trainen/analyseren
    is_daytime = (df_hist["pv_estimate"] > 0.01) | (df_hist["pv_actual"] > 0.01)
    df_day = df_hist[is_daytime].copy()

    if len(df_day) < 10:
        return "<div class='p-4 text-muted'>Wachten op meer daglicht-data...</div>"

    # 4. Features voorbereiden
    try:
        X = forecaster.model._prepare_features(df_day)
        y = df_day["pv_actual"]

        # 5. Bereken Importance (n_repeats=2 houdt het snel genoeg voor een dashboard)
        # Voor wetenschappelijke precisie wil je 10, voor een dashboard is 2-3 prima.
        result = permutation_importance(
            forecaster.model.model, X, y, n_repeats=2, random_state=42, n_jobs=-1
        )
    except Exception as e:
        logger.error(f"Fout bij berekenen importance: {e}")
        return "<div class='p-4 text-danger'>Kon importance niet berekenen.</div>"

    # 6. Sorteren en Labelen
    sorted_idx = result.importances_mean.argsort()

    # Vertaal de labels naar leesbaar Nederlands
    labels_raw = X.columns[sorted_idx]
    labels_clean = []

    label_map = {
        "pv_estimate": "Solcast Basis",
        "pv_estimate10": "Solcast Min (10%)",
        "pv_estimate90": "Solcast Max (90%)",
        "radiation": "Straling",
        "hour_cos": "Tijdstip (Uur)",
        "hour_sin": "Tijdstip (Cyclisch)",
        "temp": "Temperatuur",
        "cloud": "Bewolking",
        "diffuse": "Diffuus Licht",
        "tilted": "Dakhelling Effect",
        "uncertainty": "Onzekerheid",
        "wind": "Wind",
        "precipitation": "Neerslag",
        "doy_cos": "Seizoen",
        "doy_sin": "Seizoen (Cyclisch)",
    }

    for label in labels_raw:
        clean = label_map.get(label, label)  # Pak vertaling of origineel
        labels_clean.append(clean)

    values = result.importances_mean[sorted_idx]

    # 7. Plotly Grafiek Maken
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels_clean,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="Viridis",  # Of 'Blues', 'Magma'
                showscale=False,
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        margin=dict(l=20, r=20, t=30, b=30),
        height=400,
        xaxis=dict(
            title="Impact op voorspelling",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(showgrid=False),
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
    )


def _get_behavior_plot_plotly(request: Request) -> str:
    """
    Genereert Partial Dependence Plots (PDP) voor de belangrijkste fysieke features.
    """
    coordinator = request.app.state.coordinator
    forecaster = coordinator.planner.forecaster
    database = coordinator.collector.database

    if not forecaster.model.is_fitted:
        return ""

    # 1. Data ophalen (Laatste 14 dagen is genoeg voor gedrag)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)
    df_hist = database.get_forecast_history(cutoff_date)

    # Filter op daglicht
    is_daytime = (df_hist["pv_estimate"] > 0.01) | (df_hist["pv_actual"] > 0.01)
    df_day = df_hist[is_daytime].copy()

    if len(df_day) < 50:
        return "<div class='p-4 text-muted'>Te weinig data voor gedragsanalyse.</div>"

    try:
        X = forecaster.model._prepare_features(df_day)

        # 2. Kies de features die we willen inspecteren
        # We pakken hardcoded de 3 interessantste, dat is sneller dan eerst sorteren.
        features_to_plot = ["radiation", "pv_estimate10", "temp"]
        feature_labels = ["Straling (W/m²)", "Solcast Min (kW)", "Temperatuur (°C)"]

        # 3. Maak Subplots (1 rij, 3 kolommen)
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=feature_labels, horizontal_spacing=0.05
        )

        # 4. Berekenen en tekenen per feature
        for i, feature in enumerate(features_to_plot):
            if feature not in X.columns:
                continue

            # Dit is het zware werk: Sklearn rekent de lijn uit
            pdp_results = partial_dependence(
                forecaster.model.model, X, [feature], grid_resolution=20, kind="average"
            )

            x_vals = pdp_results["grid_values"][0]
            y_vals = pdp_results["average"][0]

            # Voeg lijn toe aan de subplot
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=feature,
                    line=dict(width=3, color="#4fa8ff"),  # Mooi blauw
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )

        # 5. Opmaak
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgb(28, 28, 28)",
            plot_bgcolor="rgb(28, 28, 28)",
            margin=dict(l=20, r=20, t=50, b=20),
            height=300,  # Niet te hoog maken
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )

        return pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False},
        )

    except Exception as e:
        logger.error(f"Fout bij behavior plot: {e}")
        return "<div class='p-4 text-danger'>Kon gedrag niet analyseren.</div>"
