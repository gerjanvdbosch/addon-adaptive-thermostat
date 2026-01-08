import logging
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

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

    coordinator = request.app.state.coordinator
    context = coordinator.context
    forecast = context.forecast

    details = {}

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
            "Prognose Nu": f"{forecast.energy_now:.2f} kW",
            "Prognose Beste": f"{forecast.energy_best:.2f} kW",
            "Opp. Kosten": f"{forecast.opportunity_cost * 100:.3f} %",
            "Betrouwbaarheid": f"{forecast.confidence * 100:.3f} %",
            "Bias": f"{forecast.current_bias:.2f}",
            "Geplande Start": start_str,
        }
    explanation = _get_model_explanation(coordinator)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            # We geven nu de HTML string door aan de template
            "forecast_plot": plot_html,
            "details": details,
            "explanation": explanation,
        },
    )


@api.get("/solar/explain", response_class=JSONResponse)
def explain_model_prediction(request: Request):
    """
    Geeft de SHAP values terug voor het huidige tijdstip.
    """
    coordinator = request.app.state.coordinator
    context = coordinator.context
    forecaster = coordinator.planner.forecaster

    # 1. Validatie
    if (
        not hasattr(context, "forecast_df")
        or context.forecast_df is None
        or context.forecast_df.empty
    ):
        return JSONResponse({"error": "Geen data beschikbaar"}, status_code=404)

    if not forecaster.model.is_fitted:
        return JSONResponse({"error": "Model is nog niet getraind"}, status_code=503)

    # 2. Zoek de rij die het dichtst bij 'nu' ligt
    local_tz = datetime.now().astimezone().tzinfo
    current_time = datetime.now(local_tz)

    # Zorg dat de dataframe timestamps ook timezone-aware zijn voor vergelijking
    df = context.forecast_df.copy()

    # Kleine hack: pandas searchsorted werkt het best als alles in UTC of offset-naive is
    # We converteren zoek-tijd naar de dataframe tijdzone (vaak UTC)
    target_time = current_time.astimezone(df["timestamp"].dt.tz)

    idx = df["timestamp"].searchsorted(target_time)
    idx = min(idx, len(df) - 1)

    # Pak de rij als DataFrame (niet Series, want prepare_features verwacht DF)
    row = df.iloc[[idx]].copy()

    # 3. Vraag uitleg aan het model
    explanation = forecaster.model.explain(row)

    return JSONResponse(explanation)


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
        return "<div class='alert alert-info'>Geen data beschikbaar. Wacht op de eerste meting (max 1 minuut).</div>"

    local_tz = datetime.now().astimezone().tzinfo
    local_now = context.now.astimezone(local_tz).replace(tzinfo=None)

    # --- 1. DATA VOORBEREIDING (Identiek aan origineel) ---
    df = context.forecast_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    df["power_ml"] = forecaster.model.predict(df)
    df["power_corrected"] = forecaster.nowcaster.apply(
        df, context.now, "power_ml", actual_pv=context.stable_pv
    )

    for col in ["pv_estimate", "power_ml", "power_corrected"]:
        df[col] = df[col].round(3)

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
        return "<div class='alert alert-warning'>Geen relevante data om te tonen (nacht).</div>"

    zon_uren = df[df["power_corrected"] > 0]

    if not zon_uren.empty:
        x_start = zon_uren["timestamp_local"].min() - timedelta(hours=1)
        x_end = zon_uren["timestamp_local"].max() + timedelta(hours=2)
    else:
        # Fallback: als er helemaal geen zon is, toon gewoon alles
        x_start = df["timestamp_local"].min()
        x_end = df["timestamp_local"].max()

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
        df_hist["pv_actual"] = df_hist["pv_actual"].round(3)
        df_hist_plot = df_hist.copy()

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

    #     fig.add_trace(go.Scatter(
    #         x=df_plot["timestamp_local"], y=df_plot["power_pure_ml"],
    #         mode="lines", name="Model",
    #         line=dict(color="#9467bd", dash="dot", width=1.5),
    #         opacity=0.8,
    #         visible="legendonly" # Standaard aan of uit? Zet op True om direct te zien
    #     ))

    if not df_hist_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=df_hist_plot["timestamp_local"],
                y=df_hist_plot["pv_actual"],
                mode="lines",
                name="PV energy",
                legendgroup="history",
                line=dict(color="#ffffff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 255, 255, 0.05)",
                opacity=0.8,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[df_hist_plot["timestamp_local"].iloc[-1], local_now],
                y=[df_hist_plot["pv_actual"].iloc[-1], context.stable_pv],
                mode="lines",
                line=dict(color="#ffffff", dash="dash", width=1.5),  # Wit en gestippeld
                opacity=0.8,
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
            name="PV actual",
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
            name="Prediction",
            line=dict(color="#ffa500", width=2),  # Oranje lijn
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
            # Verticale lijn start
            fig.add_vline(
                x=local_start,
                line_width=2,
                line_dash="dash",
                line_color="#2ca02c",
            )

            # 2. DE TEKST (Handmatig toevoegen)
            fig.add_annotation(
                x=local_start,
                y=1,  # Helemaal bovenin
                yref="paper",  # Y-coordinaat is relatief (0 tot 1)
                text="Start",
                showarrow=False,
                font=dict(color="#2ca02c"),
                xanchor="left",  # Tekst begint links van de lijn
                yanchor="top",
                xshift=5,  # Klein beetje marge van de lijn af
            )

            # Gearceerd gebied (Duration)
            duration_end = local_start + timedelta(hours=forecaster.optimizer.duration)
            fig.add_vrect(
                x0=local_start,
                x1=duration_end,
                fillcolor="#2ca02c",
                opacity=0.15,
                layer="below",
                line_width=0,
            )

    # Algemene Layout
    # y_max = max(df["pv_estimate"].max(), df["pv_actual"].max()) * 1.25

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        title=dict(
            text="Solar Prediction",
        ),
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=40),
        height=500,
        hovermode="x unified",  # Laat alle waardes zien op 1 verticale lijn
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )


def _get_model_explanation(coordinator) -> dict:
    """
    Bereidt de SHAP data voor zodat de HTML het direct kan tonen.
    """
    context = coordinator.context
    forecaster = coordinator.planner.forecaster

    # Geen data of model? Leeg teruggeven
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
        df = context.forecast_df.copy()

        # Voorspelling toevoegen (Raw ML nodig voor Piek-detectie)
        preds = forecaster.model.predict(df)
        if isinstance(preds, pd.DataFrame):
            df["power_ml"] = preds["raw_ml"]
        else:
            df["power_ml"] = preds

        # Zoek index van 'Nu'
        target_time = current_time.astimezone(df["timestamp"].dt.tz)
        idx_now = df["timestamp"].searchsorted(target_time)
        idx_now = min(idx_now, len(df) - 1)

        # --- SLIMME KEUZE LOGICA (Nu vs Piek) ---
        prediction_now = df.iloc[idx_now]["power_ml"]

        row = None
        time_label = ""

        # Als het donker is (< 0.1 kW), zoek de piek van de dag
        if prediction_now < 0.1:
            idx_max = df["power_ml"].idxmax()
            peak_val = df.iloc[idx_max]["power_ml"]

            if peak_val > 0.1:
                row = df.iloc[[idx_max]].copy()
                ts = row["timestamp"].dt.tz_convert(local_tz).iloc[0]
                time_label = f"Piek om {ts.strftime('%H:%M')}"
            else:
                row = df.iloc[[idx_now]].copy()
                time_label = "Nu (Nacht/Donker)"
        else:
            row = df.iloc[[idx_now]].copy()
            time_label = "Nu"

        # Vraag SHAP waardes op
        shap_data = forecaster.model.explain(row)

        # --- DATA VOORBEREIDEN VOOR TEMPLATE ---
        # We willen niet sorteren in Jinja, dat is gedoe. Doen we hier in Python.

        base_val = shap_data.pop("Base", "0.00")
        pred_val = shap_data.pop("Prediction", "0.00")

        factors = []
        for key, val_str in shap_data.items():
            try:
                val = float(val_str)
                # Filter verwaarloosbare waardes
                if abs(val) < 0.01:
                    continue

                # Mooiere labels
                label = key.replace("_sin", "").replace("_cos", "").replace("pv_", "")
                if label == "estimate":
                    label = "Solcast Forecast"
                if label == "radiation":
                    label = "Straling"
                if label == "uncertainty":
                    label = "Onzekerheid"

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
