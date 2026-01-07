import logging
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from datetime import timedelta, datetime
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

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            # We geven nu de HTML string door aan de template
            "forecast_plot": plot_html,
            "details": details,
        },
    )


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

    # --- 2. PLOT GENERATIE (PLOTLY) ---
    cutoff_date = local_now.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=10)

    df_hist = database.get_forecast_history(cutoff_date)
    df_hist_plot = pd.DataFrame()

    logger.info(f"History DataFrame has {len(df_hist)} rows.")

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
            name="Model",
            line=dict(color="#4fa8ff", dash="dot", width=1),
            opacity=0.8,
        )
    )

    if not df_hist_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=df_hist_plot["timestamp_local"],
                y=df_hist_plot["pv_actual"],
                mode="lines",
                name="PV energy",
                line=dict(color="#ffffff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 255, 255, 0.05)",
                opacity=0.8,
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

    # D. Corrected Solar (Solid Green)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_local"],
            y=df["power_corrected"],
            mode="lines",
            name="Solar Optimizer",
            line=dict(color="#ffa500", width=2),  # Matplotlib 'g-' equivalent
        )
    )

    # E. Load Projection (Rood, Step)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_local"],
            y=df["consumption"],
            mode="lines",
            name="Load Projection",
            line=dict(color="#ff5555", width=2, shape="hv"),  # shape='hv' is step-post
        )
    )

    # F. Netto Solar (Filled Area)
    # In Plotly is fill='tozeroy' makkelijk, maar om specifiek netto te kleuren gebruiken we de berekende kolom
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_local"],
            y=df["net_power"],
            mode="lines",  # Geen markers
            name="Netto Solar",
            line=dict(width=0),  # Geen rand
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.2)",  # Green met alpha
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
                annotation_text="Start",
                annotation_position="top left",
                annotation_font_color="#2ca02c",
            )

            # Gearceerd gebied (Duration)
            duration_end = local_start + timedelta(hours=forecaster.optimizer.duration)
            fig.add_vrect(
                x0=local_start,
                x1=duration_end,
                fillcolor="#2ca02c",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

    # Algemene Layout
    # y_max = max(df["pv_estimate"].max(), df["consumption"].max()) * 1.25
    y_max = max(df["pv_estimate"].max()) * 1.25

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        title=dict(
            text="Solar Prognose",
        ),
        xaxis=dict(title="Tijd", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(
            title="Vermogen (kW)",
            range=[0, y_max],
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
