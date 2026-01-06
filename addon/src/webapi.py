import logging
import io
import matplotlib.dates as mdates
import base64

from matplotlib.figure import Figure
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from datetime import timedelta, datetime
from pathlib import Path


logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@api.get("/", response_class=HTMLResponse)
def index(request: Request):
    image = _get_solar_forecast_image(request)
    image_b64 = base64.b64encode(image).decode("utf-8")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "forecast_image": image_b64},
    )


@api.get("/solar/forecast", response_class=Response)
def get_solar_plot(request: Request):
    image = _get_solar_forecast_image(request)
    return Response(content=image, media_type="image/png")


def _get_solar_forecast_image(request: Request):
    coordinator = request.app.state.coordinator
    context = coordinator.context
    forecaster = coordinator.planner.forecaster
    forecast = context.forecast

    if not hasattr(context, "forecast") or context.forecast is None:
        return Response(
            content="Geen data beschikbaar. Wacht op de eerste meting (max 1 minuut).",
            status_code=202,
        )

    local_tz = datetime.now().astimezone().tzinfo
    local_now = context.now.astimezone(local_tz).replace(tzinfo=None)

    df = context.forecast_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    df["power_ml"] = forecaster.model.predict(df)
    df["power_corrected"] = forecaster.nowcaster.apply(
        df, context.now, "power_ml", actual_pv=context.stable_pv
    )

    # Load & Net Power projectie
    baseload = forecaster.optimizer.avg_baseload
    df["consumption"] = baseload

    future_mask = df["timestamp_local"] >= local_now
    future_indices = df.index[future_mask]
    decay_steps = 2
    for i, idx in enumerate(future_indices[: decay_steps + 1]):
        factor = 1.0 - (i / decay_steps)
        blended_load = (context.stable_load * factor) + (baseload * (1 - factor))
        df.at[idx, "consumption"] = max(blended_load, baseload)

    df.loc[df["timestamp_local"] < local_now, "consumption"] = context.stable_load
    df["net_power"] = (df["power_corrected"] - df["consumption"]).clip(lower=0)

    # 3. FILTERING (Nu kan 'power_corrected' wel gebruikt worden)
    one_hour_ago = local_now - timedelta(hours=1)

    # Filter: 1u terug en alleen waar zon is of nabij 'nu'
    mask = (df["timestamp_local"] >= one_hour_ago) & (
        (df["pv_estimate"] > 0.0)
        | (df["power_corrected"] > 0.0)
        | (df["timestamp_local"] <= local_now + timedelta(hours=1))
    )
    df_plot = df[mask].copy()

    if df_plot.empty:
        return Response(
            content="Geen relevante data om te tonen (nacht).", status_code=202
        )

    # --- PLOT GENERATIE ---
    fig = Figure(figsize=(12, 7), dpi=200)
    ax = fig.add_subplot(111)

    # Gebruik df_plot["timestamp"] voor alle plot calls
    ax.plot(
        df_plot["timestamp_local"],
        df_plot["pv_estimate"],
        "--",
        label="Raw Solcast (Forecast)",
        color="gray",
        alpha=0.4,
    )
    ax.plot(
        df_plot["timestamp_local"],
        df_plot["power_ml"],
        ":",
        label="Model Correction",
        color="blue",
        alpha=0.6,
    )
    ax.axhline(
        y=context.stable_pv,
        color="darkgreen",
        linestyle=":",
        alpha=0.5,
        xmin=0.05,
        xmax=0.95,
    )
    # En een duidelijke marker op de NU-lijn
    x_min_plot = df_plot["timestamp_local"].min()

    # Teken de horizontale lijn:
    # y = de waarde, xmin = begin van de grafiek, xmax = nu
    ax.hlines(
        y=context.stable_pv,
        xmin=x_min_plot,
        xmax=local_now,
        color="darkgreen",
        linestyle=":",
        alpha=0.4,
        linewidth=1,
    )

    # De Stip (Huidige Meting) exact op het eindpunt van de stippellijn
    ax.scatter(
        local_now,
        context.stable_pv,
        color="darkgreen",
        s=120,
        edgecolors="white",
        linewidths=1.5,
        zorder=15,
        label=f"Actuele PV Meting ({context.stable_pv:.2f} kW)",
    )
    ax.plot(
        df_plot["timestamp_local"],
        df_plot["power_corrected"],
        "g-",
        linewidth=2,
        label="Corrected Solar (Nowcast)",
    )
    ax.step(
        df_plot["timestamp_local"],
        df_plot["consumption"],
        where="post",
        color="red",
        linewidth=2,
        label="Load Projection",
    )
    ax.fill_between(
        df_plot["timestamp_local"],
        0,
        df_plot["net_power"],
        color="green",
        alpha=0.15,
        label="Netto Solar",
    )

    # "NU" lijn
    ax.axvline(local_now, color="black", linestyle="-", alpha=0.6, linewidth=1)
    y_max = max(df_plot["pv_estimate"].max(), df_plot["consumption"].max()) * 1.35
    ax.text(
        local_now, y_max * 0.95, " NU", color="black", fontweight="bold", fontsize=9
    )

    # Start Window
    if forecast and forecast.planned_start:
        local_start = forecast.planned_start.astimezone(local_tz).replace(tzinfo=None)
        # Check of startmoment in het plot-window valt
        if local_start >= df_plot["timestamp_local"].min():
            ax.axvline(
                local_start,
                color="orange",
                linestyle="--",
                linewidth=2.5,
                label=f"Start ({local_start.strftime('%H:%M')})",
                zorder=9,
            )

            duration_end = local_start + timedelta(hours=forecaster.optimizer.duration)
            ax.axvspan(local_start, duration_end, color="orange", alpha=0.12, zorder=1)

    # X-as formattering
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(
        f"Solar Optimizer: {forecast.action.value}\n{forecast.reason}",
        fontsize=11,
        pad=15,
    )
    ax.set_ylabel("Vermogen (kW)")
    ax.set_xlabel("Tijd")

    # Legenda (Rechtsboven)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=9,
        framealpha=0.8,
        edgecolor="silver",
        borderaxespad=0.2,
    )

    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_ylim(0, y_max)
    fig.autofmt_xdate()

    # EXPORT
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=200, bbox_inches="tight")
    output.seek(0)

    return output.getvalue()
