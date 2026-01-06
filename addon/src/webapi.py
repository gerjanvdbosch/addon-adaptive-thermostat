import logging
import io
import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from fastapi import FastAPI, Response, Request
from datetime import timedelta, datetime


logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")


@api.get("/")
def index():
    return {"status": "ok"}


@api.get("/solar/plot")
def get_solar_plot(request: Request):
    try:
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
        df = context.forecast_df.copy()
        df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz)
        local_now = context.now.astimezone(local_tz)

        df["power_ml"] = forecaster.model.predict(df)
        df["power_corrected"] = forecaster.nowcaster.apply(
            df, context.now, "power_ml", actual_pv=context.stable_pv
        )

        baseload = forecaster.optimizer.avg_baseload
        df["consumption"] = baseload

        future_mask = df["timestamp"] >= context.now
        future_indices = df.index[future_mask]
        decay_steps = 3
        for i, idx in enumerate(future_indices[: decay_steps + 1]):
            factor = 1.0 - (i / decay_steps)
            blended_load = (context.stable_load * factor) + (baseload * (1 - factor))
            df.at[idx, "consumption"] = max(blended_load, baseload)

        df.loc[df["timestamp"] < context.now, "consumption"] = context.stable_load
        df["net_power"] = (df["power_corrected"] - df["consumption"]).clip(lower=0)

        # --- PLOT GENERATIE ---
        fig = Figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

        # Gebruik df["timestamp_local"] voor alle plot calls
        ax.plot(
            df["timestamp_local"],
            df["pv_estimate"],
            "--",
            label="Raw Solcast (Forecast)",
            color="gray",
            alpha=0.4,
        )
        ax.plot(
            df["timestamp_local"],
            df["power_ml"],
            ":",
            label="ML Model Output",
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            df["timestamp_local"],
            df["power_corrected"],
            "g-",
            linewidth=2,
            label="Corrected Solar (Nowcast)",
        )
        ax.step(
            df["timestamp_local"],
            df["consumption"],
            where="post",
            color="red",
            linewidth=2,
            label="Load Projection",
        )
        ax.fill_between(
            df["timestamp_local"],
            0,
            df["net_power"],
            color="green",
            alpha=0.15,
            label="Netto Solar",
        )

        # Markeer "NU" in lokale tijd
        ax.axvline(local_now, color="black", linestyle="-", alpha=0.5)
        y_max = (
            max(df["pv_estimate"].max(), df["consumption"].max()) * 1.2
        )  # Iets meer ruimte bovenin
        ax.text(local_now, y_max * 0.95, " NU", color="black", fontweight="bold")

        # Markeer geplande start
        if forecast and forecast.planned_start:
            local_start = forecast.planned_start.astimezone(local_tz)
            ax.axvline(
                local_start,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Start ({local_start.strftime('%H:%M')})",
            )

            duration_end = local_start + timedelta(hours=forecaster.optimizer.duration)
            ax.axvspan(local_start, duration_end, color="orange", alpha=0.1)

        # --- AS FORMATTERING (Alleen tijd) ---
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title(
            f"Solar Optimizer: {forecast.action.value}\n{forecast.reason}",
            fontsize=12,
            pad=15,
        )
        ax.set_ylabel("Vermogen (kW)")
        ax.set_xlabel("Tijd")

        # --- RECHTSBOVEN: LEGENDA BOVENAAN ---
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=9,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )

        # --- RECHTSBOVEN: INFOBOX ONDER LEGENDA ---
        info_text = (
            f"PV Nu: {context.stable_pv:.2f} kW\n"
            f"Load Nu: {context.stable_load:.2f} kW\n"
            f"Bias: {forecast.current_bias:.2f}x\n"
            f"Confidence: {forecast.confidence:.1%}"
        )

        # De y-waarde 0.75 plaatst de box net onder een standaard legenda
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="silver")
        ax.text(
            0.98,
            0.70,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, y_max)
        fig.autofmt_xdate()
        fig.tight_layout()

        output = io.BytesIO()
        FigureCanvas(fig).print_png(output, dpi=200)

        return Response(content=output.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"[WebApi] Fout bij genereren grafiek: {e}")
        return {"error": str(e)}
