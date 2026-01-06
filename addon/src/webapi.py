import logging
import io
import matplotlib.dates as mdates

from matplotlib.figure import Figure
from fastapi import FastAPI, Response, Request
from datetime import timedelta
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")


@api.get("/")
def index():
    return {"status": "ok"}


@api.get("/solar/forecast")
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

        local_tz = ZoneInfo("Europe/Amsterdam")
        local_now = context.now.astimezone(local_tz)
        logger.info(
            f"[WebApi] Generating solar forecast plot for {local_now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

        df = context.forecast_df.copy()
        df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz)

        df["power_ml"] = forecaster.model.predict(df)
        df["power_corrected"] = forecaster.nowcaster.apply(
            df, context.now, "power_ml", actual_pv=context.stable_pv
        )

        # Load & Net Power projectie
        baseload = forecaster.optimizer.avg_baseload
        df["consumption"] = baseload

        future_mask = df["timestamp_local"] >= local_now
        future_indices = df.index[future_mask]
        decay_steps = 3
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
            local_start = forecast.planned_start.astimezone(local_tz)
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

                duration_end = local_start + timedelta(
                    hours=forecaster.optimizer.duration
                )
                ax.axvspan(
                    local_start, duration_end, color="orange", alpha=0.12, zorder=1
                )

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

        # Infobox (Onder Legenda)
        info_text = (
            f"PV Nu: {context.stable_pv:.2f} kW\n"
            f"Load Nu: {context.stable_load:.2f} kW\n"
            f"Bias: {forecast.current_bias:.2f}x\n"
            f"Confidence: {forecast.confidence:.1%}"
        )
        props = dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="silver"
        )
        ax.text(
            0.95,
            0.78,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_ylim(0, y_max)
        fig.autofmt_xdate()

        # EXPORT
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=200, bbox_inches="tight")
        output.seek(0)

        return Response(content=output.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"[WebApi] Grafiek fout: {e}", exc_info=True)
        return {"error": str(e)}
