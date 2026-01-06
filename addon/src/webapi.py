import logging
import io
import matplotlib.dates as mdates

from matplotlib.figure import Figure
from fastapi import FastAPI, Response, Request
from datetime import timedelta, datetime


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

        local_tz = datetime.now().astimezone().tzinfo
        local_now = context.now.astimezone(local_tz)
        df = context.forecast_df.copy()

        df["power_ml"] = forecaster.model.predict(df)
        df["power_corrected"] = forecaster.nowcaster.apply(
            df, context.now, "power_ml", actual_pv=context.stable_pv
        )

        # Load & Net Power
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

        # --- FILTERING (TIJD EN ZON) ---
        # 1. Maximaal 1 uur terug vanaf 'nu'
        one_hour_ago = context.now - timedelta(hours=1)

        # 2. Filter: Tijd >= 1u geleden EN (er is zon-output OF het is dichtbij 'nu')
        # We houden 'nu' altijd in beeld, ook als het donker is.
        mask = (df["timestamp"] >= one_hour_ago) & (
            (df["pv_estimate"] > 0.0)
            | (df["power_corrected"] > 0.0)
            | (df["timestamp"] <= context.now + timedelta(minutes=30))
        )

        df_plot = df[mask].copy()
        df_plot["timestamp_local"] = df_plot["timestamp"].dt.tz_convert(local_tz)

        if df_plot.empty:
            return Response(
                content="Geen relevante data om te tonen (nacht).", status_code=202
            )

        # --- PLOT GENERATIE (DPI=200) ---
        fig = Figure(figsize=(12, 7), dpi=200)
        ax = fig.add_subplot(111)

        # Gebruik df_plot["timestamp"] voor alle plot calls
        ax.plot(
            df_plot["timestamp"],
            df_plot["pv_estimate"],
            "--",
            label="Raw Solcast (Forecast)",
            color="gray",
            alpha=0.4,
        )
        ax.plot(
            df_plot["timestamp"],
            df_plot["power_ml"],
            ":",
            label="ML Model Output",
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            df_plot["timestamp"],
            df_plot["power_corrected"],
            "g-",
            linewidth=2,
            label="Corrected Solar (Nowcast)",
        )
        ax.step(
            df_plot["timestamp"],
            df_plot["consumption"],
            where="post",
            color="red",
            linewidth=2,
            label="Load Projection",
        )
        ax.fill_between(
            df_plot["timestamp"],
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
            # Alleen tekenen als de starttijd binnen ons gefilterde window valt
            if forecast.planned_start >= df_plot["timestamp"].min():
                ax.axvline(
                    forecast.planned_start,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Start ({forecast.planned_start.astimezone(local_tz).strftime('%H:%M')})",
                )
                duration_end = forecast.planned_start + timedelta(
                    hours=forecaster.optimizer.duration
                )
                ax.axvspan(
                    forecast.planned_start, duration_end, color="orange", alpha=0.1
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
            0.99,
            0.80,
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
