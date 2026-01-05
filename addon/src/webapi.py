import logging
import io
import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from fastapi import FastAPI, Response, Request
from datetime import timedelta


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
        forecaster = coordinator.forecaster

        forecast = context.forecast
        df = context.forecast_df.copy()

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

        # 1. Solar lijnen
        ax.plot(
            df["timestamp"],
            df["pv_estimate"],
            "--",
            label="Raw Solcast (Forecast)",
            color="gray",
            alpha=0.4,
        )
        ax.plot(
            df["timestamp"],
            df["power_ml"],
            ":",
            label="ML Model Output",
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            df["timestamp"],
            df["power_corrected"],
            "g-",
            linewidth=2,
            label="Corrected Solar (Nowcast)",
        )

        # 2. Load lijn (Verbruik) - Gebruik step voor realistisch verbruik
        ax.step(
            df["timestamp"],
            df["consumption"],
            where="post",
            color="red",
            linewidth=2,
            label="Geprojecteerd Verbruik (Load)",
        )

        # 3. Netto area
        ax.fill_between(
            df["timestamp"],
            0,
            df["net_power"],
            color="green",
            alpha=0.15,
            label="Netto Beschikbare Solar",
        )

        # 4. Markeer "NU"
        ax.axvline(context.now, color="black", linestyle="-", alpha=0.5)
        # Bepaal y-limiet voor tekstplaatsing
        y_max = max(df["pv_estimate"].max(), df["consumption"].max()) * 1.1
        ax.text(context.now, y_max * 0.9, " NU", color="black", fontweight="bold")

        # 5. Markeer geplande start en het window
        if forecast and forecast.planned_start:
            ax.axvline(
                forecast.planned_start,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Geplande Start ({forecast.planned_start.strftime('%H:%M')})",
            )

            # Teken een blok voor de DHW duration
            duration_end = forecast.planned_start + timedelta(
                hours=forecaster.optimizer.duration
            )
            ax.axvspan(
                forecast.planned_start,
                duration_end,
                color="orange",
                alpha=0.1,
                label="DHW Run Window",
            )

        # --- STYLING & INFO BOX ---
        ax.set_title(
            f"Solar & Load Simulatie\nActie: {forecast.action.value} - {forecast.reason}",
            fontsize=12,
            pad=15,
        )
        ax.set_ylabel("Vermogen (kW)")
        ax.set_xlabel("Tijd (UTC)")

        # Voeg een tekstbox toe met de belangrijkste metrics
        info_text = (
            f"Meting PV: {context.stable_pv:.2f} kW\n"
            f"Huisverbruik: {context.stable_load:.2f} kW\n"
            f"Bias Factor: {forecast.current_bias:.2f}x\n"
            f"Confidence: {forecast.confidence:.1%}"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        ax.text(
            0.02,
            0.95,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, y_max)

        fig.autofmt_xdate()
        fig.tight_layout()

        # --- CONVERTEER NAAR BYTES ---
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)

        return Response(content=output.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"[WebApi] Fout bij genereren grafiek: {e}")
        return {"error": "Could not generate plot"}
