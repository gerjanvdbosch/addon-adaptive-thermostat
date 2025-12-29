import logging
import threading
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fastapi.responses import StreamingResponse
from unittest.mock import MagicMock, patch

# Importeer je classes
from solar import SolarAI
from datetime import datetime, timedelta
from typing import List, Optional
from utils import safe_bool
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from db import (
    Session,
    Setpoint,
    SolarRecord,
    PresenceRecord,
    HeatingCycle,
    DhwSession,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Adaptive Thermostat API")

# ==============================================================================
# GLOBAL STATE (Koppeling met Coordinator)
# ==============================================================================
GLOBAL_COORDINATOR = None


def set_coordinator(coordinator_instance):
    """
    Wordt aangeroepen vanuit main.py om de actieve coordinator door te geven.
    Hierdoor kan de API acties uitvoeren op de draaiende AI modellen.
    """
    global GLOBAL_COORDINATOR
    GLOBAL_COORDINATOR = coordinator_instance
    logger.info("API: Coordinator linked successfully.")


# ==============================================================================
# Pydantic Models
# ==============================================================================


class SetpointOut(BaseModel):
    id: int
    timestamp: datetime
    setpoint: Optional[float]
    current_setpoint: Optional[float]
    hvac_mode: Optional[int]
    heat_demand: Optional[int]
    current_temp: Optional[float]
    temp_change: Optional[float]
    home_presence: Optional[bool]
    outside_temp: Optional[float]
    min_temp: Optional[float]
    max_temp: Optional[float]
    solar_kwh: Optional[float]
    wind_speed: Optional[float]
    wind_dir_sin: Optional[float]
    wind_dir_cos: Optional[float]

    class Config:
        from_attributes = True


class SolarOut(BaseModel):
    timestamp: datetime
    pv_estimate: Optional[float]
    pv_estimate10: Optional[float]
    pv_estimate90: Optional[float]
    actual_pv_yield: Optional[float]

    class Config:
        from_attributes = True


class PresenceOut(BaseModel):
    timestamp: datetime
    is_home: bool

    class Config:
        from_attributes = True


class ThermalOut(BaseModel):
    id: int
    timestamp: datetime
    start_temp: Optional[float]
    end_temp: Optional[float]
    duration_minutes: Optional[float]
    avg_outside_temp: Optional[float]
    avg_solar: Optional[float]
    avg_supply_temp: Optional[float]

    class Config:
        from_attributes = True


class DhwOut(BaseModel):
    timestamp: datetime
    sensor_id: int
    value: float
    hvac_mode: Optional[float]

    class Config:
        from_attributes = True


# ------------------------


class TrainResponse(BaseModel):
    status: str
    target: str
    background: bool


class DeleteResponse(BaseModel):
    status: str
    deleted_key: str


class AIStatus(BaseModel):
    thermostat: dict
    solar: dict
    presence: dict
    thermal: dict
    dhw: dict
    system: dict


# ==============================================================================
# TRAINING & CONTROL ENDPOINTS
# ==============================================================================


@app.get("/status", response_model=AIStatus)
def get_current_status():
    """
    Haalt de actuele voorspellingen en status op van alle AI modellen.
    Dit is de 'live' weergave van wat het systeem nu denkt en plant.
    """

    if GLOBAL_COORDINATOR is None:
        raise HTTPException(status_code=503, detail="Coordinator not linked.")

    try:
        # 1. Verzamel de meest recente sensor data voor context
        raw_data = GLOBAL_COORDINATOR.collector.read_sensors()
        features = GLOBAL_COORDINATOR.collector.features_from_raw(raw_data)

        cur_sp = features.get("current_setpoint", 0.0)
        cur_temp = features.get("current_temp", 0.0)

        # Haal de invloeden op (Thermostaat)
        thermostat_influences = GLOBAL_COORDINATOR.thermostat_ai.get_influence_factors(
            features, cur_sp
        )

        # 2. Vraag Thermostaat voorspelling
        rec_sp = GLOBAL_COORDINATOR.thermostat_ai.get_recommended_setpoint(
            features, cur_sp
        )

        # ----------------------------------------------------------------------
        # 3. Vraag Solar aanbeveling & Invloeden (Uitgebreid)
        # ----------------------------------------------------------------------
        solar_ai = GLOBAL_COORDINATOR.solar_ai
        solar_rec = solar_ai.get_solar_recommendation()

        # Enum handling: Zorg dat we de string waarde krijgen
        action_val = solar_rec.get("action")
        status_str = (
            action_val.value if hasattr(action_val, "value") else str(action_val)
        )

        # Bereken de Readable Influences (Live)
        solar_influences = {}
        if solar_ai.cached_solcast_data:
            now_utc = pd.Timestamp.now(tz="UTC")
            df_now = pd.DataFrame(solar_ai.cached_solcast_data)
            df_now["timestamp"] = pd.to_datetime(df_now["period_start"], utc=True)

            # Zoek dichtstbijzijnde record
            df_now["time_diff"] = (df_now["timestamp"] - now_utc).abs()
            nearest_row = df_now.nsmallest(1, "time_diff")

            if not nearest_row.empty:
                try:
                    X_now = solar_ai._create_features(nearest_row)
                    solar_influences = solar_ai.get_influence_factors(X_now)
                except Exception as e:
                    logger.warning(f"WebAPI: Kon solar influences niet berekenen: {e}")

        # ----------------------------------------------------------------------

        # 4. Vraag Thermal voorspelling (hoe lang duurt opwarmen naar cur_sp)
        heating_mins = GLOBAL_COORDINATOR.thermal_ai.predict_heating_time(
            cur_sp, features
        )

        thermal_influences = GLOBAL_COORDINATOR.thermal_ai.get_influence_factors(
            cur_sp, features
        )

        # 5. Vraag Presence voorspelling (kans op thuiskomst binnen de opwarmtijd)
        should_preheat, prob = GLOBAL_COORDINATOR.presence_ai.should_preheat(
            dynamic_minutes=heating_mins
        )

        target_time = datetime.now() + timedelta(minutes=(heating_mins or 0))
        presence_influences = GLOBAL_COORDINATOR.presence_ai.get_influence_factors(
            target_time
        )

        sww_temp = features.get("dhw_temp", 45.0)

        dhw_rec = GLOBAL_COORDINATOR.dhw_ai.get_recommendation(
            sww_temp, solar_rec.get("action")
        )

        # Vraag invloeden op (voor het lookahead moment)
        lookahead = GLOBAL_COORDINATOR.dhw_ai.lookahead_minutes
        target_ts = datetime.now() + timedelta(minutes=lookahead)
        dhw_influences = GLOBAL_COORDINATOR.dhw_ai.get_influence_factors(target_ts)

        return {
            "thermostat": {
                "current_setpoint": cur_sp,
                "recommended_setpoint": round(rec_sp, 2),
                "delta": round(rec_sp - cur_sp, 2),
                "explanation": thermostat_influences,
            },
            "solar": {
                "status": status_str,
                "reason": solar_rec.get("reason"),
                "planned_start": solar_rec.get("plan_start"),
                "bias": round(solar_ai.smoothed_bias, 2),
                "explanation": solar_influences,
            },
            "presence": {
                "is_home": safe_bool(features.get("home_presence")),
                "preheat_trigger": should_preheat,
                "probability_home_soon": round(prob, 2),
                "lookahead_minutes": round(heating_mins or 0),
                "explanation": presence_influences,
            },
            "thermal": {
                "current_temp": cur_temp,
                "predicted_minutes_to_reach_target": round(heating_mins or 0),
                "explanation": thermal_influences,
            },
            "dhw": {
                "current_temp": sww_temp,
                "target_temp": dhw_rec,
                "explanation": dhw_influences,
            },
            "system": {
                "hvac_mode": GLOBAL_COORDINATOR._get_hvac_mode(raw_data),
                "compressor_active": GLOBAL_COORDINATOR._is_compressor_active(
                    GLOBAL_COORDINATOR._get_hvac_mode(raw_data)
                ),
                "last_switch_time": GLOBAL_COORDINATOR.last_switch_time,
            },
        }
    except Exception as e:
        logger.exception(f"API: Error fetching status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse)
def trigger_training(
    model: str = Query(
        "all",
        description="Model to train: all, thermostat, solar, presence, thermal, dhw",
    )
):
    """
    Start handmatig een trainingssessie.
    Draait in de achtergrond zodat de API direct antwoord geeft.
    """

    if GLOBAL_COORDINATOR is None:
        raise HTTPException(
            status_code=503, detail="Coordinator not linked. System starting up?"
        )

    def _train_task():
        logger.info(f"API: Manual training triggered for '{model}'")
        try:
            if model == "all":
                GLOBAL_COORDINATOR.perform_nightly_training()
            elif model == "thermostat":
                GLOBAL_COORDINATOR.thermostat_ai.train()
            elif model == "solar":
                GLOBAL_COORDINATOR.solar_ai.train()
            elif model == "presence":
                GLOBAL_COORDINATOR.presence_ai.train()
            elif model == "thermal":
                GLOBAL_COORDINATOR.thermal_ai.train()
            elif model == "dhw":
                GLOBAL_COORDINATOR.dhw_ai.train()
            else:
                logger.warning(f"API: Unknown model type '{model}'")
        except Exception as e:
            logger.exception(f"API: Training failed for {model}: {e}")

    # Start thread
    t = threading.Thread(target=_train_task, daemon=True)
    t.start()

    return {"status": "started", "target": model, "background": True}


# ==============================================================================
# HISTORY ENDPOINTS (READ ONLY)
# ==============================================================================


@app.get("/history/setpoint", response_model=List[SetpointOut])
def get_setpoint_history(limit: int = 100, offset: int = 0):
    s = Session()
    try:
        stmt = (
            select(Setpoint)
            .order_by(desc(Setpoint.timestamp))
            .limit(limit)
            .offset(offset)
        )
        results = s.execute(stmt).scalars().all()
        return results
    finally:
        s.close()


@app.get("/history/solar", response_model=List[SolarOut])
def get_solar_history(limit: int = 100, offset: int = 0):
    s = Session()
    try:
        stmt = (
            select(SolarRecord)
            .order_by(desc(SolarRecord.timestamp))
            .limit(limit)
            .offset(offset)
        )
        results = s.execute(stmt).scalars().all()
        return results
    finally:
        s.close()


@app.get("/history/presence", response_model=List[PresenceOut])
def get_presence_history(limit: int = 100, offset: int = 0):
    s = Session()
    try:
        stmt = (
            select(PresenceRecord)
            .order_by(desc(PresenceRecord.timestamp))
            .limit(limit)
            .offset(offset)
        )
        results = s.execute(stmt).scalars().all()
        return results
    finally:
        s.close()


@app.get("/history/thermal", response_model=List[ThermalOut])
def get_thermal_history(limit: int = 100, offset: int = 0):
    s = Session()
    try:
        stmt = (
            select(HeatingCycle)
            .order_by(desc(HeatingCycle.timestamp))
            .limit(limit)
            .offset(offset)
        )
        results = s.execute(stmt).scalars().all()
        return results
    finally:
        s.close()


@app.get("/history/dhw", response_model=List[DhwOut])
def get_dhw_history(limit: int = 100, offset: int = 0):
    s = Session()
    try:
        stmt = (
            select(DhwSession)
            .order_by(desc(DhwSession.timestamp))
            .limit(limit)
            .offset(offset)
        )
        results = s.execute(stmt).scalars().all()
        return results
    finally:
        s.close()


# --------------------------


# ==============================================================================
# DELETE ENDPOINTS
# ==============================================================================


@app.delete("/history/setpoint/{item_id}", response_model=DeleteResponse)
def delete_setpoint(item_id: int):
    """Verwijder een setpoint record op basis van ID."""
    s = Session()
    try:
        record = s.get(Setpoint, item_id)
        if not record:
            raise HTTPException(status_code=404, detail="Setpoint record not found")

        s.delete(record)
        s.commit()
        return {"status": "success", "deleted_key": str(item_id)}
    except Exception as e:
        s.rollback()
        logger.error(f"Failed to delete setpoint {item_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        s.close()


@app.delete("/history/thermal/{item_id}", response_model=DeleteResponse)
def delete_thermal(item_id: int):
    """Verwijder een thermal record op basis van ID."""
    s = Session()
    try:
        record = s.get(HeatingCycle, item_id)
        if not record:
            raise HTTPException(status_code=404, detail="HeatingCycle record not found")

        s.delete(record)
        s.commit()
        return {"status": "success", "deleted_key": str(item_id)}
    except Exception as e:
        s.rollback()
        logger.error(f"Failed to delete thermal {item_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        s.close()


@app.delete("/history/solar/{timestamp}", response_model=DeleteResponse)
def delete_solar(timestamp: datetime):
    """
    Verwijder een solar record op basis van timestamp (Primary Key).
    Let op: Timestamp moet URL-encoded zijn (bijv. 2025-12-21T10:00:00Z)
    """
    s = Session()
    try:
        record = s.get(SolarRecord, timestamp)
        if not record:
            raise HTTPException(status_code=404, detail="Solar record not found")

        s.delete(record)
        s.commit()
        return {"status": "success", "deleted_key": timestamp.isoformat()}
    except Exception as e:
        s.rollback()
        logger.error(f"Failed to delete solar {timestamp}: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        s.close()


@app.delete("/history/presence/{timestamp}", response_model=DeleteResponse)
def delete_presence(timestamp: datetime):
    """
    Verwijder een presence record op basis van timestamp (Primary Key).
    """
    s = Session()
    try:
        record = s.get(PresenceRecord, timestamp)
        if not record:
            raise HTTPException(status_code=404, detail="Presence record not found")

        s.delete(record)
        s.commit()
        return {"status": "success", "deleted_key": timestamp.isoformat()}
    except Exception as e:
        s.rollback()
        logger.error(f"Failed to delete presence {timestamp}: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        s.close()


@app.delete("/history/dhw/{timestamp}/{sensor_id}", response_model=DeleteResponse)
def delete_dhw(timestamp: datetime, sensor_id: int):
    """
    Verwijder een DHW record. Omdat de Primary Key waarschijnlijk samengesteld is
    (timestamp + sensor_id), hebben we beide nodig.
    """
    s = Session()
    try:
        # We gebruiken s.get() met een tuple voor composite keys
        record = s.get(DhwSession, (timestamp, sensor_id))
        if not record:
            raise HTTPException(status_code=404, detail="DHW record not found")

        s.delete(record)
        s.commit()
        return {
            "status": "success",
            "deleted_key": f"{timestamp.isoformat()}_{sensor_id}",
        }
    except Exception as e:
        s.rollback()
        logger.error(f"Failed to delete dhw {timestamp} / {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        s.close()


@app.get("/simulation/solar")
def get_solar_simulation_plot(
    date: str = Query(None, description="Datum in YYYY-MM-DD formaat.")
):
    """
    Simuleert exact het gedrag van het script:
    1. Voorspellen (met oude kennis)
    2. Meten (nieuwe waarde onthullen)
    3. Leren (bias updaten)
    """
    if GLOBAL_COORDINATOR is None:
        raise HTTPException(status_code=503, detail="Coordinator niet geladen.")

    # --- 1. DATUM & DATA OPHALEN ---
    target_date = date if date else datetime.now().strftime("%Y-%m-%d")
    try:
        start_ts = datetime.strptime(target_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Ongeldig datumformaat.")

    end_ts = start_ts + timedelta(days=1)

    s = Session()
    try:
        stmt = (
            select(SolarRecord)
            .where(SolarRecord.timestamp >= start_ts)
            .where(SolarRecord.timestamp < end_ts)
            .order_by(SolarRecord.timestamp)
        )
        records = s.execute(stmt).scalars().all()
        if not records:
            raise HTTPException(
                status_code=404, detail=f"Geen data gevonden voor {target_date}."
            )

        df = pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "pv_estimate": r.pv_estimate,
                    "pv_estimate10": r.pv_estimate10,
                    "pv_estimate90": r.pv_estimate90,
                    "actual_pv_yield": r.actual_pv_yield,
                }
                for r in records
            ]
        )
    finally:
        s.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)

    # Resample naar 1 minuut voor de simulatie loop
    df_sim = df.resample("1min").interpolate(method="linear")

    # --- 2. MOCK OMGEVING ---
    mock_ha = MagicMock()
    sim_states = {}

    # We beginnen met 0 vermogen (nacht)
    sim_states["sensor.mock_pv"] = "0.0"

    mock_ha.get_state.side_effect = lambda entity_id: sim_states.get(entity_id, "0.0")

    # Forecast payload (Solcast geeft dit in 1x voor de hele dag)
    forecast_payload = []
    for ts, row in df_sim.iterrows():
        forecast_payload.append(
            {
                "period_start": ts.isoformat(),
                "pv_estimate": row["pv_estimate"],
                "pv_estimate10": row["pv_estimate10"],
                "pv_estimate90": row["pv_estimate90"],
            }
        )

    mock_ha.get_payload.return_value = {
        "attributes": {"detailedForecast": forecast_payload}
    }

    # Instellingen overnemen van productie
    real_ai = GLOBAL_COORDINATOR.solar_ai
    opts = {
        "system_max_kw": real_ai.system_max_kw,
        "duration_hours": real_ai.duration_hours,
        "min_viable_kw": real_ai.min_viable_kw,
        "state_length": 1,
        "solar_interval_seconds": 60,
        "sensor_pv_power": "sensor.mock_pv",
        "sensor_solcast_poll": "sensor.mock_poll",
    }

    sim_ai = SolarAI(mock_ha, opts)

    # Brein overnemen (Model & Weights)
    if real_ai.is_fitted:
        sim_ai.model = real_ai.model
        sim_ai.is_fitted = True
        sim_ai.ml_weight = real_ai.ml_weight
        sim_ai.solcast_weight = real_ai.solcast_weight

    # --- 3. TIJD MOCKING FUNCTIES ---
    time_ref = {"current": pd.Timestamp(start_ts).tz_localize("UTC")}

    def fake_now_ts(tz=None):
        t = time_ref["current"]
        return t.tz_convert(tz) if tz else t

    def fake_now_dt(tz=None):
        return fake_now_ts(tz).to_pydatetime()

    # --- 4. DE SIMULATIE LOOP ---
    results = []

    # Variabele om de waarde van de VORIGE iteratie vast te houden
    last_known_pv = 0.0

    # Patches: Datetime, Pandas Timestamp en DB-writes blokkeren
    with patch("solar.datetime") as mock_datetime, patch(
        "solar.pd.Timestamp.now", side_effect=fake_now_ts
    ), patch("solar.upsert_solar_record"):

        mock_datetime.now.side_effect = fake_now_dt
        sim_states["sensor.mock_poll"] = start_ts.isoformat()

        for current_sim_time in df_sim.index:
            # 1. Tijd zetten
            time_ref["current"] = current_sim_time

            # Huidige ECHTE waarde uit data halen
            actual_val_now = df_sim.loc[current_sim_time, "actual_pv_yield"]
            actual_val_now = actual_val_now if pd.notna(actual_val_now) else 0.0

            # -----------------------------------------------------------
            # STAP A: INPUT ZETTEN (Oude situatie)
            # -----------------------------------------------------------
            # We zetten de sensor op wat we WISTEN (vorige minuut).
            # Dit voorkomt dat de predictie 'spiekt' naar de toekomst.
            sim_states["sensor.mock_pv"] = str(last_known_pv * 1000)

            # -----------------------------------------------------------
            # STAP B: VOORSPELLING BEREKENEN (Wat DENKT de AI?)
            # -----------------------------------------------------------
            bias_at_decision_moment = sim_ai.smoothed_bias
            raw_solcast = df_sim.loc[current_sim_time, "pv_estimate"]

            # Bereken exact zoals SolarAI dat intern doet
            ai_mixed_power = raw_solcast

            if sim_ai.is_fitted and sim_ai.model:
                try:
                    # Features maken op basis van tijd (en potentieel 'last_known_pv' als feature)
                    row_df = df_sim.loc[[current_sim_time]].copy()
                    row_df["timestamp"] = row_df.index

                    feat = sim_ai._create_features(row_df)
                    pred_ml = sim_ai.model.predict(feat)[0]

                    ai_mixed_power = (sim_ai.ml_weight * pred_ml) + (
                        sim_ai.solcast_weight * raw_solcast
                    )
                except Exception:
                    pass

            # Pas bias toe (DIT IS DE WAARDE IN DE GRAFIEK 'AI Pred')
            final_ai_pred = max(
                0.0,
                min((ai_mixed_power * bias_at_decision_moment), sim_ai.system_max_kw),
            )

            # -----------------------------------------------------------
            # STAP C: WERKELIJKHEID ONTHULLEN & UPDATEN
            # -----------------------------------------------------------
            # Nu zetten we de sensor op de actuele waarde van NU.
            sim_states["sensor.mock_pv"] = str(actual_val_now * 1000)

            # Run Cycle: De AI ziet nu het verschil tussen zijn inschatting en de werkelijkheid
            # en past zijn bias aan voor de VOLGENDE minuut.
            sim_ai.run_cycle()

            # Voorbereiden voor volgende loop
            last_known_pv = actual_val_now

            # -----------------------------------------------------------
            # STAP D: OPSLAAN
            # -----------------------------------------------------------
            res = sim_ai.last_stable_advice
            ctx = res.get("context")

            results.append(
                {
                    "time": current_sim_time,
                    "pv": actual_val_now,
                    "forecast": raw_solcast,
                    "p10": df_sim.loc[current_sim_time, "pv_estimate10"],
                    "p90": df_sim.loc[current_sim_time, "pv_estimate90"],
                    "ai_pred": final_ai_pred,
                    "threshold": ctx.trigger_threshold_kw if ctx else 0.0,
                    "status": res["action"].value,
                    "bias": bias_at_decision_moment,  # De bias die gebruikt is voor DEZE beslissing
                }
            )

    # --- 5. PLOTTEN ---
    res_df = pd.DataFrame(results)

    # Filter: Alleen relevante uren (beetje marge rondom activiteit)
    is_active = (res_df["forecast"] > 0.001) | (res_df["pv"] > 0.001)
    if is_active.any():
        start_pos = max(0, res_df.index[is_active][0] - 60)
        end_pos = min(len(res_df), res_df.index[is_active][-1] + 60)
        res_df = res_df.iloc[start_pos:end_pos]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. Onzekerheidsbanden (Achtergrond)
    ax.fill_between(
        res_df["time"],
        res_df["p10"],
        res_df["p90"],
        color="#3399FF",
        alpha=0.05,
        label="Solcast Range",
    )
    ax.plot(res_df["time"], res_df["p10"], color="#3399FF", linestyle=":", alpha=0.4)
    ax.plot(res_df["time"], res_df["p90"], color="#3399FF", linestyle=":", alpha=0.4)

    # 2. Start Zones (Groen vlak)
    is_start = res_df["status"] == "START"
    has_data = res_df["pv"].notna()

    # Bepaal Y-as schaal voor netjes inkleuren
    y_max = res_df[["ai_pred", "pv", "forecast"]].max().max()
    if pd.isna(y_max) or y_max < 0.1:
        y_max = 1.0

    ax.fill_between(
        res_df["time"],
        0,
        y_max * 1.1,
        where=(is_start & has_data),
        color="green",
        alpha=0.1,
        label="Status: START",
    )

    # 3. De Lijnen
    # Solcast (Raw)
    ax.plot(
        res_df["time"],
        res_df["forecast"],
        label="Solcast Forecast",
        color="#004080",
        linestyle="--",
        alpha=0.6,
    )

    # Werkelijke PV (Oranje)
    if res_df["pv"].notna().any():
        ax.plot(res_df["time"], res_df["pv"], label="Actueel PV", color="orange", lw=2)

    # SolarAI Prediction (Donkerblauw, dik) -> Deze moet nu loslopen van oranje!
    ax.plot(
        res_df["time"],
        res_df["ai_pred"],
        label="SolarAI Predictie",
        color="#004080",
        lw=2,
    )

    # Trigger Drempel (Rood)
    ax.plot(
        res_df["time"],
        res_df["threshold"],
        label="Drempel",
        color="red",
        linestyle="--",
        linewidth=1,
    )

    # 4. Opmaak
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.title(f"SolarAI Simulatie: {target_date}")
    plt.ylabel("Vermogen (kW)")
    plt.xlabel("Tijd (UTC)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
