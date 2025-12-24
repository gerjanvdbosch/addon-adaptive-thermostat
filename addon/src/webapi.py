import logging
import threading
import pandas as pd

from datetime import datetime
from typing import List, Optional
from utils import safe_bool
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from db import Session, Setpoint, SolarRecord, PresenceRecord, HeatingCycle

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
        influences_thermostat = GLOBAL_COORDINATOR.thermostat_ai.get_influence_factors(
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
                # Gebruik de interne functies van SolarAI om features te maken
                # Dit is veilig omdat we de instantie delen
                try:
                    X_now = solar_ai._create_features(nearest_row)
                    solar_influences = solar_ai._get_readable_influences(X_now)
                except Exception as e:
                    logger.warning(f"API: Kon solar influences niet berekenen: {e}")

        # ----------------------------------------------------------------------

        # 4. Vraag Thermal voorspelling (hoe lang duurt opwarmen naar comfort?)
        comfort_temp = GLOBAL_COORDINATOR.settings.get("home_fallback", 20.0)
        heating_mins = GLOBAL_COORDINATOR.thermal_ai.predict_heating_time(
            comfort_temp, features
        )

        # 5. Vraag Presence voorspelling (kans op thuiskomst binnen de opwarmtijd)
        should_preheat, prob = GLOBAL_COORDINATOR.presence_ai.should_preheat(
            dynamic_minutes=heating_mins
        )

        return {
            "thermostat": {
                "current_setpoint": cur_sp,
                "recommended_setpoint": round(rec_sp, 2),
                "delta": round(rec_sp - cur_sp, 2),
                "explanation": influences_thermostat,
            },
            "solar": {
                "status": status_str,
                "reason": solar_rec.get("reason"),
                "planned_start": solar_rec.get("plan_start"),
                "bias": round(solar_ai.smoothed_bias, 2),
                "influences": solar_influences,  # De nieuwe feature!
            },
            "presence": {
                "is_home": safe_bool(features.get("home_presence")),
                "preheat_trigger": should_preheat,
                "probability_home_soon": round(prob, 2),
                "lookahead_minutes": round(heating_mins or 0),
            },
            "thermal": {
                "current_temp": cur_temp,
                "target_comfort": comfort_temp,
                "predicted_minutes_to_reach_target": round(heating_mins or 0),
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
        "all", description="Model to train: all, thermostat, solar, presence, thermal"
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
