import logging
import threading
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc

# Project imports
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
    wind_speed: Optional[float]
    wind_dir_sin: Optional[float]
    wind_dir_cos: Optional[float]
    solar_kwh: Optional[float]

    class Config:
        from_attributes = True


class SolarOut(BaseModel):
    timestamp: datetime
    solcast_est: Optional[float]
    solcast_10: Optional[float]
    solcast_90: Optional[float]
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
    outside_temp: Optional[float]
    duration_minutes: Optional[float]

    class Config:
        from_attributes = True


class TrainResponse(BaseModel):
    status: str
    target: str
    background: bool


# ==============================================================================
# TRAINING & CONTROL ENDPOINTS
# ==============================================================================


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
