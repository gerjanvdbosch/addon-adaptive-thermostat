import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Header
from pydantic import BaseModel
from sqlalchemy import select, desc

# Project imports
from db import Session, Setpoint, SolarRecord, PresenceRecord, HeatingCycle

logger = logging.getLogger(__name__)

app = FastAPI(title="Adaptive Thermostat API")


# ==============================================================================
# Pydantic Models (Output formatting)
# ==============================================================================


class SetpointOut(BaseModel):
    id: int
    timestamp: datetime
    setpoint: Optional[float]
    current_setpoint: Optional[float]
    home_presence: Optional[bool]
    hvac_mode: Optional[int]
    heat_demand: Optional[int]
    current_temp: Optional[float]
    outside_temp: Optional[float]
    solar_kwh: Optional[float]

    class Config:
        from_attributes = True


class SolarOut(BaseModel):
    timestamp: datetime
    solcast_est: Optional[float]
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
# HISTORY ENDPOINTS (READ ONLY)
# ==============================================================================


@app.get("/history/thermostat", response_model=List[SetpointOut])
def get_setpoint_history(
    limit: int = 100, offset: int = 0, x_addon_token: Optional[str] = Header(None)
):
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
def get_solar_history(
    limit: int = 100, offset: int = 0, x_addon_token: Optional[str] = Header(None)
):
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
def get_presence_history(
    limit: int = 100, offset: int = 0, x_addon_token: Optional[str] = Header(None)
):
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
def get_thermal_history(
    limit: int = 100, offset: int = 0, x_addon_token: Optional[str] = Header(None)
):
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
