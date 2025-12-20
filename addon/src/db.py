import os
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

from sqlalchemy import (
    create_engine, Column, Integer, Float, DateTime,
    Boolean, JSON, text, select
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SASession

Base = declarative_base()
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    raise RuntimeError("Environment variable DB_PATH is not set.")

engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
)
Session = sessionmaker(bind=engine)

# ==============================================================================
# TABELLEN
# ==============================================================================

class Setpoint(Base):
    """Kern-tabel voor ThermostatAI: Slaat elk leermoment op in eigen kolommen."""
    __tablename__ = "setpoints"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # --- TARGETS (Wat de AI moet leren) ---
    setpoint = Column(Float, index=True) # De 'nieuwe' waarde (User override of AI resultaat)
    observed_current_setpoint = Column(Float) # De waarde 'vóór' de wijziging

    # --- TIJD FEATURES ---
    hour_sin = Column(Float)
    hour_cos = Column(Float)
    day_sin = Column(Float)
    day_cos = Column(Float)
    doy_sin = Column(Float)
    doy_cos = Column(Float)

    # --- STATUS & CONTEXT ---
    home_occupied = Column(Boolean)
    hvac_mode = Column(Integer)
    current_temp = Column(Float)
    temp_change = Column(Float)
    current_setpoint = Column(Float) # Baseline

    # --- WEER ---
    outside_temp = Column(Float)
    min_temp = Column(Float)
    max_temp = Column(Float)
    wind_speed = Column(Float)
    wind_dir_sin = Column(Float)
    wind_dir_cos = Column(Float)
    solar_kwh = Column(Float)

class SolarRecord(Base):
    """Zonne-energie historie voor SolarAI."""
    __tablename__ = "solar_history"
    timestamp = Column(DateTime, primary_key=True)
    solcast_est = Column(Float)
    solcast_10 = Column(Float)
    solcast_90 = Column(Float)
    actual_pv_yield = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class HeatingCycle(Base):
    """Traagheid van het huis voor ThermalAI (Warmtepomp)."""
    __tablename__ = "heating_cycles"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    start_temp = Column(Float)
    end_temp = Column(Float)
    outside_temp = Column(Float)
    duration_minutes = Column(Float)

class PresenceRecord(Base):
    """Aanwezigheidspatronen voor PresenceAI."""
    __tablename__ = "presence_history"
    timestamp = Column(DateTime, primary_key=True)
    is_home = Column(Boolean, index=True)

Base.metadata.create_all(engine)

# ==============================================================================
# FUNCTIES
# ==============================================================================

def insert_setpoint(feature_dict: dict, setpoint: float, observed_current: float):
    """Slaat een leermoment op. Mapt de dict automatisch naar de kolommen."""
    s: SASession = Session()
    try:
        # Filter de dict zodat alleen keys die als kolom bestaan worden gebruikt
        valid_data = {k: v for k, v in feature_dict.items() if hasattr(Setpoint, k)}
        record = Setpoint(
            setpoint=setpoint,
            observed_current_setpoint=observed_current,
            **valid_data
        )
        s.add(record)
        s.commit()
    except Exception:
        logger.exception("DB: Fout bij opslaan setpoint")
        s.rollback()
    finally:
        s.close()

def fetch_training_setpoints_df(days: int = 60):
    """Haalt data op voor ThermostatAI direct als Pandas DataFrame."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(Setpoint).where(
        Setpoint.timestamp >= cutoff,
        Setpoint.setpoint.isnot(None),
        Setpoint.observed_current_setpoint.isnot(None)
    )
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)

def upsert_solar_record(timestamp, **kwargs):
    s: SASession = Session()
    try:
        record = s.get(SolarRecord, timestamp)
        if not record:
            record = SolarRecord(timestamp=timestamp)
            s.add(record)
        for k, v in kwargs.items():
            if hasattr(record, k): setattr(record, k, v)
        s.commit()
    finally: s.close()

def fetch_solar_training_data_orm(days: int = 180):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(SolarRecord).where(SolarRecord.timestamp >= cutoff, SolarRecord.actual_pv_yield.isnot(None))
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)

def upsert_heating_cycle(timestamp, start_temp, end_temp, outside_temp, duration_minutes):
    s: SASession = Session()
    try:
        s.add(HeatingCycle(timestamp=timestamp, start_temp=start_temp, end_temp=end_temp,
                           outside_temp=outside_temp, duration_minutes=duration_minutes))
        s.commit()
    finally: s.close()

def fetch_heating_cycles(days: int = 90):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(HeatingCycle).where(HeatingCycle.timestamp >= cutoff)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)

def upsert_presence_record(timestamp, is_home):
    s: SASession = Session()
    try:
        ts_rounded = timestamp.replace(second=0, microsecond=0) # Voorkom dubbelingen
        record = s.get(PresenceRecord, ts_rounded)
        if not record:
            s.add(PresenceRecord(timestamp=ts_rounded, is_home=is_home))
        else:
            record.is_home = is_home
        s.commit()
    finally: s.close()

def fetch_presence_history(days: int = 60):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(PresenceRecord).where(PresenceRecord.timestamp >= cutoff)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)

def cleanup_old_data(days: int = 365):
    s: SASession = Session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        s.query(Setpoint).filter(Setpoint.timestamp < cutoff).delete()
        s.query(SolarRecord).filter(SolarRecord.timestamp < cutoff).delete()
        s.query(HeatingCycle).filter(HeatingCycle.timestamp < cutoff).delete()
        s.query(PresenceRecord).filter(PresenceRecord.timestamp < cutoff).delete()
        s.commit()
        s.execute(text("VACUUM"))
    finally: s.close()