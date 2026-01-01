import os
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    DateTime,
    Boolean,
    text,
    select,
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
    timestamp = Column(DateTime, default=datetime.now, index=True)
    setpoint = Column(
        Float, index=True
    )  # De 'nieuwe' waarde (User override of AI resultaat)
    current_setpoint = Column(Float)  # Baseline
    home_presence = Column(Boolean)
    hvac_mode = Column(Integer)
    heat_demand = Column(Integer)
    current_temp = Column(Float)
    temp_change = Column(Float)
    outside_temp = Column(Float)
    min_temp = Column(Float)
    max_temp = Column(Float)
    solar_kwh = Column(Float)
    wind_speed = Column(Float)
    wind_dir_sin = Column(Float)
    wind_dir_cos = Column(Float)


class SolarRecord(Base):
    """Zonne-energie historie voor Solar."""

    __tablename__ = "solar_history"
    timestamp = Column(DateTime, primary_key=True)
    pv_estimate = Column(Float)
    pv_estimate10 = Column(Float)
    pv_estimate90 = Column(Float)
    actual_pv_yield = Column(Float, nullable=True)
    temp = Column(Float, nullable=True)
    cloud = Column(Float, nullable=True)
    radiation = Column(Float, nullable=True)
    diffuse = Column(Float, nullable=True)
    irradiance = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PresenceRecord(Base):
    """Aanwezigheidspatronen voor PresenceAI."""

    __tablename__ = "presence_history"
    timestamp = Column(DateTime, primary_key=True)
    is_home = Column(Boolean, index=True)


class HeatingCycle(Base):
    """Opwarm-fase: Hoe snel stijgt de temp als de WP aan staat?"""

    __tablename__ = "heating_cycles"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    start_temp = Column(Float)
    end_temp = Column(Float)
    duration_minutes = Column(Float)
    avg_outside_temp = Column(Float)
    avg_supply_temp = Column(Float)  # Belangrijk voor WP
    rate = Column(Float)  # Graden per uur stijging


class CoolingCycle(Base):
    """Afkoel-fase: Hoe snel zakt de temp als alles UIT staat? (Isolatie)"""

    __tablename__ = "cooling_cycles"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    start_temp = Column(Float)
    end_temp = Column(Float)
    duration_minutes = Column(Float)
    avg_outside_temp = Column(Float)
    rate = Column(Float)  # Graden per uur daling


Base.metadata.create_all(engine)

# ==============================================================================
# FUNCTIES
# ==============================================================================


def insert_setpoint(feature_dict: dict, setpoint: float, observed_current: float):
    """Slaat een leermoment op. Mapt de dict automatisch naar de kolommen."""
    s: SASession = Session()
    try:
        # Filter de dict zodat alleen keys die als kolom bestaan worden gebruikt
        valid_data = {
            k: v
            for k, v in feature_dict.items()
            if hasattr(Setpoint, k) and k not in {"setpoint", "current_setpoint"}
        }
        record = Setpoint(
            setpoint=setpoint, current_setpoint=observed_current, **valid_data
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
        Setpoint.current_setpoint.isnot(None),
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
            if hasattr(record, k):
                setattr(record, k, v)
        s.commit()
    finally:
        s.close()


def fetch_solar_training_data_orm(days: int = 180):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(SolarRecord).where(
        SolarRecord.timestamp >= cutoff, SolarRecord.actual_pv_yield.isnot(None)
    )
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def upsert_heating_cycle(
    timestamp,
    start_temp,
    end_temp,
    duration_minutes,
    avg_outside_temp,
    avg_solar,
    avg_supply_temp,
):
    s: SASession = Session()
    try:
        s.add(
            HeatingCycle(
                timestamp=timestamp,
                start_temp=start_temp,
                end_temp=end_temp,
                duration_minutes=duration_minutes,
                avg_outside_temp=avg_outside_temp,
                avg_solar=avg_solar,
                avg_supply_temp=avg_supply_temp,
            )
        )
        s.commit()
    except Exception:
        logger.exception("DB: Fout bij opslaan heating cycle")
        s.rollback()
    finally:
        s.close()


def fetch_heating_cycles(days: int = 90):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(HeatingCycle).where(HeatingCycle.timestamp >= cutoff)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def upsert_presence_record(timestamp, is_home):
    s: SASession = Session()
    try:
        ts_rounded = timestamp.replace(second=0, microsecond=0)  # Voorkom dubbelingen
        record = s.get(PresenceRecord, ts_rounded)
        if not record:
            s.add(PresenceRecord(timestamp=ts_rounded, is_home=is_home))
        else:
            record.is_home = is_home
        s.commit()
    finally:
        s.close()


def fetch_presence_history(days: int = 60):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(PresenceRecord).where(PresenceRecord.timestamp >= cutoff)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


# ==============================================================================
# FUNCTIES - DHW (SWW)
# ==============================================================================


def save_heating_cycle(start, end, duration_min, outside, supply):
    if duration_min < 15 or (end - start) < 0.1:
        return  # Negeer ruis

    rate = (end - start) / (duration_min / 60.0)  # Graden per uur

    s = Session()
    try:
        s.add(
            HeatingCycle(
                timestamp=datetime.utcnow(),
                start_temp=start,
                end_temp=end,
                duration_minutes=duration_min,
                avg_outside_temp=outside,
                avg_supply_temp=supply,
                rate=rate,
            )
        )
        s.commit()
    except Exception as e:
        logger.error(f"DB Error: {e}")
    finally:
        s.close()


def save_cooling_cycle(start, end, duration_min, outside):
    if duration_min < 60 or (start - end) < 0.1:
        return  # Afkoelen duurt lang

    # Rate = Graden daling per uur per 10 graden verschil binnen/buiten (Normalisatie)
    delta_t = max(1.0, start - outside)
    raw_drop_per_hour = (start - end) / (duration_min / 60.0)

    # Genormaliseerde rate (lekfactor)
    # Hoeveel graden verliezen we per uur als het buiten 0 is en binnen 1?
    normalized_rate = raw_drop_per_hour / delta_t

    s = Session()
    try:
        s.add(
            CoolingCycle(
                timestamp=datetime.utcnow(),
                start_temp=start,
                end_temp=end,
                duration_minutes=duration_min,
                avg_outside_temp=outside,
                rate=normalized_rate,
            )
        )
        s.commit()
    except Exception as e:
        logger.error(f"DB Error: {e}")
    finally:
        s.close()


def fetch_physics_stats(days=60):
    """Haalt de gemiddelde rates op uit de DB voor het Planner model."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    s = Session()
    try:
        # 1. Heating Rate (Mediaan)
        q_heat = s.query(HeatingCycle.rate).filter(HeatingCycle.timestamp >= cutoff)
        heat_rates = [r[0] for r in q_heat.all()]
        avg_heat = sorted(heat_rates)[len(heat_rates) // 2] if heat_rates else 1.0

        # 2. Cooling Rate (Mediaan)
        q_cool = s.query(CoolingCycle.rate).filter(CoolingCycle.timestamp >= cutoff)
        cool_rates = [r[0] for r in q_cool.all()]
        avg_cool = sorted(cool_rates)[len(cool_rates) // 2] if cool_rates else 0.5

        return avg_heat, avg_cool
    finally:
        s.close()


def cleanup_old_data(days: int = 730):
    s: SASession = Session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        s.query(Setpoint).filter(Setpoint.timestamp < cutoff).delete()
        s.query(SolarRecord).filter(SolarRecord.timestamp < cutoff).delete()
        s.query(HeatingCycle).filter(HeatingCycle.timestamp < cutoff).delete()
        s.query(PresenceRecord).filter(PresenceRecord.timestamp < cutoff).delete()
        s.commit()
        s.execute(text("VACUUM"))
    finally:
        s.close()
