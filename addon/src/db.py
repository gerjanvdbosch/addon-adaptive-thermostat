import os
from datetime import datetime, timedelta
import pandas as pd

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
    text,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SASession

Base = declarative_base()

DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    raise RuntimeError(
        "Environment variable DB_PATH is not set. Please export DB_PATH before starting the application."
    )

engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
)
Session = sessionmaker(bind=engine)


class Sample(Base):
    __tablename__ = "samples"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    data = Column(JSON)
    label_setpoint = Column(Float, nullable=True)
    user_override = Column(Boolean, default=False)
    predicted_setpoint = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)


class Setpoint(Base):
    __tablename__ = "setpoints"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    # setpoint and observed_current_setpoint == override
    setpoint = Column(Float, nullable=True)
    observed_current_setpoint = Column(Float, nullable=True)
    data = Column(JSON)


class SolarRecord(Base):
    __tablename__ = "solar_history"

    # Timestamp is de Primary Key (uniek en geïndexeerd)
    timestamp = Column(DateTime, primary_key=True)

    # Forecast Data
    solcast_est = Column(Float, default=0.0)
    solcast_10 = Column(Float, default=0.0)
    solcast_90 = Column(Float, default=0.0)

    # Actuals (Gemiddelden over 30 min)
    actual_pv_yield = Column(Float, nullable=True)
    actual_consumption = Column(Float, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def upsert_solar_record(timestamp, **kwargs):
    """
    Maakt een record aan of update het als het bestaat.
    Gebruikt ORM sessie management.
    """
    s: SASession = Session()
    try:
        record = s.get(SolarRecord, timestamp)
        if not record:
            record = SolarRecord(timestamp=timestamp)
            s.add(record)

        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.updated_at = datetime.utcnow()
        s.commit()
    except Exception as e:
        print(f"DB Error upsert_solar_record: {e}")
        s.rollback()
    finally:
        s.close()


def fetch_solar_training_data_orm(days: int = 365):
    """
    Haalt trainingsdata op via ORM, maar geoptimaliseerd voor Pandas.
    Geeft een DataFrame terug.
    """
    cutoff = datetime.now() - timedelta(days=days)

    # We selecteren alleen de kolommen die we nodig hebben voor training
    # Dit is veel sneller dan het laden van volledige Python objecten
    stmt = (
        select(
            SolarRecord.timestamp,
            SolarRecord.solcast_est,
            SolarRecord.solcast_10,
            SolarRecord.solcast_90,
            SolarRecord.actual_pv_yield,
        )
        .where(SolarRecord.timestamp >= cutoff)
        .where(SolarRecord.actual_pv_yield.isnot(None))
    )

    # Pandas kan direct lezen van een SQLAlchemy connectie/statement
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    # Zorg dat timestamp correcte types heeft
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def cleanup_old_data(days: int = 730):
    """Onderhoud: Verwijder data ouder dan X dagen."""
    s: SASession = Session()
    try:
        cutoff = datetime.now() - timedelta(days=days)

        # Bulk delete via ORM
        s.query(SolarRecord).filter(SolarRecord.timestamp < cutoff).delete(
            synchronize_session=False
        )
        s.query(Sample).filter(Sample.timestamp < cutoff).delete(
            synchronize_session=False
        )

        s.commit()

        # SQLite Vacuum om schijfruimte vrij te geven
        s.execute(text("VACUUM"))
    except Exception as e:
        print(f"Cleanup Error: {e}")
    finally:
        s.close()


def insert_setpoint(
    data: dict,
    setpoint=None,
    observed_current=None,
) -> int:
    s: SASession = Session()
    try:
        sample = Setpoint(
            data=data, setpoint=setpoint, observed_current_setpoint=observed_current
        )
        s.add(sample)
        s.commit()
        s.refresh(sample)
        return sample.id
    finally:
        s.close()


def update_setpoint(setpoint_id: int, setpoint: float, observed_current=None) -> None:
    s: SASession = Session()
    try:
        row = s.get(Setpoint, setpoint_id)
        if row is not None:
            row.setpoint = setpoint
        if observed_current is not None:
            row.observed_current_setpoint = observed_current
        s.commit()
    finally:
        s.close()


def fetch_setpoints(limit: int = 1):
    s: SASession = Session()
    try:
        rows = s.query(Setpoint).order_by(Setpoint.timestamp.desc()).limit(limit).all()
        return rows
    finally:
        s.close()


def fetch_unlabeled_setpoints(limit: int = 1):
    s: SASession = Session()
    try:
        rows = (
            s.query(Setpoint)
            .filter(Setpoint.observed_current_setpoint.is_(None))
            .filter(Setpoint.setpoint.is_(None))
            .order_by(Setpoint.timestamp.desc())
            .limit(limit)
            .all()
        )
        return rows
    finally:
        s.close()


def fetch_training_setpoints(days: int = 30):
    s: SASession = Session()
    try:
        cutoff = datetime.now() - timedelta(days=days)
        rows = (
            s.query(Setpoint)
            .filter(Setpoint.timestamp >= cutoff)
            .filter(Setpoint.setpoint.isnot(None))
            .filter(Setpoint.observed_current_setpoint.isnot(None))
            .all()
        )
        return rows
    finally:
        s.close()


def insert_sample(data: dict, label_setpoint=None, user_override: bool = False) -> int:
    s: SASession = Session()
    try:
        sample = Sample(
            data=data, label_setpoint=label_setpoint, user_override=user_override
        )
        s.add(sample)
        s.commit()
        s.refresh(sample)
        return sample.id
    finally:
        s.close()


def fetch_training_data(days: int = 30):
    s: SASession = Session()
    try:
        cutoff = datetime.now() - timedelta(days=days)
        rows = (
            s.query(Sample)
            .filter(Sample.timestamp >= cutoff)
            .filter(Sample.label_setpoint.isnot(None))
            .all()
        )
        return rows
    finally:
        s.close()


def fetch_unlabeled(limit: int = 1):
    s: SASession = Session()
    try:
        rows = (
            s.query(Sample)
            .filter(Sample.label_setpoint.is_(None))
            .order_by(Sample.timestamp.desc())
            .limit(limit)
            .all()
        )
        return rows
    finally:
        s.close()


def fetch(limit: int = 1):
    s: SASession = Session()
    try:
        rows = s.query(Sample).order_by(Sample.timestamp.desc()).limit(limit).all()
        return rows
    finally:
        s.close()


def update_label(sample_id: int, label_setpoint: float, user_override: bool = False):
    s: SASession = Session()
    try:
        row = s.get(Sample, sample_id)
        if row is not None:
            row.label_setpoint = label_setpoint
            row.user_override = user_override
            if getattr(row, "predicted_setpoint", None) is not None:
                try:
                    row.prediction_error = abs(
                        float(row.predicted_setpoint) - float(label_setpoint)
                    )
                except Exception:
                    row.prediction_error = None
            s.commit()
    finally:
        s.close()


def update_sample_prediction(
    sample_id: int,
    predicted_setpoint=None,
    prediction_error=None,
):
    s: SASession = Session()
    try:
        row = s.get(Sample, sample_id)
        if row is not None:
            if predicted_setpoint is not None:
                try:
                    row.predicted_setpoint = float(predicted_setpoint)
                except Exception:
                    row.predicted_setpoint = None
            if prediction_error is not None:
                try:
                    row.prediction_error = float(prediction_error)
                except Exception:
                    row.prediction_error = None
            s.commit()
    finally:
        s.close()


def remove_unlabeled_samples(days: int = 30):
    s: SASession = Session()
    try:
        cutoff = datetime.now() - timedelta(days=days)
        s.query(Sample).filter(
            Sample.timestamp < cutoff,
            Sample.label_setpoint.is_(None),
        ).delete()
        s.commit()
    finally:
        s.close()


def remove_unlabeled_setpoints(days: int = 30):
    s: SASession = Session()
    try:
        cutoff = datetime.now() - timedelta(days=days)
        s.query(Setpoint).filter(
            Setpoint.timestamp < cutoff,
            Setpoint.observed_current_setpoint.is_(None),
            Setpoint.setpoint.is_(None),
        ).delete()
        s.commit()
    finally:
        s.close()
