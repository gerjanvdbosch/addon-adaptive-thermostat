import os
from typing import List, Optional
from datetime import datetime, timedelta

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
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


Base.metadata.create_all(engine)


def insert_sample(
    data: dict, label_setpoint: Optional[float] = None, user_override: bool = False
) -> int:
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


def fetch_training_data(days: int = 30) -> List[Sample]:
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


def fetch_unlabeled(limit: int = 1) -> List[Sample]:
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


def fetch(limit: int = 1) -> List[Sample]:
    s: SASession = Session()
    try:
        rows = s.query(Sample).order_by(Sample.timestamp.desc()).limit(limit).all()
        return rows
    finally:
        s.close()


def update_label(
    sample_id: int, label_setpoint: float, user_override: bool = False
) -> None:
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
    predicted_setpoint: Optional[float] = None,
    prediction_error: Optional[float] = None,
) -> None:
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
