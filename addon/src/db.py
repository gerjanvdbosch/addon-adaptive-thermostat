import os
import datetime
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
DB_PATH = os.getenv("DB_PATH", "/db/samples.sqlite")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


class Sample(Base):
    __tablename__ = "samples"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    data = Column(JSON)
    label_setpoint = Column(Float, nullable=True)
    user_override = Column(Boolean, default=False)


Base.metadata.create_all(engine)


def insert_sample(data: dict, label_setpoint=None, user_override=False):
    s = Session()
    sample = Sample(data=data, label_setpoint=label_setpoint, user_override=user_override)
    s.add(sample)
    s.commit()
    sample_id = sample.id
    s.close()
    return sample_id


def fetch_training_data(days=30):
    s = Session()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    rows = s.query(Sample).filter(Sample.timestamp >= cutoff).filter(Sample.label_setpoint != None).all()
    s.close()
    return rows


def fetch_unlabeled(limit=1):
    s = Session()
    rows = s.query(Sample).filter(Sample.label_setpoint == None).order_by(Sample.timestamp.desc()).limit(limit).all()
    s.close()
    return rows


def update_label(sample_id, label_setpoint, user_override=False):
    s = Session()
    row = s.get(Sample, sample_id)
    if row:
        row.label_setpoint = label_setpoint
        row.user_override = user_override
        s.commit()
    s.close()
