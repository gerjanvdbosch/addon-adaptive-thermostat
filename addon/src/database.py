import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Float, DateTime, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from config import Config

Base = declarative_base()


class SolarForecast(Base):
    """Het tabel-model voor alle zonne- en weerdata."""

    __tablename__ = "solar_forecast"

    timestamp = Column(DateTime, primary_key=True, index=True)

    # Voorspelling (Solcast)
    pv_estimate = Column(Float)
    pv_estimate10 = Column(Float)
    pv_estimate90 = Column(Float)

    # Werkelijke opbrengst (Home Assistant)
    pv_actual = Column(Float, nullable=True)

    # Weerdata (OpenMeteo)
    temp = Column(Float)
    cloud = Column(Float)
    wind = Column(Float)
    precipitation = Column(Float)
    radiation = Column(Float)
    diffuse = Column(Float)
    tilted = Column(Float)


class Database:
    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__)
        self.database_path = config.database_path

        self.engine = create_engine(
            f"sqlite:///{self.database_path}", connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        Base.metadata.create_all(bind=self.engine)

    def save_forecast(self, df: pd.DataFrame):
        """Slaat een DataFrame op. Bestaande timestamps worden bijgewerkt (upsert)."""
        if df.empty:
            return

        session: Session = self.SessionLocal()
        try:
            # We zetten de DataFrame om naar een lijst van dictionaries
            valid_columns = {c.key for c in SolarForecast.__table__.columns}
            records = df.to_dict(orient="records")
            for record in records:
                # 'merge' kijkt naar de primary key (timestamp).
                # Bestaat hij al? Dan update. Bestaat hij niet? Dan insert.
                filtered_record = {
                    k: v for k, v in record.items() if k in valid_columns
                }
                obj = SolarForecast(**filtered_record)
                session.merge(obj)

            session.commit()
            self.logger.debug(f"[DB] {len(records)} records opgeslagen/geüpdate")
        except Exception as e:
            session.rollback()
            self.logger.error(f"[DB] Fout bij opslaan forecast: {e}")
        finally:
            session.close()

    def update_pv_actual(self, ts: datetime, yield_kw: float):
        """Update alleen de werkelijke opbrengst voor een specifiek tijdstip."""
        session: Session = self.SessionLocal()
        try:
            # Zoek het record voor dit kwartier
            record = (
                session.query(SolarForecast)
                .where(SolarForecast.timestamp == ts)
                .first()
            )
            if record:
                record.pv_actual = yield_kw
                session.commit()
            else:
                # Als het record nog niet bestaat (bijv. geen forecast), maken we een leeg record aan
                new_record = SolarForecast(timestamp=ts, pv_actual=yield_kw)
                session.add(new_record)
                session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"[DB] Fout bij updaten actual yield: {e}")
        finally:
            session.close()

    def get_forecast_history(self, cutoff_date: datetime):
        """Haalt historische data op als Pandas DataFrame voor training."""
        try:
            stmt = (
                select(SolarForecast)
                .where(SolarForecast.timestamp >= cutoff_date)
                .where(SolarForecast.pv_actual.isnot(None))
                .order_by(SolarForecast.timestamp.asc())
            )
            with self.engine.connect() as conn:
                df = pd.read_sql(stmt, conn)

            # Zorg dat de timestamp kolom ook echt als datetime wordt herkend door Pandas
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            return df
        except Exception as e:
            self.logger.error(f"[DB] Fout bij ophalen historie: {e}")
            return pd.DataFrame()
