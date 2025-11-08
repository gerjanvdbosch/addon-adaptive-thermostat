import os
import datetime
import logging

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from db import update_label, insert_sample
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)
app = FastAPI(title="Adaptive Thermostat API")

fe = FeatureExtractor()


class LabelPayload(BaseModel):
    sample_id: int | None = Field(None, description="Optional existing sample id to update")
    timestamp: str | None = Field(None, description="Optional ISO timestamp")
    entity_id: str | None = Field(None, description="Originating entity id")
    new_setpoint: float = Field(..., description="The new setpoint value to record")
    user_override: bool = Field(True, description="Whether this was a user-initiated override")
    sensors: dict | None = Field(None, description="Optional raw sensor snapshot (English keys)")


@app.post("/label")
def receive_label(
    payload: LabelPayload,
    x_addon_token: str | None = Header(None)
):
    expected_token = os.getenv("ADDON_API_TOKEN")
    if expected_token:
        if not x_addon_token or x_addon_token != expected_token:
            logger.warning("Rejected /label request due to invalid token")
            raise HTTPException(status_code=403, detail="Invalid addon token")

    try:
        ts = None
        if payload.timestamp:
            try:
                ts = datetime.datetime.fromisoformat(payload.timestamp)
            except Exception:
                ts = datetime.datetime.utcnow()

        if payload.sample_id is not None:
            update_label(payload.sample_id, float(payload.new_setpoint), user_override=bool(payload.user_override))
            logger.info("Updated label for sample_id=%s -> %s", payload.sample_id, payload.new_setpoint)
            return {"status": "updated", "sample_id": payload.sample_id}

        sensors = payload.sensors or {}
        features = fe.features_from_raw(sensors, timestamp=ts)
        insert_sample(
            {"timestamp": (ts or datetime.datetime.utcnow()).isoformat(), "features": features},
            label_setpoint=float(payload.new_setpoint),
            user_override=bool(payload.user_override),
        )
        logger.info("Inserted new labeled sample setpoint=%s entity=%s", payload.new_setpoint, payload.entity_id)
        return {"status": "inserted", "new_setpoint": payload.new_setpoint}

    except Exception as e:
        logger.exception("Error handling /label request: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
