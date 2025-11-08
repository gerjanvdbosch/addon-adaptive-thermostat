import os
import datetime
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Query, Path
from pydantic import BaseModel, Field
import joblib

from db import Session, Sample, Metric, insert_sample, update_label

logger = logging.getLogger(__name__)
app = FastAPI(title="Adaptive Thermostat API")


def _check_token(x_addon_token: Optional[str]):
    expected_token = os.getenv("ADDON_API_TOKEN")
    if expected_token:
        if not x_addon_token or x_addon_token != expected_token:
            logger.warning("Rejected request due to invalid token")
            raise HTTPException(status_code=403, detail="Invalid addon token")


class LabelPayload(BaseModel):
    sample_id: Optional[int] = Field(None, description="Optional existing sample id to update")
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp")
    entity_id: Optional[str] = Field(None, description="Originating entity id")
    new_setpoint: float = Field(..., description="The new setpoint value to record")
    user_override: bool = Field(True, description="Whether this was a user-initiated override")
    sensors: Optional[dict] = Field(None, description="Optional raw sensor snapshot (English keys)")


class MetricOut(BaseModel):
    id: int
    timestamp: datetime.datetime
    model_type: Optional[str]
    mae: Optional[float]
    n_samples: Optional[int]
    meta: Optional[dict]


class SampleOut(BaseModel):
    id: int
    timestamp: datetime.datetime
    data: Optional[dict]
    label_setpoint: Optional[float]
    user_override: Optional[bool]
    predicted_setpoint: Optional[float]
    prediction_error: Optional[float]


class PredictionOut(BaseModel):
    sample_id: int
    timestamp: datetime.datetime
    predicted_setpoint: Optional[float]
    prediction_error: Optional[float]
    current_setpoint: Optional[float]
    features: Optional[dict]


@app.post("/label")
def receive_label(payload: LabelPayload, x_addon_token: Optional[str] = Header(None)):
    _check_token(x_addon_token)
    try:
        ts = None
        if payload.timestamp:
            try:
                ts = datetime.datetime.datetime.fromisoformat(payload.timestamp)
            except Exception:
                ts = datetime.datetime.utcnow()

        if payload.sample_id is not None:
            update_label(payload.sample_id, float(payload.new_setpoint), user_override=bool(payload.user_override))
            logger.info("Updated label for sample_id=%s -> %s", payload.sample_id, payload.new_setpoint)
            return {"status": "updated", "sample_id": payload.sample_id}

        sensors = payload.sensors or {}
        insert_sample(
            {"timestamp": (ts or datetime.datetime.utcnow()).isoformat(), "features": sensors},
            label_setpoint=float(payload.new_setpoint),
            user_override=bool(payload.user_override),
        )
        logger.info("Inserted new labeled sample setpoint=%s entity=%s", payload.new_setpoint, payload.entity_id)
        return {"status": "inserted", "new_setpoint": payload.new_setpoint}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error handling /label request: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics", response_model=List[MetricOut])
def list_metrics(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    x_addon_token: Optional[str] = Header(None)
):
    _check_token(x_addon_token)
    s = Session()
    try:
        rows = s.query(Metric).order_by(Metric.timestamp.desc()).limit(limit).offset(offset).all()
        out = []
        for r in rows:
            out.append(MetricOut(
                id=r.id,
                timestamp=r.timestamp,
                model_type=r.model_type,
                mae=r.mae,
                n_samples=r.n_samples,
                meta=r.meta or {}
            ))
        return out
    finally:
        s.close()


@app.get("/samples", response_model=List[SampleOut])
def list_samples(
    labeled: Optional[bool] = Query(None, description="Filter by labeled/unlabeled. None = both"),
    user_override: Optional[bool] = Query(None, description="Filter by user_override flag"),
    has_prediction: Optional[bool] = Query(None, description="Filter samples that have predicted_setpoint"),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    x_addon_token: Optional[str] = Header(None)
):
    _check_token(x_addon_token)
    s = Session()
    try:
        q = s.query(Sample)
        if labeled is True:
            q = q.filter(Sample.label_setpoint != None)
        elif labeled is False:
            q = q.filter(Sample.label_setpoint == None)
        if user_override is not None:
            q = q.filter(Sample.user_override == bool(user_override))
        if has_prediction is True:
            q = q.filter(Sample.predicted_setpoint != None)
        elif has_prediction is False:
            q = q.filter(Sample.predicted_setpoint == None)

        rows = q.order_by(Sample.timestamp.desc()).limit(limit).offset(offset).all()
        out = []
        for r in rows:
            out.append(SampleOut(
                id=r.id,
                timestamp=r.timestamp,
                data=r.data or {},
                label_setpoint=r.label_setpoint,
                user_override=r.user_override,
                predicted_setpoint=r.predicted_setpoint,
                prediction_error=r.prediction_error
            ))
        return out
    finally:
        s.close()


@app.get("/samples/{sample_id}", response_model=SampleOut)
def get_sample(sample_id: int = Path(..., ge=1), x_addon_token: Optional[str] = Header(None)):
    _check_token(x_addon_token)
    s = Session()
    try:
        row = s.get(Sample, sample_id)
        if not row:
            raise HTTPException(status_code=404, detail="Sample not found")
        return SampleOut(
            id=row.id,
            timestamp=row.timestamp,
            data=row.data or {},
            label_setpoint=row.label_setpoint,
            user_override=row.user_override,
            predicted_setpoint=row.predicted_setpoint,
            prediction_error=row.prediction_error
        )
    finally:
        s.close()


@app.get("/predictions", response_model=List[PredictionOut])
def list_predictions(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    since: Optional[str] = Query(None, description="ISO datetime to filter predictions after"),
    x_addon_token: Optional[str] = Header(None)
):
    _check_token(x_addon_token)
    s = Session()
    try:
        q = s.query(Sample).filter(Sample.predicted_setpoint != None)
        if since:
            try:
                dt = datetime.datetime.fromisoformat(since)
                q = q.filter(Sample.timestamp >= dt)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'since' timestamp")
        rows = q.order_by(Sample.timestamp.desc()).limit(limit).offset(offset).all()
        out = []
        for r in rows:
            features = None
            current_sp = None
            if r.data and isinstance(r.data, dict):
                features = r.data.get("features") or r.data
                if isinstance(features, dict):
                    current_sp = features.get("current_setpoint")
            out.append(PredictionOut(
                sample_id=r.id,
                timestamp=r.timestamp,
                predicted_setpoint=r.predicted_setpoint,
                prediction_error=r.prediction_error,
                current_setpoint=current_sp,
                features=features
            ))
        return out
    finally:
        s.close()


@app.get("/predictions/latest", response_model=Optional[PredictionOut])
def latest_prediction(x_addon_token: Optional[str] = Header(None)):
    _check_token(x_addon_token)
    s = Session()
    try:
        row = s.query(Sample).filter(Sample.predicted_setpoint != None).order_by(Sample.timestamp.desc()).first()
        if not row:
            return None
        features = None
        current_sp = None
        if row.data and isinstance(row.data, dict):
            features = row.data.get("features") or row.data
            if isinstance(features, dict):
                current_sp = features.get("current_setpoint")
        return PredictionOut(
            sample_id=row.id,
            timestamp=row.timestamp,
            predicted_setpoint=row.predicted_setpoint,
            prediction_error=row.prediction_error,
            current_setpoint=current_sp,
            features=features
        )
    finally:
        s.close()


# Model summary endpoint (reads model paths from environment variables)
class ModelMetaOut(BaseModel):
    model_type: Optional[str] = None
    file_path: Optional[str] = None
    present: bool = False
    meta: Optional[dict] = None
    mae: Optional[float] = None
    n_samples: Optional[int] = None
    trained_at: Optional[datetime.datetime] = None
    note: Optional[str] = None


class ModelSummaryOut(BaseModel):
    full: ModelMetaOut
    partial: ModelMetaOut
    best_source: Optional[str] = None
    retrieved_at: datetime.datetime


def _load_model_meta_from_path(path: Optional[str]) -> ModelMetaOut:
    out = ModelMetaOut(file_path=path, present=False, model_type=None, meta=None, mae=None, n_samples=None, trained_at=None)
    if not path:
        out.note = "no path configured"
        return out
    if not os.path.exists(path):
        out.present = False
        out.note = "file not found"
        return out
    try:
        obj = joblib.load(path)
        meta = obj.get("meta") if isinstance(obj, dict) else None
        out.present = True
        out.meta = meta or {}
        out.model_type = "pipeline_or_estimator"
        try:
            out.mae = float(out.meta.get("mae")) if out.meta.get("mae") is not None else None
        except Exception:
            out.mae = None
        try:
            out.n_samples = int(out.meta.get("n_samples")) if out.meta.get("n_samples") is not None else None
        except Exception:
            out.n_samples = None
        ta = out.meta.get("trained_at") if out.meta else None
        if ta:
            try:
                out.trained_at = datetime.datetime.fromisoformat(ta)
            except Exception:
                out.trained_at = None
        out.note = "loaded"
    except Exception as e:
        logger.exception("Failed loading model file %s: %s", path, e)
        out.present = True
        out.note = f"failed to load: {e}"
    return out


@app.get("/model_summary", response_model=ModelSummaryOut)
def model_summary(x_addon_token: Optional[str] = Header(None)):
    """
    Returns a summary of available models and the most recent full-model metric (OOF MAE).
    """
    _check_token(x_addon_token)

    full_path = os.getenv("MODEL_PATH_FULL")
    partial_path = os.getenv("MODEL_PATH_PARTIAL")

    full_meta = _load_model_meta_from_path(full_path)
    partial_meta = _load_model_meta_from_path(partial_path)

    # prefer DB metric for authoritative OOF MAE if available (full model)
    s = Session()
    try:
        latest_metric = s.query(Metric).filter(Metric.model_type == "full").order_by(Metric.timestamp.desc()).first()
        if latest_metric:
            try:
                full_meta.mae = float(latest_metric.mae) if latest_metric.mae is not None else full_meta.mae
                full_meta.n_samples = int(latest_metric.n_samples) if latest_metric.n_samples is not None else full_meta.n_samples
                full_meta.trained_at = latest_metric.timestamp
                full_meta.note = (full_meta.note or "") + " ; metric_record_used"
            except Exception:
                pass
    finally:
        s.close()

    best = None
    if full_meta.present and (full_meta.meta or full_meta.mae is not None):
        best = "full"
    elif partial_meta.present and (partial_meta.meta or partial_meta.mae is not None):
        best = "partial"

    return ModelSummaryOut(
        full=full_meta,
        partial=partial_meta,
        best_source=best,
        retrieved_at=datetime.datetime.utcnow()
    )
