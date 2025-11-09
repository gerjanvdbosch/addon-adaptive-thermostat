import os
import datetime
import logging
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Header, Query, Path
from pydantic import BaseModel, Field
import joblib
import threading
import json

from db import Session, Sample, Metric, insert_sample, update_label
from ha_client import HAClient
from trainer import Trainer

logger = logging.getLogger(__name__)
app = FastAPI(title="Adaptive Thermostat API")


def _check_token(x_addon_token: Optional[str]):
    expected_token = os.getenv("ADDON_API_TOKEN")
    if expected_token:
        if not x_addon_token or x_addon_token != expected_token:
            logger.warning("Rejected request due to invalid token")
            raise HTTPException(status_code=403, detail="Invalid addon token")


def _load_opts_from_env() -> dict:
    """
    Minimal options loader for on-demand train endpoints.
    Mirrors main.load_options keys used by Trainer/HAClient.
    """
    sensors = None
    s = os.getenv("SENSORS", None)
    if s:
        try:
            sensors = json.loads(s)
        except Exception:
            sensors = None

    return {
        "climate_entity": os.getenv("CLIMATE_ENTITY", "climate.woonkamer"),
        "shadow_mode": bool(os.getenv("SHADOW_MODE")),
        "shadow_setpoint": os.getenv("SHADOW_SETPOINT"),
        "sample_interval_seconds": int(os.getenv("SAMPLE_INTERVAL_SECONDS", 300)),
        "partial_fit_interval_seconds": int(os.getenv("PARTIAL_FIT_INTERVAL_SECONDS", 3600)),
        "full_retrain_time": os.getenv("FULL_RETRAIN_TIME", "03:00"),
        "min_setpoint": float(os.getenv("MIN_SETPOINT", 15.0)),
        "max_setpoint": float(os.getenv("MAX_SETPOINT", 24.0)),
        "min_change_threshold": float(os.getenv("MIN_CHANGE_THRESHOLD", 0.3)),
        "buffer_days": int(os.getenv("BUFFER_DAYS", 30)),
        "addon_api_token": os.getenv("ADDON_API_TOKEN", None),
        "webapi_host": os.getenv("WEBAPI_HOST", "0.0.0.0"),
        "webapi_port": int(os.getenv("WEBAPI_PORT", os.getenv("WEBAPI_PORT", 8000))),
        "model_path_partial": os.getenv("MODEL_PATH_PARTIAL"),
        "model_path_full": os.getenv("MODEL_PATH_FULL"),
        "use_unlabeled": bool(os.getenv("USE_UNLABELED")),
        "pseudo_limit": int(os.getenv("PSEUDO_LIMIT", 1000)),
        "weight_label": float(os.getenv("WEIGHT_LABEL", 1.0)),
        "weight_pseudo": float(os.getenv("WEIGHT_PSEUDO", 0.25)),
        "sensors": sensors,
    }


class LabelPayload(BaseModel):
    sample_id: Optional[int] = Field(
        None, description="Optional existing sample id to update"
    )
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp")
    entity_id: Optional[str] = Field(None, description="Originating entity id")
    new_setpoint: float = Field(..., description="The new setpoint value to record")
    user_override: bool = Field(
        True, description="Whether this was a user-initiated override"
    )
    sensors: Optional[dict] = Field(
        None, description="Optional raw sensor snapshot (English keys)"
    )


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
            update_label(
                payload.sample_id,
                float(payload.new_setpoint),
                user_override=bool(payload.user_override),
            )
            logger.info(
                "Updated label for sample_id=%s -> %s",
                payload.sample_id,
                payload.new_setpoint,
            )
            return {"status": "updated", "sample_id": payload.sample_id}

        sensors = payload.sensors or {}
        insert_sample(
            {
                "timestamp": (ts or datetime.datetime.utcnow()).isoformat(),
                "features": sensors,
            },
            label_setpoint=float(payload.new_setpoint),
            user_override=bool(payload.user_override),
        )
        logger.info(
            "Inserted new labeled sample setpoint=%s entity=%s",
            payload.new_setpoint,
            payload.entity_id,
        )
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
    x_addon_token: Optional[str] = Header(None),
):
    _check_token(x_addon_token)
    s = Session()
    try:
        rows = (
            s.query(Metric)
            .order_by(Metric.timestamp.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        out = []
        for r in rows:
            out.append(
                MetricOut(
                    id=r.id,
                    timestamp=r.timestamp,
                    model_type=r.model_type,
                    mae=r.mae,
                    n_samples=r.n_samples,
                    meta=r.meta or {},
                )
            )
        return out
    finally:
        s.close()


@app.get("/samples", response_model=List[SampleOut])
def list_samples(
    labeled: Optional[bool] = Query(
        None, description="Filter by labeled/unlabeled. None = both"
    ),
    user_override: Optional[bool] = Query(
        None, description="Filter by user_override flag"
    ),
    has_prediction: Optional[bool] = Query(
        None, description="Filter samples that have predicted_setpoint"
    ),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    x_addon_token: Optional[str] = Header(None),
):
    _check_token(x_addon_token)
    s = Session()
    try:
        q = s.query(Sample)
        if labeled is True:
            q = q.filter(Sample.label_setpoint.isnot(None))
        elif labeled is False:
            q = q.filter(Sample.label_setpoint.is_(None))
        if user_override is not None:
            q = q.filter(Sample.user_override == bool(user_override))
        if has_prediction is True:
            q = q.filter(Sample.predicted_setpoint.isnot(None))
        elif has_prediction is False:
            q = q.filter(Sample.predicted_setpoint.is_(None))

        rows = q.order_by(Sample.timestamp.desc()).limit(limit).offset(offset).all()
        out = []
        for r in rows:
            out.append(
                SampleOut(
                    id=r.id,
                    timestamp=r.timestamp,
                    data=r.data or {},
                    label_setpoint=r.label_setpoint,
                    user_override=r.user_override,
                    predicted_setpoint=r.predicted_setpoint,
                    prediction_error=r.prediction_error,
                )
            )
        return out
    finally:
        s.close()


@app.get("/samples/{sample_id}", response_model=SampleOut)
def get_sample(
    sample_id: int = Path(..., ge=1), x_addon_token: Optional[str] = Header(None)
):
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
            prediction_error=row.prediction_error,
        )
    finally:
        s.close()


@app.get("/predictions", response_model=List[PredictionOut])
def list_predictions(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    since: Optional[str] = Query(
        None, description="ISO datetime to filter predictions after"
    ),
    x_addon_token: Optional[str] = Header(None),
):
    _check_token(x_addon_token)
    s = Session()
    try:
        q = s.query(Sample).filter(Sample.predicted_setpoint.isnot(None))
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
            out.append(
                PredictionOut(
                    sample_id=r.id,
                    timestamp=r.timestamp,
                    predicted_setpoint=r.predicted_setpoint,
                    prediction_error=r.prediction_error,
                    current_setpoint=current_sp,
                    features=features,
                )
            )
        return out
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
    out = ModelMetaOut(
        file_path=path,
        present=False,
        model_type=None,
        meta=None,
        mae=None,
        n_samples=None,
        trained_at=None,
    )
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
            out.mae = (
                float(out.meta.get("mae")) if out.meta.get("mae") is not None else None
            )
        except Exception:
            out.mae = None
        try:
            out.n_samples = (
                int(out.meta.get("n_samples"))
                if out.meta.get("n_samples") is not None
                else None
            )
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


@app.get("/model/partial", response_model=Dict[str, Any])
def debug_partial_model(x_addon_token: Optional[str] = Header(None)):
    _check_token(x_addon_token)
    path = os.getenv("MODEL_PATH_PARTIAL")
    out: Dict[str, Any] = {
        "file_path": path,
        "present": False,
        "loaded": False,
        "meta": None,
        "model_type": None,
        "has_scaler": False,
        "estimator_summary": None,
        "note": None,
    }
    if not path:
        out["note"] = "MODEL_PATH_PARTIAL not configured"
        return out
    if not os.path.exists(path):
        out["note"] = "file not found"
        return out
    try:
        obj = joblib.load(path)
        out["present"] = True

        if isinstance(obj, dict):
            meta = obj.get("meta", {})
            model = obj.get("model")
            scaler = obj.get("scaler")
            out["meta"] = meta or {}
            out["has_scaler"] = scaler is not None
        else:
            model = obj
            out["meta"] = {}
            out["has_scaler"] = False

        out["model_type"] = type(model).__name__ if model is not None else None

        est_info = {}
        try:
            if hasattr(model, "named_steps"):
                est_info["is_pipeline"] = True
                steps = list(model.named_steps.keys())
                est_info["pipeline_steps"] = steps
                final = model.named_steps.get(steps[-1]) if steps else None
                final_model = final
            else:
                est_info["is_pipeline"] = False
                final_model = model

            if final_model is not None:
                if hasattr(final_model, "coef_"):
                    coef = getattr(final_model, "coef_", None)
                    try:
                        est_info["coef_shape"] = getattr(coef, "shape", None)
                        est_info["coef_sample"] = coef.tolist() if getattr(coef, "size", 0) <= 20 else f"array(len={getattr(coef,'size',0)})"
                    except Exception:
                        est_info["coef_sample"] = "unserializable"
                if hasattr(final_model, "intercept_"):
                    try:
                        est_info["intercept"] = float(getattr(final_model, "intercept_"))
                    except Exception:
                        est_info["intercept"] = str(getattr(final_model, "intercept_"))
                for attr in ("alpha", "eta0", "learning_rate", "max_iter"):
                    if hasattr(final_model, attr):
                        est_info[attr] = getattr(final_model, attr)
                if isinstance(obj, dict) and obj.get("scaler") is not None:
                    sc = obj.get("scaler")
                    try:
                        est_info["scaler_mean_shape"] = getattr(sc, "mean_").shape
                    except Exception:
                        pass
            out["estimator_summary"] = est_info
        except Exception as e:
            out["estimator_summary"] = {"error": str(e)}

        out["loaded"] = True
        out["note"] = "loaded"
        return out
    except Exception as e:
        logger.exception("Failed loading partial model %s: %s", path, e)
        out["loaded"] = False
        out["note"] = f"failed to load: {e}"
        return out


@app.get("/model/full", response_model=Dict[str, Any])
def debug_full_model(x_addon_token: Optional[str] = Header(None)):
    _check_token(x_addon_token)
    path = os.getenv("MODEL_PATH_FULL")
    out: Dict[str, Any] = {
        "file_path": path,
        "present": False,
        "loaded": False,
        "meta": None,
        "model_type": None,
        "is_pipeline": False,
        "has_scaler": False,
        "estimator_summary": None,
        "note": None,
    }
    if not path:
        out["note"] = "MODEL_PATH_FULL not configured"
        return out
    if not os.path.exists(path):
        out["note"] = "file not found"
        return out
    try:
        obj = joblib.load(path)
        out["present"] = True

        if isinstance(obj, dict):
            meta = obj.get("meta", {})
            model = obj.get("model")
            scaler = obj.get("scaler")
            out["meta"] = meta or {}
            out["has_scaler"] = scaler is not None
        else:
            model = obj
            out["meta"] = {}
            out["has_scaler"] = False

        out["model_type"] = type(model).__name__ if model is not None else None

        est_info = {}
        try:
            if hasattr(model, "named_steps"):
                est_info["is_pipeline"] = True
                steps = list(model.named_steps.keys())
                est_info["pipeline_steps"] = steps
                final = model.named_steps.get(steps[-1]) if steps else None
                final_model = final
                if "scaler" in steps:
                    out["has_scaler"] = True
            else:
                est_info["is_pipeline"] = False
                final_model = model

            if final_model is not None:
                if hasattr(final_model, "coef_"):
                    try:
                        coef = getattr(final_model, "coef_")
                        est_info["coef_shape"] = getattr(coef, "shape", None)
                        est_info["coef_sample"] = coef.tolist() if getattr(coef, "size", 0) <= 20 else f"array(len={getattr(coef,'size',0)})"
                    except Exception:
                        est_info["coef_sample"] = "unserializable"
                if hasattr(final_model, "intercept_"):
                    try:
                        intercept = getattr(final_model, "intercept_")
                        est_info["intercept"] = float(intercept) if hasattr(intercept, "__float__") else str(intercept)
                    except Exception:
                        pass
                for attr in ("alpha", "fit_intercept", "normalize"):
                    if hasattr(final_model, attr):
                        try:
                            est_info[attr] = getattr(final_model, attr)
                        except Exception:
                            pass
                if isinstance(obj, dict) and obj.get("scaler") is not None:
                    sc = obj.get("scaler")
                    try:
                        est_info["scaler_mean_shape"] = getattr(sc, "mean_").shape
                    except Exception:
                        pass

            out["estimator_summary"] = est_info
        except Exception as e:
            out["estimator_summary"] = {"error": str(e)}

        out["loaded"] = True
        out["note"] = "loaded"
        return out
    except Exception as e:
        logger.exception("Failed loading full model %s: %s", path, e)
        out["loaded"] = False
        out["note"] = f"failed to load: {e}"
        return out


@app.post("/train/full")
def trigger_full_train(x_addon_token: Optional[str] = Header(None)):
    """
    Trigger a full retrain in background. Returns immediately with a job status.
    """
    _check_token(x_addon_token)
    opts = _load_opts_from_env()
    ha = HAClient(opts)
    trainer = Trainer(ha, opts)

    def _run():
        try:
            logger.info("Triggered full retrain via API")
            trainer.full_retrain_job()
            logger.info("Full retrain job finished (API-triggered)")
        except Exception:
            logger.exception("Exception in API-triggered full retrain")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "job": "full_retrain"}


@app.post("/train/partial")
def trigger_partial_train(x_addon_token: Optional[str] = Header(None)):
    """
    Trigger a partial_fit job in background. Returns immediately.
    """
    _check_token(x_addon_token)
    opts = _load_opts_from_env()
    ha = HAClient(opts)
    trainer = Trainer(ha, opts)

    def _run():
        try:
            logger.info("Triggered partial_fit via API")
            trainer.partial_fit_job()
            logger.info("Partial fit job finished (API-triggered)")
        except Exception:
            logger.exception("Exception in API-triggered partial fit")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "job": "partial_fit"}
