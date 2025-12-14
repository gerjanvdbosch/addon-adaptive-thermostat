import os
from datetime import datetime
import logging
import joblib
import threading

from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Header, Query, Path
from pydantic import BaseModel, Field
from config import load_options
from db import Session, Sample, Setpoint, insert_sample, update_label, update_setpoint
from ha_client import HAClient
from trainer import Trainer
from trainer_delta import TrainerDelta

logger = logging.getLogger(__name__)
app = FastAPI(title="Adaptive Thermostat API")


def _check_token(x_addon_token: Optional[str]):
    expected_token = os.getenv("ADDON_API_TOKEN")
    if expected_token:
        if not x_addon_token or x_addon_token != expected_token:
            logger.warning("Rejected request due to invalid token")
            raise HTTPException(status_code=403, detail="Invalid addon token")


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


class SetpointOut(BaseModel):
    id: int
    timestamp: datetime
    data: Optional[dict]
    setpoint: Optional[float]
    observed_current_setpoint: Optional[float]


class SampleOut(BaseModel):
    id: int
    timestamp: datetime
    data: Optional[dict]
    label_setpoint: Optional[float]
    user_override: Optional[bool]
    predicted_setpoint: Optional[float]
    prediction_error: Optional[float]


class PredictionOut(BaseModel):
    sample_id: int
    timestamp: datetime
    predicted_setpoint: Optional[float]
    prediction_error: Optional[float]
    current_setpoint: Optional[float]
    features: Optional[dict]


@app.post("/label")
def receive_label(payload: LabelPayload, x_addon_token: Optional[str] = Header(None)):
    _check_token(x_addon_token)
    try:
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
            sensors,
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


@app.get("/setpoints", response_model=List[SetpointOut])
def list_setpoints(
    labeled: Optional[bool] = Query(
        None, description="Filter by labeled/unlabeled. None = both"
    ),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    x_addon_token: Optional[str] = Header(None),
):
    _check_token(x_addon_token)
    s = Session()
    try:
        q = s.query(Setpoint)
        if labeled is True:
            q = q.filter(Setpoint.setpoint.isnot(None))
        rows = q.order_by(Setpoint.timestamp.desc()).limit(limit).offset(offset).all()
        out = []
        for r in rows:
            out.append(
                SetpointOut(
                    id=r.id,
                    timestamp=r.timestamp,
                    data=getattr(r, "data", None) or {},
                    setpoint=getattr(r, "setpoint", None),
                    observed_current_setpoint=getattr(
                        r, "observed_current_setpoint", None
                    ),
                )
            )
        return out
    finally:
        s.close()


class SetpointPatch(BaseModel):
    setpoint: Optional[float] = Field(
        None, description="New setpoint value or null to clear"
    )
    observed_current_setpoint: Optional[float] = Field(
        None, description="Observed baseline to store"
    )


@app.post("/setpoints/{setpoint_id}", status_code=200)
def patch_setpoint_minimal(
    setpoint_id: int = Path(..., ge=1),
    payload: SetpointPatch = None,
    x_addon_token: Optional[str] = Header(None),
):
    _check_token(x_addon_token)

    if payload is None:
        raise HTTPException(status_code=400, detail="Empty payload")

    # Coerce / validate fields
    sp_val = None
    if payload.setpoint is not None:
        try:
            sp_val = float(payload.setpoint)
        except Exception:
            raise HTTPException(
                status_code=400, detail="setpoint must be numeric or null"
            )

    obs_val = None
    if payload.observed_current_setpoint is not None:
        try:
            obs_val = float(payload.observed_current_setpoint)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="observed_current_setpoint must be numeric or null",
            )

    # Apply update
    try:
        update_setpoint(setpoint_id, setpoint=sp_val, observed_current=obs_val)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed updating setpoint")

    # Return updated row
    s = Session()
    try:
        row = s.get(Setpoint, setpoint_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Setpoint not found")
        return {
            "id": row.id,
            "timestamp": row.timestamp if isinstance(row.timestamp, datetime) else None,
            "setpoint": getattr(row, "setpoint", None),
            "observed_current_setpoint": getattr(
                row, "observed_current_setpoint", None
            ),
        }
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


@app.delete("/samples/{sample_id}")
def delete_sample(
    sample_id: int = Path(..., ge=1), x_addon_token: Optional[str] = Header(None)
):
    _check_token(x_addon_token)
    s = Session()
    try:
        sample = s.get(Sample, sample_id)
        if not sample:
            raise HTTPException(status_code=404, detail="Sample not found")
        s.delete(sample)
        s.commit()
        return {"status": "deleted", "sample_id": sample_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting sample %s: %s", sample_id, e)
        s.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        s.close()


@app.delete("/setpoints/{setpoint_id}")
def delete_setpoint(
    setpoint_id: int = Path(..., ge=1), x_addon_token: Optional[str] = Header(None)
):
    _check_token(x_addon_token)
    s = Session()
    try:
        sample = s.get(Setpoint, setpoint_id)
        if not sample:
            raise HTTPException(status_code=404, detail="Setpoint not found")
        s.delete(sample)
        s.commit()
        return {"status": "deleted", "setpoint_id": setpoint_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting setpoint %s: %s", setpoint_id, e)
        s.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
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
                dt = datetime.fromisoformat(since)
                q = q.filter(Sample.timestamp >= dt)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'since' timestamp")
        rows = q.order_by(Sample.timestamp.desc()).limit(limit).offset(offset).all()
        out = []
        for r in rows:
            features = None
            current_sp = None
            if r.data and isinstance(r.data, dict):
                features = r.data
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
    trained_at: Optional[datetime] = None
    note: Optional[str] = None


class ModelSummaryOut(BaseModel):
    full: ModelMetaOut
    partial: ModelMetaOut
    best_source: Optional[str] = None
    retrieved_at: datetime


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
                out.trained_at = datetime.fromisoformat(ta)
            except Exception:
                out.trained_at = None
        out.note = "loaded"
    except Exception as e:
        logger.exception("Failed loading model file %s: %s", path, e)
        out.present = True
        out.note = f"failed to load: {e}"
    return out


def _to_builtin_type(val: Any):
    """Recursively convert numpy types, arrays and other non-serializables to native types."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if _np is not None:
        if isinstance(val, _np.generic):
            return val.item()
        if isinstance(val, _np.ndarray):
            return val.tolist()
    if isinstance(val, dict):
        return {str(k): _to_builtin_type(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_to_builtin_type(v) for v in list(val)]
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return repr(val)
    try:
        return str(val)
    except Exception:
        return repr(val)


def _extract_model_info(payload: dict) -> Dict[str, Any]:
    """Extract helpful model introspection fields (feature_importances, params, iterations)."""
    model = payload.get("model") if isinstance(payload, dict) else payload
    info: Dict[str, Any] = {}
    try:
        # pipeline with named step 'model'
        if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps"):
            core = model.named_steps["model"]
            if hasattr(core, "feature_importances_"):
                info["feature_importances"] = _to_builtin_type(
                    core.feature_importances_
                )
            if hasattr(core, "n_estimators"):
                info["n_estimators"] = int(core.n_estimators)
            if hasattr(core, "max_iter"):
                info["max_iter"] = int(core.max_iter)
            if hasattr(core, "n_iter_"):
                info["n_iter_"] = _to_builtin_type(getattr(core, "n_iter_"))
            if hasattr(core, "best_iteration_"):
                info["best_iteration_"] = _to_builtin_type(
                    getattr(core, "best_iteration_")
                )
            try:
                info["model_params"] = _to_builtin_type(core.get_params())
            except Exception:
                info["model_params"] = str(core)
        else:
            # raw estimator fallback
            if hasattr(model, "feature_importances_"):
                info["feature_importances"] = _to_builtin_type(
                    model.feature_importances_
                )
            try:
                info["model_params"] = _to_builtin_type(model.get_params())
            except Exception:
                info["model_repr"] = str(model)
    except Exception:
        logger.exception("Model introspection failed")
    return info


@app.get("/model/full", response_model=Dict[str, Any])
def get_full_model_fullmeta():
    """
    Return everything in payload['meta'] (if present) plus derived model info and file diagnostics.
    This endpoint converts numpy / non-serializable types to built-in Python types.
    """
    MODEL_PATH = "/config/models/full_model2.joblib"
    if not os.path.exists(MODEL_PATH):
        logger.info("Model file not found at %s", MODEL_PATH)
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        payload = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model file %s", MODEL_PATH)
        raise HTTPException(status_code=500, detail=f"Failed loading model: {e}")

    # meta payload (return every key present)
    if isinstance(payload, dict) and "meta" in payload:
        raw_meta = payload.get("meta") or {}
        meta = _to_builtin_type(raw_meta)
    else:
        meta = {"note": "No meta key found in model payload"}

    # derived info about the model object
    model_info = _extract_model_info(payload)

    # file diagnostics
    try:
        stat = os.stat(MODEL_PATH)
        file_info = {
            "path": MODEL_PATH,
            "size_bytes": stat.st_size,
            "modified_ts": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
        }
    except Exception:
        file_info = {"path": MODEL_PATH}

    response = {
        "meta": meta,
        "model_info": _to_builtin_type(model_info),
        "file": file_info,
    }

    # include entire payload keys (safe-serialized) if you want — uncomment to include everything:
    # response["raw_payload"] = _to_builtin_type(payload)

    return _to_builtin_type(response)


@app.get("/model/delta", response_model=Dict[str, Any])
def get_full_model_full_delta_meta():
    """
    Return everything in payload['meta'] (if present) plus derived model info and file diagnostics.
    This endpoint converts numpy / non-serializable types to built-in Python types.
    """
    MODEL_PATH = "/config/models/delta_model.joblib"
    if not os.path.exists(MODEL_PATH):
        logger.info("Model file not found at %s", MODEL_PATH)
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        payload = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model file %s", MODEL_PATH)
        raise HTTPException(status_code=500, detail=f"Failed loading model: {e}")

    # meta payload (return every key present)
    if isinstance(payload, dict) and "meta" in payload:
        raw_meta = payload.get("meta") or {}
        meta = _to_builtin_type(raw_meta)
    else:
        meta = {"note": "No meta key found in model payload"}

    # derived info about the model object
    model_info = _extract_model_info(payload)

    # file diagnostics
    try:
        stat = os.stat(MODEL_PATH)
        file_info = {
            "path": MODEL_PATH,
            "size_bytes": stat.st_size,
            "modified_ts": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
        }
    except Exception:
        file_info = {"path": MODEL_PATH}

    response = {
        "meta": meta,
        "model_info": _to_builtin_type(model_info),
        "file": file_info,
    }

    # include entire payload keys (safe-serialized) if you want — uncomment to include everything:
    # response["raw_payload"] = _to_builtin_type(payload)

    return _to_builtin_type(response)


@app.post("/train/full")
def trigger_full_train(
    force: bool = Query(
        False, description="If true, force overwrite even if MAE is not improved"
    ),
    x_addon_token: Optional[str] = Header(None),
):
    """
    Trigger a full retrain in background. Returns immediately with a job status.
    """
    _check_token(x_addon_token)
    opts = load_options()
    ha = HAClient(opts)
    # trainer = Trainer(ha, opts)
    trainer = Trainer(ha, opts)
    trainer_delta = TrainerDelta(ha, opts)

    def _run():
        try:
            logger.info("Triggered full retrain via API (force=%s)", force)
            # trainer.full_retrain_job(force=force)
            trainer.train_job(force=force)
            trainer_delta.train_job(force=force)
            logger.info("Full retrain job finished (API-triggered)")
        except Exception:
            logger.exception("Exception in API-triggered full retrain")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "job": "full_retrain", "force": bool(force)}
