import os
import datetime
import logging
import joblib
import threading

from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Header, Query, Path
from pydantic import BaseModel, Field
from config import load_options
from db import Session, Sample, insert_sample, update_label
from ha_client import HAClient
from trainer import Trainer
from trainer2 import Trainer2

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
                        est_info["coef_sample"] = (
                            coef.tolist()
                            if getattr(coef, "size", 0) <= 20
                            else f"array(len={getattr(coef,'size',0)})"
                        )
                    except Exception:
                        est_info["coef_sample"] = "unserializable"
                if hasattr(final_model, "intercept_"):
                    try:
                        est_info["intercept"] = float(
                            getattr(final_model, "intercept_")
                        )
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
                        est_info["coef_sample"] = (
                            coef.tolist()
                            if getattr(coef, "size", 0) <= 20
                            else f"array(len={getattr(coef,'size',0)})"
                        )
                    except Exception:
                        est_info["coef_sample"] = "unserializable"
                if hasattr(final_model, "intercept_"):
                    try:
                        intercept = getattr(final_model, "intercept_")
                        est_info["intercept"] = (
                            float(intercept)
                            if hasattr(intercept, "__float__")
                            else str(intercept)
                        )
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


@app.get("/model/full2", response_model=Dict[str, Any])
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
    trainer = Trainer(ha, opts)
    trainer2 = Trainer2(ha, opts)

    def _run():
        try:
            logger.info("Triggered full retrain via API (force=%s)", force)
            trainer.full_retrain_job(force=force)
            trainer2.full_retrain_job(force=force)
            logger.info("Full retrain job finished (API-triggered)")
        except Exception:
            logger.exception("Exception in API-triggered full retrain")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "job": "full_retrain", "force": bool(force)}


@app.post("/train/partial")
def trigger_partial_train(x_addon_token: Optional[str] = Header(None)):
    """
    Trigger a partial_fit job in background. Returns immediately.
    """
    _check_token(x_addon_token)
    opts = load_options()
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
