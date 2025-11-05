import os
import threading
import time
import logging
import json
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from collector import Collector
from trainer import Trainer
from inferencer import Inferencer
from ha_client import HAClient

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_options():
    # Support passing sensors mapping as JSON string in SENSORS env var or via mapped config (opts injected by Supervisor)
    try:
        sensors = json.loads(os.getenv("SENSORS", None))
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
        "sensors": sensors
    }


def start_api(host: str, port: int):
    uvicorn.run("webapi:app", host=host, port=port, log_level="info")


def main():
    opts = load_options()
    ha = HAClient()
    collector = Collector(ha, opts)
    trainer = Trainer(ha, opts)
    inferencer = Inferencer(ha, opts)

    api_thread = threading.Thread(target=start_api, args=(opts["webapi_host"], opts["webapi_port"]), daemon=True)
    api_thread.start()
    logger.info("Started internal web API on %s:%s", opts["webapi_host"], opts["webapi_port"])

    scheduler = BackgroundScheduler()
    scheduler.add_job(collector.sample_and_store, 'interval', seconds=opts["sample_interval_seconds"], id='collector')
    scheduler.add_job(trainer.partial_fit_job, 'interval', seconds=opts["partial_fit_interval_seconds"], id='partial_fit')

    hh, mm = map(int, opts["full_retrain_time"].split(":"))
    
    scheduler.add_job(trainer.full_retrain_job, 'cron', hour=hh, minute=mm, id='full_retrain')
    scheduler.add_job(inferencer.inference_job, 'interval', seconds=60, id='inference')

    scheduler.start()
    logger.info("Adaptive Thermostat add-on started")
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping scheduler and exiting")
        scheduler.shutdown()


if __name__ == "__main__":
    main()

