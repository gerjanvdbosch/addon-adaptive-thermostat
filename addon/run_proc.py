from api import APIService
import yaml, os

here = os.path.dirname(__file__)
with open(os.path.join(here, "config_default.yaml"), "r") as f:
    cfg_default = yaml.safe_load(f)

model_dir = os.environ.get("MODEL_DIR", cfg_default.get("model_dir", "/data"))
service = APIService(model_dir, cfg_default)

# Flask built-in server, threaded to allow simple concurrency
service.app.run(host="0.0.0.0", port=5000, threaded=True)
