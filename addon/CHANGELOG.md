# Changelog

## v0.1 – Initial Version

### Features

- **Incremental learning:** `SGDRegressor` for fast adaptation to new user overrides.
- **Periodic full retrains:** `Ridge` with grid search and `TimeSeriesSplit` for robust, stable models.
- **Pseudo-labeling support:** use current setpoint as weak label with configurable weight and caps.
- **Safety checks:** plausibility range, minimum change threshold, stability/hysteresis window, cooldown.
- **SQLite-backed dataset:** explicit separation between features, user labels, and prediction metadata.
- **FastAPI endpoints:** manual labeling, model inspection, metrics, and manual train triggers.
- **Configurable scheduling:** sampling, partial updates, inference interval, and scheduled full retrain.
- **Explainable feature engineering:** time cyclical encodings, weather/thermostat signals, operational status.
- **Logging and monitoring hooks:** MAE, coefficient drift, and sample-weight summaries.
