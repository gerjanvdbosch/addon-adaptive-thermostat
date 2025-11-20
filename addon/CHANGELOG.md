# Changelog

## Version 0.2

### Features

*   **Incremental learning:** fast partial updates and conservative retrains using scikit-learn estimators.
*   **Periodic full retrains:** HistGradientBoostingRegressor with RandomizedSearchCV and TimeSeriesSplit for robust, stable models.
*   **Delta-model support:** TrainerDelta / InferencerDelta learn and apply deltas (predicted\_delta + explicit baseline) to reduce echo/shadow learning.
*   **Masking of current\_setpoint:** current\_setpoint is masked during training and inference to prevent trivial identity predictions.
*   **Feature set (FEATURE\_ORDER):** cyclical time encodings (hour\_sin/hour\_cos, day\_sin/day\_cos, month\_sin/month\_cos), season, day\_or\_night, current\_setpoint (masked), current\_temp, temp\_change, min/max temps (today/tomorrow), solar\_kwh/solar\_chance (today/tomorrow), wind\_speed/dir, outside\_temp, hvac\_mode.
*   **Inference engines:** Inferencer (full-model) and InferencerDelta (delta-model) with baseline reconstruction, masked vectors and per-sample persistence.
*   **Explainability & metrics:** saved model.meta includes feature\_order, trained\_at, mae/val\_mae, n\_samples, chosen\_params, runtime\_seconds, top\_features (native/SHAP/permutation) and file diagnostics.
*   **Operational defaults & opts:** configurable buffer\_days, min\_train\_size, val\_fraction, stable\_seconds, cooldown\_seconds, min\_change\_threshold, auto-mask controls and promotion/refit/promotion guards.


## Version 0.1

### Features

*   **Incremental learning:** `SGDRegressor` for fast adaptation to new user overrides.
*   **Periodic full retrains:** `Ridge` with grid search and `TimeSeriesSplit` for robust, stable models.
*   **Pseudo-labeling support:** use current setpoint as weak label with configurable weight and caps.
*   **Safety checks:** plausibility range, minimum change threshold, stability/hysteresis window, cooldown.
*   **SQLite-backed dataset:** explicit separation between features, user labels, and prediction metadata.
*   **FastAPI endpoints:** manual labeling, model inspection, metrics, and manual train triggers.
*   **Configurable scheduling:** sampling, partial updates, inference interval, and scheduled full retrain.
*   **Explainable feature engineering:** time cyclical encodings, weather/thermostat signals, operational status.
*   **Logging and monitoring hooks:** MAE, coefficient drift, and sample-weight summaries.
