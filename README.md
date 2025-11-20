Adaptive Thermostat
===================

An adaptive thermostat add-on for Home Assistant using scikit-learn. Learns from user setpoint overrides and adapts via conservative incremental updates plus periodic full retrains. Designed for small, growing labeled datasets with safety guards and auditability.

> Experimental — use in test environments only. Defaults are conservative but this project is not production‑hardened. Enable automatic actions only after validation, monitoring and rollback procedures are in place.

Key Features
------------

*   **Incremental learning:** fast partial updates and conservative retrains using scikit-learn estimators.
*   **Periodic full retrains:** HistGradientBoostingRegressor with RandomizedSearchCV and TimeSeriesSplit for robust, stable models.
*   **Delta-model support:** TrainerDelta / InferencerDelta learn and apply deltas (predicted\_delta + explicit baseline) to reduce echo/shadow learning.
*   **Masking of current\_setpoint:** current\_setpoint is masked during training and inference to prevent trivial identity predictions.
*   **Pseudo-labeling support:** optional pseudo-labeling from unlabeled samples’ current\_setpoint with configurable limits, weights and max multiplier.
*   **Safety checks:** plausibility ranges, min\_change\_threshold, stability/hysteresis timer, cooldown between actions and sample-age guards.
*   **Feature set (FEATURE\_ORDER):** cyclical time encodings (hour\_sin/hour\_cos, day\_sin/day\_cos, month\_sin/month\_cos), season, day\_or\_night, current\_setpoint (masked), current\_temp, temp\_change, min/max temps (today/tomorrow), solar\_kwh/solar\_chance (today/tomorrow), wind\_speed/dir, outside\_temp, hvac\_mode.
*   **SQLite-backed plumbing:** db helpers for fetch\_training\_data, fetch\_training\_setpoints, fetch\_unlabeled, update\_label/update\_setpoint, insert\_sample and update\_sample\_prediction for provenance.
*   **Inference engines:** Inferencer (full-model) and InferencerDelta (delta-model) with baseline reconstruction, masked vectors and per-sample persistence.
*   **Explainability & metrics:** saved model.meta includes feature\_order, trained\_at, mae/val\_mae, n\_samples, chosen\_params, runtime\_seconds, top\_features (native/SHAP/permutation) and file diagnostics.
*   **Operational defaults & opts:** configurable buffer\_days, min\_train\_size, val\_fraction, stable\_seconds, cooldown\_seconds, min\_change\_threshold, auto-mask controls and promotion/refit/promotion guards.


Installation
------------

Use the following URL to add this repository:

```txt
https://github.com/gerjanvdbosch/addon-adaptive-thermostat
```

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fgerjanvdbosch%2Faddon-adaptive-thermostat)