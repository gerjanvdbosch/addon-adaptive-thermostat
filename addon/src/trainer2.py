import os
import shutil
import logging
import joblib
import time
import math
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from db import fetch_training_data, fetch_unlabeled, update_sample_prediction
from collector import FEATURE_ORDER
from scipy.stats import loguniform, randint

logger = logging.getLogger(__name__)


def _atomic_dump(obj: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _assemble_matrix(rows, feature_order):
    """
    Bouw X en y, maar y is delta = label_setpoint - current_setpoint.
    Rows zonder current_setpoint worden overgeslagen.
    """
    X = []
    y = []
    used_rows = []
    for r in rows:
        feat = r.data if r.data and isinstance(r.data, dict) else None
        if not feat:
            continue
        try:
            label = float(r.label_setpoint)
            current = feat.get("current_setpoint")
            if current is None:
                # cannot compute delta zonder current_setpoint
                logger.debug(
                    "Skipping row %s: missing current_setpoint", getattr(r, "id", None)
                )
                continue
            current = float(current)
            # sanity bounds on label
            if not (14 <= label <= 25.0):
                logger.info("MLTrainer: invalid temp %s", str(label))
                continue

            # compute delta target
            label_delta = label - current

            # build feature vector (coerce to floats; keep current_setpoint optionally)
            vec = []
            for k in feature_order:
                v = feat.get(k)
                if v is None:
                    v = 0.0
                else:
                    try:
                        v = float(v)
                    except Exception:
                        logger.warning(
                            "Feature %s value not numeric in training row %s: %r; coercing to 0.0",
                            k,
                            getattr(r, "id", None),
                            v,
                        )
                        v = 0.0
                vec.append(v)

            X.append(vec)
            y.append(label_delta)
            used_rows.append(r)
        except Exception:
            logger.exception(
                "Skipping corrupt row %s in assemble_matrix", getattr(r, "id", None)
            )
    if not X:
        return None, None, []
    return np.array(X, dtype=float), np.array(y, dtype=float), used_rows


class Trainer2:
    """
    Trainer die het model laat leren DELTA = label_setpoint - current_setpoint.
    Pseudo-labeling is uitgeschakeld als quick-fix; herintroduceer pas met veilige policy.
    """

    def __init__(self, ha_client, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        # self.model_path = self.opts.get("model_path_full")
        self.model_path = "/config/models/full_model2.joblib"
        if not self.model_path:
            raise RuntimeError("model_path_full must be provided in opts.")
        self.feature_order = FEATURE_ORDER
        self.backend = "sklearn_histgb"
        self.random_state = int(self.opts.get("random_state", 42))

    def _fetch_data(self):
        buffer_days = int(self.opts.get("buffer_days", 30))
        rows = fetch_training_data(days=buffer_days)

        # allow opt to control whether only user_override rows count as labeled
        require_user_override = bool(self.opts.get("require_user_override", True))
        labeled_rows = [
            r
            for r in rows
            if r.label_setpoint is not None
            and (not require_user_override or getattr(r, "user_override", False))
        ]
        X_lab, y_lab, used_rows = _assemble_matrix(labeled_rows, self.feature_order)

        pseudo_count = 0
        pseudo_X = None
        pseudo_y = None

        X_list = []
        y_list = []

        if X_lab is not None:
            X_list.append(X_lab)
            y_list.append(y_lab)

        MIN_LABELS_FOR_PSEUDO = int(self.opts.get("min_labels_for_pseudo", 30))
        allow_pseudo_by_config = bool(self.opts.get("use_unlabeled", False))
        if allow_pseudo_by_config and (len(used_rows) >= MIN_LABELS_FOR_PSEUDO):
            try:
                unl = fetch_unlabeled(limit=int(self.opts.get("pseudo_limit", 1000)))
                pseudo_rows = []
                for r in unl:
                    if getattr(r, "label_setpoint", None) is not None:
                        continue
                    feat = r.data if r.data and isinstance(r.data, dict) else None
                    if not feat:
                        continue
                    pseudo_label = feat.get("current_setpoint")
                    if pseudo_label is None:
                        continue
                    # DO NOT blindly use pseudo_label as absolute label when training delta.
                    # If re-enabling pseudo-labeling later, compute pseudo_delta or use model-based pseudo.
                    r.label_setpoint = pseudo_label
                    pseudo_rows.append(r)
                X_pseudo, y_pseudo, _ = _assemble_matrix(
                    pseudo_rows, self.feature_order
                )
                if X_pseudo is not None:
                    pseudo_X = X_pseudo
                    pseudo_y = y_pseudo
                    pseudo_count = len(X_pseudo)
                    logger.info(
                        "MLTrainer: collected %s pseudo samples", str(pseudo_count)
                    )
            except Exception:
                logger.exception(
                    "MLTrainer: failed fetching unlabeled for pseudo labeling"
                )
                pseudo_count = 0
        else:
            if allow_pseudo_by_config:
                logger.info(
                    "MLTrainer: pseudo-labeling disabled because labeled count=%s < min_for_pseudo=%s",
                    str(len(used_rows)),
                    str(MIN_LABELS_FOR_PSEUDO),
                )

        if pseudo_X is not None:
            X_list.append(pseudo_X)
            y_list.append(pseudo_y)

        if not X_list:
            return None, None, None, None, 0

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        n_labeled = len(used_rows)
        n_total = len(y)

        MAX_PSEUDO_MULTIPLIER = float(self.opts.get("max_pseudo_multiplier", 5.0))
        if n_labeled > 0:
            max_allowed_pseudo = int(MAX_PSEUDO_MULTIPLIER * max(1, n_labeled))
            actual_pseudo = n_total - n_labeled
            if actual_pseudo > max_allowed_pseudo:
                X_l = X[:n_labeled]
                y_l = y[:n_labeled]
                X_p = X[n_labeled : n_labeled + max_allowed_pseudo]
                y_p = y[n_labeled : n_labeled + max_allowed_pseudo]
                X = np.vstack([X_l, X_p])
                y = np.concatenate([y_l, y_p])
                pseudo_count = max_allowed_pseudo
                n_total = len(y)
                logger.warning(
                    "Trimmed pseudo samples to %s (max multiplier=%s)",
                    str(pseudo_count),
                    str(MAX_PSEUDO_MULTIPLIER),
                )

        sample_weight = np.ones(n_total, dtype=float)
        if n_total > n_labeled:
            weight_label = float(self.opts.get("weight_label", 1.0))
            weight_pseudo = float(self.opts.get("weight_pseudo", 0.1))
            sample_weight *= weight_label
            sample_weight[n_labeled:n_total] = weight_pseudo

        logger.info(
            "Data fetched: n_labeled=%d n_total=%d n_pseudo=%d",
            n_labeled,
            n_total,
            pseudo_count,
        )

        # quick label stats for debugging potential constant-label problems (report deltas)
        try:
            if y is not None and len(y):
                logger.info(
                    "Training deltas: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
                    len(y),
                    float(np.mean(y)),
                    float(np.std(y)),
                    float(np.min(y)),
                    float(np.max(y)),
                )
        except Exception:
            logger.exception("Failed logging training delta stats")

        return X, y, used_rows, sample_weight, int(pseudo_count)

    def _search_estimator(self):
        base = HistGradientBoostingRegressor(random_state=self.random_state)

        user_dist = self.opts.get("search_param_dist")
        if user_dist:
            return base, user_dist

        compact = {
            "max_iter": [100, 200, 400],
            "max_leaf_nodes": [15, 31, 63],
            "learning_rate": [0.001, 0.01, 0.03],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [1e-6, 0.001, 0.01],
            "max_features": [0.5, 0.7, 1.0],
            "validation_fraction": [0.05, 0.1, 0.15],
        }

        extended = {
            "max_iter": [300, 600, 1000, 1500, 2000],
            "max_leaf_nodes": [15, 31, 63, 127, 255],
            "learning_rate": [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05],
            "min_samples_leaf": [5, 10, 20, 40, 80],
            "l2_regularization": [1e-6, 1e-3, 0.01, 0.1, 1.0],
            "max_features": [0.4, 0.6, 0.8, 1.0],
            "validation_fraction": [0.05, 0.1, 0.15],
        }

        # decide which to return
        mode = self.opts.get("search_mode", "compact")
        expand_flag = bool(self.opts.get("expand_search_next", False))
        n_labeled = int(self.opts.get("last_n_labeled", 0))
        min_labels_to_expand = int(self.opts.get("min_labels_to_expand", 100))

        if expand_flag and n_labeled >= min_labels_to_expand:
            mode = "extended"

        param_dist = compact if mode == "compact" else extended

        # Always use scipy distributions for RandomizedSearchCV
        sampled = {}
        sampled["learning_rate"] = loguniform(1e-4, 1e-1)
        sampled["l2_regularization"] = loguniform(1e-6, 1.0)

        mi_min, mi_max = min(param_dist["max_iter"]), max(param_dist["max_iter"])
        sampled["max_iter"] = randint(mi_min, mi_max + 1)

        mln_min, mln_max = min(param_dist["max_leaf_nodes"]), max(
            param_dist["max_leaf_nodes"]
        )
        sampled["max_leaf_nodes"] = randint(mln_min, mln_max + 1)

        ms_min, ms_max = min(param_dist["min_samples_leaf"]), max(
            param_dist["min_samples_leaf"]
        )
        sampled["min_samples_leaf"] = randint(ms_min, ms_max + 1)

        sampled["max_features"] = param_dist.get("max_features", [1.0])
        sampled["validation_fraction"] = param_dist.get("validation_fraction", [0.1])

        return base, sampled

    def _time_splits(self, n_labeled):
        min_train = int(self.opts.get("min_train_size", 30))
        if n_labeled < min_train:
            return None
        n_splits = min(3, max(2, n_labeled // 10))
        if n_splits >= n_labeled:
            return None
        return n_splits

    def _report_household_drift(self, used_rows, preds_abs, y_true_abs):
        try:
            hh_map = {}
            for i, r in enumerate(used_rows):
                hh = getattr(r, "household_id", None) or "unknown"
                hh_map.setdefault(hh, []).append((preds_abs[i], y_true_abs[i]))
            for hh, vals in hh_map.items():
                errors = [
                    abs(p - t) for p, t in vals if np.isfinite(p) and np.isfinite(t)
                ]
                if not errors:
                    continue
                hh_mae = float(np.mean(errors))
                logger.debug("Household %s MAE=%.4f (n=%d)", hh, hh_mae, len(errors))
        except Exception:
            logger.exception("Failed computing per-household drift")

    def train_job(self, force: bool = False):
        start_time = time.time()
        X, y_delta, used_rows, sample_weight, pseudo_count = self._fetch_data()
        if X is None:
            logger.info("MLTrainer: no training data")
            return

        n_total = len(y_delta)
        n_labeled = len(used_rows)
        self.opts["last_n_labeled"] = n_labeled

        min_search_labels = int(self.opts.get("min_search_labels", 50))
        n_iter_compact = int(self.opts.get("n_iter_compact", 20))
        n_iter_extended = int(self.opts.get("n_iter_extended", 100))

        if n_labeled < min_search_labels:
            self.opts["search_mode"] = "compact"
            n_iter = max(5, n_iter_compact)
            logger.info(
                "Few labeled samples (%d), shrinking hypersearch to n_iter=%d and compact mode",
                n_labeled,
                n_iter,
            )
        else:
            if bool(self.opts.get("expand_search_next", False)) and n_labeled >= int(
                self.opts.get("min_labels_to_expand", 100)
            ):
                n_iter = n_iter_extended
                self.opts["search_mode"] = "extended"
            else:
                n_iter = n_iter_compact
                self.opts["search_mode"] = "compact"

        val_frac = float(self.opts.get("val_fraction", 0.15))
        min_val_size = int(self.opts.get("min_val_size", 30))
        min_train_for_search = int(self.opts.get("min_train_for_search", 10))

        val_size = max(1, int(n_total * val_frac))
        if n_total >= (min_val_size + min_train_for_search):
            if val_size < min_val_size:
                desired_frac = min(0.5, float(min_val_size) / max(1, n_total))
                if desired_frac > val_frac:
                    logger.info(
                        "Increasing val_fraction from %.3f to %.3f to ensure min_val_size=%d (n_total=%d)",
                        val_frac,
                        desired_frac,
                        min_val_size,
                        n_total,
                    )
                    val_frac = desired_frac
                    val_size = max(1, int(n_total * val_frac))
        else:
            # Not enough total samples to satisfy both; ensure at least min_train_for_search remain for training when possible
            max_val_allowed = max(1, n_total - min_train_for_search)
            if val_size > max_val_allowed:
                logger.info(
                    "Reducing val_size from %d to %d to preserve min_train_for_search=%d (n_total=%d)",
                    val_size,
                    max_val_allowed,
                    min_train_for_search,
                    n_total,
                )
                val_size = max_val_allowed

        val_size = min(val_size, max(1, n_total - 1))
        train_idx = slice(0, n_total - val_size)
        val_idx = slice(n_total - val_size, n_total)

        X_train, y_train = X[train_idx], y_delta[train_idx]
        X_val, y_val = X[val_idx], y_delta[val_idx]
        sw_train = sample_weight[train_idx] if sample_weight is not None else None

        logger.info(
            "MLTrainer: train %d, val %d (labeled=%d pseudo=%d)",
            len(y_train),
            len(y_val),
            n_labeled,
            pseudo_count,
        )

        pipe = Pipeline(
            [("model", HistGradientBoostingRegressor(random_state=self.random_state))]
        )

        base_est, param_dist = self._search_estimator()
        # filter param_dist for valid keys on base estimator
        allowed = set(base_est.get_params().keys())
        filtered = {}
        ignored = []
        for k, v in (param_dist or {}).items():
            if k in allowed:
                filtered[k] = v
            else:
                ignored.append(k)
        if ignored:
            logger.warning(
                "Ignoring unsupported hyperparam keys for estimator: %s",
                ", ".join(sorted(ignored)),
            )
        param_dist_pipe = {f"model__{k}": v for k, v in filtered.items()}

        n_jobs = int(self.opts.get("n_jobs", 1))
        tss_splits = self._time_splits(n_labeled)

        try:
            n_train = X_train.shape[0]
        except Exception:
            n_train = 0
        cv = (
            TimeSeriesSplit(n_splits=tss_splits)
            if tss_splits and n_train > tss_splits
            else None
        )
        if cv:
            logger.info(
                "Using TimeSeriesSplit with n_splits=%d (n_train=%d)",
                tss_splits,
                n_train,
            )
        else:
            logger.info(
                "Not using TimeSeriesSplit for hypersearch (tss_splits=%s n_train=%d)",
                str(tss_splits),
                n_train,
            )

        chosen_params = None
        best_pipe = None
        best_score = None
        search_failed = False
        edge_flag = False
        best_iteration = None

        # If not enough training samples, skip expensive RandomizedSearchCV and use fallback fit
        if n_train < min_train_for_search:
            logger.info(
                "Skipping RandomizedSearchCV: n_train=%d < min_train_for_search=%d. Using fallback estimator.",
                n_train,
                min_train_for_search,
            )
            best_pipe = Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=int(self.opts.get("fallback_max_iter", 100)),
                            max_leaf_nodes=int(
                                self.opts.get("fallback_max_leaf_nodes", 31)
                            ),
                            learning_rate=float(
                                self.opts.get("fallback_learning_rate", 0.05)
                            ),
                            random_state=self.random_state,
                        ),
                    )
                ]
            )
            fallback_fit_kwargs = {}
            if sw_train is not None:
                fallback_fit_kwargs["model__sample_weight"] = sw_train
            try:
                best_pipe.fit(X_train, y_train, **fallback_fit_kwargs)
            except Exception:
                logger.exception("MLTrainer: fallback fit failed")
                return
            search_failed = True
        else:
            try:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_dist_pipe,
                    n_iter=n_iter,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=n_jobs,
                    random_state=self.random_state,
                    verbose=0,
                )
                fit_kwargs = {}
                if sw_train is not None:
                    fit_kwargs["model__sample_weight"] = sw_train
                logger.info(
                    "MLTrainer: running hyperparameter search (n_iter=%d, cv=%s)",
                    n_iter,
                    "TimeSeriesSplit" if cv else "None",
                )
                search.fit(X_train, y_train, **fit_kwargs)
                best_pipe = search.best_estimator_
                chosen_params = getattr(search, "best_params_", None)
                best_score = getattr(search, "best_score_", None)
                logger.info(
                    "Hypersearch complete: best_score=%s best_params=%s",
                    str(best_score),
                    str(chosen_params),
                )
            except Exception:
                search_failed = True
                logger.exception(
                    "MLTrainer: search failed; falling back to default estimator"
                )
                best_pipe = Pipeline(
                    [
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                max_iter=300,
                                max_leaf_nodes=63,
                                learning_rate=0.01,
                                random_state=self.random_state,
                            ),
                        )
                    ]
                )
                fallback_fit_kwargs = {}
                if sw_train is not None:
                    fallback_fit_kwargs["model__sample_weight"] = sw_train
                try:
                    best_pipe.fit(X_train, y_train, **fallback_fit_kwargs)
                except Exception:
                    logger.exception("MLTrainer: fallback fit failed")
                    return

        # capture iteration info if present
        try:
            model_obj = best_pipe.named_steps.get("model")
            if hasattr(model_obj, "n_iter_"):
                best_iteration = int(getattr(model_obj, "n_iter_"))
            elif hasattr(model_obj, "best_iteration_"):
                best_iteration = int(getattr(model_obj, "best_iteration_"))
            else:
                best_iteration = None
            if best_iteration is not None:
                logger.info("Best iteration: %s", str(best_iteration))
                chosen_max_iter = None
                if chosen_params:
                    chosen_max_iter = chosen_params.get(
                        "model__max_iter"
                    ) or chosen_params.get("max_iter")
                else:
                    chosen_max_iter = getattr(model_obj, "max_iter", None)

                if chosen_max_iter is not None and best_iteration >= int(
                    chosen_max_iter
                ):
                    logger.warning(
                        "best_iteration (%s) reached max_iter (%s)",
                        str(best_iteration),
                        str(chosen_max_iter),
                    )
                    edge_flag = True
        except Exception:
            logger.exception("Failed to capture best_iteration")

        try:
            final_refit = bool(self.opts.get("refit_on_full", True))
            if final_refit:
                fit_kwargs_all = {}
                if sample_weight is not None:
                    fit_kwargs_all["model__sample_weight"] = sample_weight
                best_pipe.fit(X, y_delta, **fit_kwargs_all)

                def predict_fn(Xq):
                    return best_pipe.predict(Xq)

            else:

                def predict_fn(Xq):
                    return best_pipe.predict(Xq)

        except Exception:
            logger.exception("MLTrainer: failed final refit/predict setup")
            return

        mae = None
        try:
            if n_labeled > 0:
                # Predict deltas then reconstruct absolute preds for meaningful MAE
                preds_delta = predict_fn(X[:n_labeled])
                # reconstruct using each used_row current_setpoint
                current_arr = np.array(
                    [
                        float(r.data.get("current_setpoint", 0.0))
                        for r in used_rows[:n_labeled]
                    ],
                    dtype=float,
                )
                preds_abs = current_arr + np.array(preds_delta, dtype=float)
                y_true_abs = np.array(
                    [float(r.label_setpoint) for r in used_rows[:n_labeled]],
                    dtype=float,
                )
                mae = float(mean_absolute_error(y_true_abs, preds_abs))
                try:
                    self._report_household_drift(
                        used_rows[:n_labeled], preds_abs, y_true_abs
                    )
                except Exception:
                    logger.exception("Per-household drift reporting failed")
        except Exception:
            logger.exception("MLTrainer: failed OOF MAE computation")
            mae = None

        val_mae = None
        try:
            if X_val is not None and len(X_val):
                val_preds_delta = predict_fn(X_val)
                # reconstruct X_val current_setpoint from used rows slice location: use labels in y_val reconstruction is not available
                # best-effort: use last len(X_val) rows of used_rows if ordered; otherwise compute from X_val using feature index
                try:
                    # find index of 'current_setpoint' in feature_order
                    ci = self.feature_order.index("current_setpoint")
                    current_val = X_val[:, ci]
                    val_preds_abs = current_val + np.array(val_preds_delta, dtype=float)
                    # reconstruct true absolute labels for validation: we have y_val as deltas, so need current_val + y_val
                    val_true_abs = current_val + np.array(y_val, dtype=float)
                    val_mae = float(mean_absolute_error(val_true_abs, val_preds_abs))
                except Exception:
                    # fallback: compute MAE on deltas
                    val_mae = float(mean_absolute_error(y_val, val_preds_delta))
        except Exception:
            logger.exception("Failed computing val MAE")
            val_mae = None

        runtime_seconds = time.time() - start_time
        logger.info(
            "Training finished runtime_seconds=%.2f n_labeled=%d n_pseudo=%d mae=%s val_mae=%s chosen_params=%s best_iteration=%s",
            runtime_seconds,
            n_labeled,
            pseudo_count,
            str(mae),
            str(val_mae),
            str(chosen_params),
            str(best_iteration),
        )

        existing_mae = None
        existing_meta = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                obj = joblib.load(self.model_path)
                existing_meta = obj.get("meta", {})
                existing_mae = existing_meta.get("mae")
            except Exception:
                logger.exception(
                    "MLTrainer: failed reading existing model for MAE comparison"
                )

        # Promotion guard and adaptive expansion counters
        MIN_LABELS_PROMOTE = int(self.opts.get("min_labels_promote", 50))
        promotion_delta = float(self.opts.get("promotion_delta_mae", 0.0))
        allow_force = bool(self.opts.get("allow_force", True))
        force_persist_min = int(self.opts.get("force_persist_min_labels", 30))

        if force and not allow_force:
            logger.warning("Force ignored because allow_force not set")
            force = False

        promote = False
        promotion_reason = None

        if force:
            # If there's no existing model on disk, allow force to persist the first model
            if existing_mae is None:
                promote = True
                promotion_reason = "force_and_no_existing_model"
                logger.info(
                    "Force requested and no existing model found; persisting forced model regardless of n_labeled (%d).",
                    n_labeled,
                )
            else:
                # existing model present — keep safety guard: require minimal labeled samples to persist
                if n_labeled >= force_persist_min:
                    promote = True
                    promotion_reason = "force"
                else:
                    logger.info(
                        "Force requested but n_labeled (%d) < force_persist_min (%d); will not persist over existing model",
                        n_labeled,
                        force_persist_min,
                    )
                    promote = False
                    promotion_reason = "force_but_insufficient_labels_for_persist"
        else:
            if existing_mae is None:
                if n_labeled >= int(
                    self.opts.get("min_labels_for_promo_if_no_existing", 20)
                ):
                    promote = True
                    promotion_reason = "no_existing_model"
                else:
                    promote = False
                    promotion_reason = "insufficient_labels_for_initial_promo"
            else:
                if mae is not None and (mae + promotion_delta) < existing_mae:
                    if n_labeled >= MIN_LABELS_PROMOTE:
                        promote = True
                        promotion_reason = "better_mae"
                    else:
                        promote = False
                        promotion_reason = "better_mae_but_insufficient_labels"
                else:
                    promote = False
                    promotion_reason = "not_better"

        # adaptive expansion policy
        edge_count_threshold = int(self.opts.get("edge_count_threshold", 2))
        min_labels_to_expand = int(self.opts.get("min_labels_to_expand", 100))
        require_labels_for_expand_promo = bool(
            self.opts.get("require_labels_for_expand_promo", True)
        )

        if edge_flag:
            self.opts["edge_count"] = int(self.opts.get("edge_count", 0)) + 1
        else:
            self.opts["edge_count"] = 0

        if (
            self.opts.get("edge_count", 0) >= edge_count_threshold
            and n_labeled >= min_labels_to_expand
        ):
            self.opts["expand_search_next"] = True
            self.opts["runs_since_expand"] = 0
            logger.info(
                "Adaptive expansion enabled: edge_count=%d n_labeled=%d -> expand_search_next=True",
                int(self.opts.get("edge_count")),
                n_labeled,
            )
        else:
            self.opts["runs_since_expand"] = (
                int(self.opts.get("runs_since_expand", 0)) + 1
            )

        if (
            self.opts.get("expand_search_next", False)
            and require_labels_for_expand_promo
            and n_labeled < min_labels_to_expand
        ):
            if promote:
                logger.info(
                    "Deferring promotion because expand_search_next=True and n_labeled (%d) < min_labels_to_expand (%d)",
                    n_labeled,
                    min_labels_to_expand,
                )
                promote = False
                promotion_reason = "deferred_due_to_expand_policy"

        promo_log = (
            "Promotion decision: promote={promote} reason={reason} mae={mae} existing_mae={existing} "
            "n_labeled={n} force={force} edge_count={edge} expand_next={expand}"
        ).format(
            promote=str(promote),
            reason=str(promotion_reason),
            mae=("None" if mae is None else f"{mae:.4f}"),
            existing=("None" if existing_mae is None else f"{existing_mae:.4f}"),
            n=str(n_labeled),
            force=str(force),
            edge=str(int(self.opts.get("edge_count", 0))),
            expand=str(bool(self.opts.get("expand_search_next", False))),
        )
        logger.info(promo_log)

        if not promote:
            logger.info(
                "MLTrainer: skipping save (promotion_reason=%s)", str(promotion_reason)
            )
            return

        top_feats = []
        importance_reliable = False
        try:
            model_core = best_pipe.named_steps.get("model")
            imps = getattr(model_core, "feature_importances_", None)
            if imps is not None:
                if np.any(np.asarray(imps, dtype=float) != 0.0):
                    pairs = []
                    for i, v in enumerate(imps):
                        name = (
                            self.feature_order[i]
                            if i < len(self.feature_order)
                            else f"f{i}"
                        )
                        pairs.append((name, float(v)))
                    pairs.sort(key=lambda t: t[1], reverse=True)
                    top_feats = pairs[: min(10, len(pairs))]
                    importance_reliable = True
        except Exception:
            logger.exception("Failed extracting native feature importances")

        try:
            if (
                (not importance_reliable)
                and X_val is not None
                and len(X_val) >= int(self.opts.get("min_val_size", 30))
            ):
                try:
                    import shap  # type: ignore

                    model_for_shap = best_pipe.named_steps.get("model")
                    logger.debug(
                        "SHAP: creating explainer for model type %s",
                        type(model_for_shap),
                    )
                    try:
                        explainer = shap.Explainer(model_for_shap)
                        shap_vals = explainer(X_val)
                        mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
                        inds = np.argsort(mean_abs)[::-1][:10]
                        top_feats = [
                            (self.feature_order[i], float(mean_abs[i])) for i in inds
                        ]
                        importance_reliable = True
                        logger.info(
                            "SHAP: computed top features via shap (n_val=%d)",
                            len(X_val),
                        )
                    except Exception:
                        logger.exception(
                            "SHAP analysis failed inside explainer; will fall back to permutation_importance"
                        )
                except Exception:
                    logger.exception(
                        "SHAP import failed or SHAP usage raised; falling back to permutation_importance"
                    )
        except Exception:
            logger.exception("SHAP wrapper failed")

        try:
            if (
                (not importance_reliable)
                and X_val is not None
                and len(X_val) >= int(self.opts.get("min_val_size", 30))
                and best_pipe is not None
            ):
                try:
                    res = permutation_importance(
                        best_pipe,
                        X_val,
                        y_val,
                        n_repeats=int(self.opts.get("perm_repeats", 30)),
                        random_state=self.random_state,
                        n_jobs=1,
                        scoring="neg_mean_absolute_error",
                    )
                    importances = res.importances_mean
                    if importances is not None and len(importances) == len(
                        self.feature_order
                    ):
                        inds = np.argsort(importances)[::-1][:10]
                        top_feats = [
                            (self.feature_order[i], float(importances[i])) for i in inds
                        ]
                        importance_reliable = True
                    else:
                        logger.warning(
                            "Permutation importance returned unexpected shape: importances=%s",
                            None if importances is None else importances.shape,
                        )
                except Exception:
                    logger.exception("Permutation importance failed")
        except Exception:
            logger.exception("Permutation importance wrapper failed")

        if not importance_reliable:
            logger.warning(
                "Feature importance not reliable for this run: X_val n=%d, n_labeled=%d",
                0 if X_val is None else len(X_val),
                n_labeled,
            )
            top_feats = [
                (name, 0.0)
                for name in self.feature_order[: min(10, len(self.feature_order))]
            ]

        grid_edges = []
        if chosen_params and isinstance(filtered, dict):
            for pk, pv in chosen_params.items():
                key = pk.replace("model__", "")
                vals = filtered.get(key) or (
                    param_dist.get(key) if isinstance(param_dist, dict) else None
                )
                if isinstance(vals, list) and vals:
                    try:
                        if math.isclose(pv, min(vals)) or math.isclose(pv, max(vals)):
                            grid_edges.append(key)
                    except Exception:
                        if str(pv) == str(min(vals)) or str(pv) == str(max(vals)):
                            grid_edges.append(key)

        file_info = {"path": self.model_path}
        try:
            st = os.stat(self.model_path) if os.path.exists(self.model_path) else None
            if st:
                file_info["size_bytes"] = st.st_size
                file_info["modified_ts"] = datetime.fromtimestamp(
                    st.st_mtime, timezone.utc
                ).isoformat()
        except Exception:
            pass

        metadata: Dict[str, Any] = {
            "feature_order": self.feature_order,
            "backend": self.backend,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "mae": mae,
            "val_mae": val_mae,
            "n_samples": n_labeled,
            "pseudo_samples_used": int(pseudo_count),
            "use_unlabeled": bool(self.opts.get("use_unlabeled", False)),
            "chosen_params": chosen_params,
            "search_best_score": best_score,
            "best_iteration": best_iteration,
            "edge_on_param": bool(edge_flag),
            "grid_edges": grid_edges,
            "search_failed": bool(search_failed),
            "random_state": self.random_state,
            "runtime_seconds": runtime_seconds,
            "top_features": [[name, float(imp)] for name, imp in top_feats],
            "feature_importance_reliable": bool(importance_reliable),
            "file": file_info,
            # crucial: mark target type
            "target": "delta",
        }

        try:
            if self.model_path and os.path.exists(self.model_path):
                try:
                    shutil.copy2(self.model_path, self.model_path + ".bak")
                except Exception:
                    logger.warning(
                        "MLTrainer: failed to create .bak for existing model; continuing"
                    )
            payload = {"model": best_pipe, "meta": metadata}
            _atomic_dump(payload, self.model_path)
            logger.info(
                "MLTrainer: saved model (MAE %s -> was %s) trained on %d labeled + %d pseudo samples",
                ("None" if mae is None else f"{mae:.4f}"),
                ("None" if existing_mae is None else f"{existing_mae:.4f}"),
                n_labeled,
                pseudo_count,
            )
        except Exception:
            logger.exception("MLTrainer: failed saving model")
            return

        # update per-sample predictions (reconstruct absolute preds)
        try:
            if n_labeled > 0:
                preds_delta = predict_fn(X[:n_labeled])
                for i, row in enumerate(used_rows[:n_labeled]):
                    try:
                        pred_d = float(preds_delta[i])
                        curr = float(row.data.get("current_setpoint", 0.0))
                        pred_abs = curr + pred_d
                        true_abs = float(row.label_setpoint)
                        err = abs(pred_abs - true_abs) if true_abs is not None else None
                        update_sample_prediction(
                            row.id, predicted_setpoint=pred_abs, prediction_error=err
                        )
                    except Exception:
                        logger.exception(
                            "Failed updating sample prediction for sample %s",
                            getattr(row, "id", None),
                        )
        except Exception:
            logger.exception("Failed to compute/update per-sample predictions")
