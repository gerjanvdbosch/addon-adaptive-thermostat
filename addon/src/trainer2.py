import os
import shutil
import logging
import joblib
import time
import math
import numpy as np
from datetime import datetime
from typing import Any, Dict

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

# try to import scipy distributions for RandomizedSearch; fallback gracefully
try:
    from scipy.stats import loguniform, randint

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from db import fetch_training_data, fetch_unlabeled, update_sample_prediction
from collector import FEATURE_ORDER

logger = logging.getLogger(__name__)


def _atomic_dump(obj: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _assemble_matrix(rows, feature_order):
    X = []
    y = []
    used_rows = []
    for r in rows:
        feat = r.data if r.data and isinstance(r.data, dict) else None
        if not feat:
            continue
        try:
            label = float(r.label_setpoint)
            if not (14 <= label <= 25.0):
                logger.info("MLTrainer: invalid temp %s", str(label))
                continue

            vec = [
                feat.get(k) if feat.get(k) is not None else 0.0 for k in feature_order
            ]
            X.append(vec)
            y.append(label)
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
    Production-ready sklearn ML trainer using HistGradientBoostingRegressor.

    Optional opts keys (use defaults as needed):
      - model_path_full (required)
      - buffer_days: 30
      - use_unlabeled: True
      - pseudo_limit: 1000
      - weight_label: 1.0
      - weight_pseudo: 0.1
      - val_fraction: 0.15
      - n_jobs: 1
      - n_iter_search: 20
      - search_mode: 'compact' or 'extended'
      - min_train_size: 30
      - min_search_labels: 50
      - refit_on_full: False
      - early_stopping_rounds: 50
      - promotion_delta_mae: 0.0
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
        labeled_rows = [
            r
            for r in rows
            if r.label_setpoint is not None and getattr(r, "user_override", False)
        ]
        X_lab, y_lab, used_rows = _assemble_matrix(labeled_rows, self.feature_order)
        X_list, y_list = ([], [])
        if X_lab is not None:
            X_list.append(X_lab)
            y_list.append(y_lab)

        pseudo_count = 0
        pseudo_X = None
        pseudo_y = None

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

        weight_label = float(self.opts.get("weight_label", 1.0))
        weight_pseudo = float(self.opts.get("weight_pseudo", 0.1))
        sample_weight = np.ones(n_total, dtype=float) * weight_label
        if n_total > n_labeled:
            sample_weight[n_labeled:n_total] = weight_pseudo

        logger.info(
            "Data fetched: n_labeled=%d n_total=%d n_pseudo=%d",
            n_labeled,
            n_total,
            pseudo_count,
        )
        return X, y, used_rows, sample_weight, int(pseudo_count)

    def _search_estimator(self):
        # default base estimator
        base = HistGradientBoostingRegressor(random_state=self.random_state)

        user_dist = self.opts.get("search_param_dist")
        if user_dist:
            return base, user_dist

        # compact grid for small data
        compact = {
            "max_iter": [100, 200, 400],
            "max_leaf_nodes": [15, 31, 63],
            "learning_rate": [0.01, 0.03, 0.05],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 0.01, 0.1],
            "max_features": [0.6, 0.8, 1.0],
            "validation_fraction": [0.1, 0.15],
        }

        # extended grid used only when enough labels and expand flag set
        extended = {
            "max_iter": [300, 600, 1000, 1500, 2000],
            "max_leaf_nodes": [15, 31, 63, 127, 255],
            "learning_rate": [0.001, 0.003, 0.005, 0.01, 0.02, 0.05],
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

        # If scipy is available, convert some params to distributions for RandomizedSearch
        if SCIPY_AVAILABLE:
            dist = {}
            # learning_rate loguniform, l2 loguniform, max_iter and max_leaf_nodes randint
            dist["learning_rate"] = loguniform(1e-4, 1e-1)
            dist["l2_regularization"] = loguniform(1e-6, 1.0)
            dist["max_iter"] = randint(
                min(param_dist["max_iter"]), max(param_dist["max_iter"]) + 1
            )
            dist["max_leaf_nodes"] = randint(
                min(param_dist["max_leaf_nodes"]), max(param_dist["max_leaf_nodes"]) + 1
            )
            dist["min_samples_leaf"] = randint(
                min(param_dist["min_samples_leaf"]),
                max(param_dist["min_samples_leaf"]) + 1,
            )
            # categorical-like remain lists
            dist["max_features"] = param_dist.get("max_features", [1.0])
            dist["validation_fraction"] = param_dist.get("validation_fraction", [0.1])
            return base, dist

        return base, param_dist

    def _time_splits(self, n_labeled):
        min_train = int(self.opts.get("min_train_size", 30))
        if n_labeled < min_train:
            return None
        n_splits = min(3, max(2, n_labeled // 10))
        if n_splits >= n_labeled:
            return None
        return n_splits

    def _report_household_drift(self, used_rows, preds, y_true):
        try:
            hh_map = {}
            for i, r in enumerate(used_rows):
                hh = getattr(r, "household_id", None) or "unknown"
                hh_map.setdefault(hh, []).append((preds[i], y_true[i]))
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
        X, y, used_rows, sample_weight, pseudo_count = self._fetch_data()
        if X is None:
            logger.info("MLTrainer: no training data")
            return

        n_total = len(y)
        n_labeled = len(used_rows)
        # keep last seen labeled count for search estimator decision
        self.opts["last_n_labeled"] = n_labeled

        min_search_labels = int(self.opts.get("min_search_labels", 50))
        n_iter_compact = int(self.opts.get("n_iter_compact", 10))
        n_iter_extended = int(self.opts.get("n_iter_extended", 50))

        if n_labeled < min_search_labels:
            self.opts["search_mode"] = "compact"
            n_iter = max(5, n_iter_compact)
            logger.info(
                "Few labeled samples (%d), shrinking hypersearch to n_iter=%d and compact mode",
                n_labeled,
                n_iter,
            )
        else:
            # if expand flag set and we have enough labels, allow extended iterations
            if bool(self.opts.get("expand_search_next", False)) and n_labeled >= int(
                self.opts.get("min_labels_to_expand", 100)
            ):
                n_iter = n_iter_extended
                self.opts["search_mode"] = "extended"
            else:
                n_iter = n_iter_compact
                self.opts["search_mode"] = "compact"

        val_frac = float(self.opts.get("val_fraction", 0.15))
        val_size = max(1, int(n_total * val_frac))
        train_idx = slice(0, n_total - val_size)
        val_idx = slice(n_total - val_size, n_total)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
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
        param_dist_pipe = {}
        for k, v in filtered.items():
            param_dist_pipe[f"model__{k}"] = v

        n_jobs = int(self.opts.get("n_jobs", 1))
        tss_splits = self._time_splits(n_labeled)
        cv = TimeSeriesSplit(n_splits=tss_splits) if tss_splits else None

        chosen_params = None
        best_pipe = None
        best_score = None
        search_failed = False
        edge_flag = False
        best_iteration = None

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
            try:
                if "reg_lambda" in self.opts:
                    best_pipe.set_params(
                        model__l2_regularization=float(self.opts.get("reg_lambda", 1.0))
                    )
                best_pipe.fit(X_train, y_train, model__sample_weight=sw_train)
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

        # detect chosen params on grid edges
        try:
            if chosen_params:
                for k, v in chosen_params.items():
                    key = k.replace("model__", "")
                    vals = filtered.get(key) or (
                        param_dist.get(key) if isinstance(param_dist, dict) else None
                    )
                    # only check if vals is a concrete list
                    if isinstance(vals, list) and vals:
                        try:
                            if math.isclose(v, min(vals)) or math.isclose(v, max(vals)):
                                edge_flag = True
                                logger.warning(
                                    "Chosen param %s=%s is on grid edge", key, str(v)
                                )
                        except Exception:
                            # fallback string compare if numeric check fails
                            if str(v) == str(min(vals)) or str(v) == str(max(vals)):
                                edge_flag = True
                                logger.warning(
                                    "Chosen param %s=%s is on grid edge", key, str(v)
                                )
        except Exception:
            logger.exception("Edge detection failed")

        # final refit (optional)
        try:
            final_refit = bool(self.opts.get("refit_on_full", True))
            if final_refit:
                fit_kwargs_all = {}
                if sample_weight is not None:
                    fit_kwargs_all["model__sample_weight"] = sample_weight
                best_pipe.fit(X, y, **fit_kwargs_all)

                def predict_fn(Xq):
                    return best_pipe.predict(Xq)

            else:

                def predict_fn(Xq):
                    return best_pipe.predict(Xq)

        except Exception:
            logger.exception("MLTrainer: failed final refit/predict setup")
            return

        # OOF MAE on labeled only
        mae = None
        try:
            if n_labeled > 0:
                preds_labeled = predict_fn(X[:n_labeled])
                mae = float(mean_absolute_error(y[:n_labeled], preds_labeled))
                try:
                    self._report_household_drift(
                        used_rows, preds_labeled, y[:n_labeled]
                    )
                except Exception:
                    logger.exception("Per-household drift reporting failed")
        except Exception:
            logger.exception("MLTrainer: failed OOF MAE computation")
            mae = None

        # validation MAE
        val_mae = None
        try:
            if X_val is not None and len(X_val):
                val_preds = predict_fn(X_val)
                val_mae = float(mean_absolute_error(y_val, val_preds))
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

        # load existing model meta if present
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

        if force and not allow_force:
            logger.warning("Force ignored because allow_force not set")
            force = False

        promote = False
        promotion_reason = None

        if force:
            promote = True
            promotion_reason = "force"
        else:
            if existing_mae is None:
                if n_labeled >= int(
                    self.opts.get("min_labels_for_promo_if_no_existing", 1)
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

        # adaptive expansion policy: update counters and decide expand flag
        edge_count_threshold = int(self.opts.get("edge_count_threshold", 2))
        min_labels_to_expand = int(self.opts.get("min_labels_to_expand", 100))
        require_labels_for_expand_promo = bool(
            self.opts.get("require_labels_for_expand_promo", True)
        )

        if edge_flag:
            self.opts["edge_count"] = int(self.opts.get("edge_count", 0)) + 1
        else:
            # decay counter on runs without edge
            self.opts["edge_count"] = 0

        # if best_iteration hit max_iter, treat as edge as well (already set via edge_flag)
        # decide whether to enable expand_search_next for future runs
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

        # If expansion just triggered but labels are insufficient for safe promotion, prevent immediate save
        if (
            self.opts.get("expand_search_next", False)
            and require_labels_for_expand_promo
            and n_labeled < min_labels_to_expand
        ):
            # force promotion False to avoid saving model found at grid edge with little data
            if promote:
                logger.info(
                    "Deferring promotion because expand_search_next=True and n_labeled (%d) < min_labels_to_expand (%d)",
                    n_labeled,
                    min_labels_to_expand,
                )
                promote = False
                promotion_reason = "deferred_due_to_expand_policy"

        # log promotion decision (safe formatted string)
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

        # prepare top features (permutation fallback omitted here for brevity)
        try:
            top_feats = []
            # attempt to extract native importances if available
            model_core = best_pipe.named_steps.get("model")
            if hasattr(model_core, "feature_importances_"):
                imps = getattr(model_core, "feature_importances_", None)
                if imps is not None:
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
        except Exception:
            logger.exception("Failed extracting top features")

        # grid edges list
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

        # file diagnostics
        file_info = {"path": self.model_path}
        try:
            st = os.stat(self.model_path) if os.path.exists(self.model_path) else None
            if st:
                file_info["size_bytes"] = st.st_size
                file_info["modified_ts"] = (
                    datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z"
                )
        except Exception:
            pass

        metadata: Dict[str, Any] = {
            "feature_order": self.feature_order,
            "backend": self.backend,
            "trained_at": datetime.utcnow().isoformat(),
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
            "file": file_info,
        }

        # persist model and meta
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

        # update per-sample predictions
        try:
            if n_labeled > 0:
                preds = predict_fn(X[:n_labeled])
                for i, row in enumerate(used_rows):
                    try:
                        pred = float(preds[i])
                        err = abs(pred - float(y[i])) if y is not None else None
                        update_sample_prediction(
                            row.id, predicted_setpoint=pred, prediction_error=err
                        )
                    except Exception:
                        logger.exception(
                            "MLTrainer: failed updating sample prediction for %s",
                            getattr(row, "id", None),
                        )
        except Exception:
            logger.exception(
                "MLTrainer: failed computing/updating per-sample predictions"
            )
