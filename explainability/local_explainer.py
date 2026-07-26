"""
HyperDecept-WB M1: Local Tree Explainer.

Wraps XGBoost + SHAP to produce per-account feature-contribution explanations.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .schemas import ContributionRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-account explanation container
# ---------------------------------------------------------------------------

@dataclass
class LocalExplanation:
    """SHAP-based local explanation for one user."""

    user_id: str
    probability: float
    predicted_label: str
    threshold: float
    base_value: float
    contributions: List[ContributionRecord] = field(default_factory=list)
    reconstruction_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "probability": self.probability,
            "predicted_label": self.predicted_label,
            "threshold": self.threshold,
            "base_value": self.base_value,
            "contributions": [c.to_dict() for c in self.contributions],
            "reconstruction_error": self.reconstruction_error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class LocalTreeExplainer:
    """Per-account explanation engine for a fitted XGBoost classifier.

    Parameters
    ----------
    fitted_model : xgboost.XGBClassifier
        A trained XGBoost model.
    X : np.ndarray  (n_samples, n_features)
        Feature matrix passed to the model.
    feature_names : List[str]
        Names for every column in *X*.
    user_ids : List[str]
        Per-row user identifiers (aligned with *X*).
    threshold : float
        Decision threshold (default 0.5).
    """

    def __init__(
        self,
        fitted_model,
        X: np.ndarray,
        feature_names: List[str],
        user_ids: List[str],
        threshold: float = 0.5,
    ) -> None:
        self.model = fitted_model
        self.X = np.asarray(X, dtype=float)
        self.feature_names = list(feature_names)
        self.user_ids = [str(uid) for uid in user_ids]
        self.threshold = float(threshold)

        if self.X.shape[0] != len(self.user_ids):
            raise ValueError(
                f"X rows ({self.X.shape[0]}) != user_ids ({len(self.user_ids)})"
            )
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X cols ({self.X.shape[1]}) != feature_names ({len(self.feature_names)})"
            )

        self._shap_values: Optional[np.ndarray] = None
        self._shap_base: Optional[float] = None
        self._probabilities: Optional[np.ndarray] = None
        self._raw_margins: Optional[np.ndarray] = None
        self._explanation_map: Dict[str, LocalExplanation] = {}

        self._compute_shap()

    # ------------------------------------------------------------------
    # SHAP computation
    # ------------------------------------------------------------------

    def _compute_shap(self) -> None:
        """Compute SHAP values and cache per-row probabilities."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for LocalTreeExplainer. "
                "Install with: pip install shap"
            ) from exc

        # -- Try fast TreeExplainer first, fall back to generic Explainer --
        raw_output = None
        explainer = None
        tree_mode = True
        tree_error = None

        try:
            explainer = shap.TreeExplainer(self.model)
        except (ValueError, TypeError, AttributeError) as exc:
            tree_error = exc
            try:
                booster = self.model.get_booster()
                explainer = shap.TreeExplainer(booster)
            except Exception:
                tree_mode = False

        if tree_mode and explainer is not None:
            raw_output = explainer.shap_values(self.X)
            self._parse_tree_shap(raw_output, explainer)
        else:
            # -- Stage 2: native XGBoost pred_contribs ---------------------
            logger.warning(
                "TreeExplainer unavailable (%s), trying native pred_contribs.",
                tree_error,
            )
            import xgboost as xgb_mod

            try:
                dmat = xgb_mod.DMatrix(self.X)
                contribs = self.model.get_booster().predict(
                    dmat, pred_contribs=True
                )
                # contribs shape: (N, F+1) -- last col is bias/base margin
                if contribs.ndim == 2 and contribs.shape[1] == len(self.feature_names) + 1:
                    self._shap_values = np.asarray(contribs[:, :-1], dtype=float)
                    self._shap_base = float(np.mean(contribs[:, -1]))
                else:
                    raise RuntimeError(
                        f"Unexpected pred_contribs shape: {contribs.shape}"
                    )
            except Exception as native_exc:
                # -- Stage 3: model-agnostic Explainer ----------------------
                logger.warning(
                    "pred_contribs also failed (%s), falling back to "
                    "model-agnostic Explainer (slowest).",
                    native_exc,
                )
                try:
                    def _margin_fn(X_arr: np.ndarray) -> np.ndarray:
                        dmat2 = xgb_mod.DMatrix(X_arr)
                        return self.model.get_booster().predict(
                            dmat2, output_margin=True
                        )

                    explainer = shap.Explainer(_margin_fn, self.X)
                    raw_output = explainer(self.X)
                    self._parse_generic_shap(raw_output, explainer)
                except Exception as exc:
                    raise RuntimeError(
                        "All SHAP explainer methods failed. "
                        "Check xgboost/shap version compatibility. "
                        f"Last error: {exc}"
                    ) from exc

        # -- Probabilities & raw margins ------------------------------------
        try:
            self._probabilities = self.model.predict_proba(self.X)[:, 1]
        except Exception:
            self._probabilities = np.full(self.X.shape[0], 0.5)

        eps = 1e-15
        probs_clipped = np.clip(self._probabilities, eps, 1.0 - eps)
        self._raw_margins = np.log(probs_clipped / (1.0 - probs_clipped))

    # ------------------------------------------------------------------
    # SHAP output parsers
    # ------------------------------------------------------------------

    def _parse_tree_shap(self, raw_output, explainer) -> None:
        """Parse SHAP values from TreeExplainer output."""
        if isinstance(raw_output, list):
            if len(raw_output) >= 2:
                sv = np.asarray(raw_output[1], dtype=float)
            else:
                sv = np.asarray(raw_output[0], dtype=float)
        elif hasattr(raw_output, "values"):
            sv = np.asarray(raw_output.values, dtype=float)
            if sv.ndim == 3 and sv.shape[2] == 2:
                sv = sv[:, :, 1]
        else:
            sv = np.asarray(raw_output, dtype=float)

        if sv.ndim == 3:
            sv = sv[:, :, 1]

        self._shap_values = sv

        # -- base value -----------------------------------------------------
        if hasattr(raw_output, "base_values"):
            bv = np.asarray(raw_output.base_values)
            if bv.ndim == 2 and bv.shape[1] >= 2:
                self._shap_base = float(bv[0, 1])
            elif bv.ndim >= 1:
                self._shap_base = float(np.mean(bv))
            else:
                self._shap_base = float(bv)
        elif explainer.expected_value is not None:
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = np.asarray(ev)
                if ev.ndim >= 1 and ev.size >= 2:
                    self._shap_base = float(ev.flat[1])
                else:
                    self._shap_base = float(ev.flat[0])
            else:
                self._shap_base = float(ev)
        else:
            self._shap_base = 0.0

    def _parse_generic_shap(self, raw_output, explainer) -> None:
        """Parse SHAP values from generic Explainer output (margin space)."""
        if hasattr(raw_output, "values"):
            sv = np.asarray(raw_output.values, dtype=float)
        else:
            sv = np.asarray(raw_output, dtype=float)

        # Generic Explainer on margin function returns (N, F).
        if sv.ndim == 3 and sv.shape[2] == 2:
            sv = sv[:, :, 1]

        self._shap_values = sv

        # -- base value -----------------------------------------------------
        if hasattr(raw_output, "base_values"):
            bv = np.asarray(raw_output.base_values)
            self._shap_base = float(np.mean(bv))
        elif explainer is not None and hasattr(explainer, "expected_value"):
            ev = np.asarray(explainer.expected_value)
            self._shap_base = float(np.mean(ev))
        else:
            self._shap_base = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_one(self, user_id: str) -> LocalExplanation:
        """Return the explanation for a single user."""
        user_id = str(user_id)
        if user_id in self._explanation_map:
            return self._explanation_map[user_id]

        try:
            idx = self.user_ids.index(user_id)
        except ValueError:
            raise KeyError(f"user_id {user_id!r} not found in explainer index")

        prob = float(self._probabilities[idx])  # type: ignore[index]
        label = "coordinated_deception" if prob >= self.threshold else "normal"
        sv = self._shap_values[idx]  # type: ignore[index]

        # -- build contributions -------------------------------------------
        contributions: List[ContributionRecord] = []
        for f_idx, f_name in enumerate(self.feature_names):
            shap_val = float(sv[f_idx])
            feature_val = float(self.X[idx, f_idx])
            direction = "supporting" if shap_val > 0 else "opposing"
            contributions.append(ContributionRecord(
                feature_name=f_name,
                feature_value=feature_val,
                contribution=shap_val,
                direction=direction,
                rank=0,  # filled below
            ))

        # Rank by absolute contribution.
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        for rank, c in enumerate(contributions, start=1):
            c.rank = rank

        # -- reconstruction error ------------------------------------------
        sum_shap = float(np.sum(sv))
        raw_margin = float(self._raw_margins[idx])  # type: ignore[index]
        recon_error = abs(self._shap_base + sum_shap - raw_margin)

        explanation = LocalExplanation(
            user_id=user_id,
            probability=prob,
            predicted_label=label,
            threshold=self.threshold,
            base_value=self._shap_base,
            contributions=contributions,
            reconstruction_error=round(recon_error, 8),
        )
        self._explanation_map[user_id] = explanation
        return explanation

    def explain_all(self) -> List[LocalExplanation]:
        """Compute (or return cached) explanations for every user."""
        if len(self._explanation_map) == len(self.user_ids):
            return [self._explanation_map[uid] for uid in self.user_ids]

        for uid in self.user_ids:
            self.explain_one(uid)
        return [self._explanation_map[uid] for uid in self.user_ids]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_jsonl(self, path: str) -> str:
        """Write one JSON object per line (one user per line)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        explanations = self.explain_all()
        with open(path, "w", encoding="utf-8") as fh:
            for exp in explanations:
                fh.write(json.dumps(exp.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Local explanations JSONL saved: %s (%d users)", path, len(explanations))
        return path

    def save_csv(self, path: str) -> str:
        """Write one row per user-feature contribution."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        explanations = self.explain_all()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "user_id", "probability", "predicted_label",
                "feature_name", "feature_value", "contribution",
                "direction", "rank", "base_value", "reconstruction_error",
            ])
            for exp in explanations:
                for c in exp.contributions:
                    writer.writerow([
                        exp.user_id,
                        exp.probability,
                        exp.predicted_label,
                        c.feature_name,
                        c.feature_value,
                        c.contribution,
                        c.direction,
                        c.rank,
                        exp.base_value,
                        exp.reconstruction_error,
                    ])
        logger.info("Local contributions CSV saved: %s (%d rows)", path,
                     len(explanations) * len(self.feature_names))
        return path
