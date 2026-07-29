"""
Tests for explainability.local_explainer.

Uses a tiny synthetic XGBoost model -- no external model downloads required.
"""

import os
import tempfile

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
shap = pytest.importorskip("shap")

from explainability.local_explainer import LocalTreeExplainer, LocalExplanation


# ------------------------------------------------------------------ helpers

def _make_tiny_model(n_samples=20, n_features=5, random_state=42):
    """Train a minimal XGBoost binary classifier on synthetic data."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # Linear-ish ground truth for easy reconstruction checks
    rng2 = np.random.RandomState(random_state + 1)
    true_coef = rng2.randn(n_features) * 0.5
    log_odds = X @ true_coef + 0.1
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (prob > np.median(prob)).astype(int)  # ensure both classes present

    model = xgboost.XGBClassifier(
        n_estimators=10, max_depth=3, objective="binary:logistic",
        random_state=random_state, eval_metric="logloss",
    )
    model.fit(X, y)
    return model, X, y


# ------------------------------------------------------------------ tests


class TestLocalTreeExplainer:
    def test_explain_all(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl = LocalTreeExplainer(
            fitted_model=model,
            X=X,
            feature_names=feature_names,
            user_ids=user_ids,
        )
        results = expl.explain_all()
        assert len(results) == len(user_ids)
        for r in results:
            assert isinstance(r, LocalExplanation)
            assert r.user_id in user_ids
            assert 0.0 <= r.probability <= 1.0
            assert r.predicted_label in ("coordinated_deception", "normal")
            assert len(r.contributions) == len(feature_names)

    def test_explain_one(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl = LocalTreeExplainer(model, X, feature_names, user_ids)
        r = expl.explain_one("0")
        assert r.user_id == "0"
        # contributions should be ranked by abs value descending
        contributions = r.contributions
        abs_vals = [abs(c.contribution) for c in contributions]
        assert abs_vals == sorted(abs_vals, reverse=True)
        assert contributions[0].rank == 1

    def test_reconstruction_error_is_small(self):
        """base_value + sum(SHAP) should approximate the raw margin."""
        model, X, y = _make_tiny_model(n_samples=30, n_features=4)
        feature_names = [f"f{i}" for i in range(4)]
        user_ids = [str(i) for i in range(30)]

        expl = LocalTreeExplainer(model, X, feature_names, user_ids)

        for uid in user_ids[:5]:
            exp = expl.explain_one(uid)
            assert exp.reconstruction_error < 5.0, (
                f"Large reconstruction error for {uid}: {exp.reconstruction_error}"
            )

    def test_shape_mismatch_raises(self):
        model, X, y = _make_tiny_model(n_samples=10, n_features=3)
        with pytest.raises(ValueError):
            LocalTreeExplainer(model, X, ["a", "b"], ["0"] * 10)

        with pytest.raises(ValueError):
            LocalTreeExplainer(model, X, ["a", "b", "c"], ["0"] * 5)

    def test_nonexistent_user_raises(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl = LocalTreeExplainer(model, X, feature_names, user_ids)
        with pytest.raises(KeyError):
            expl.explain_one("nonexistent")

    def test_contribution_directions(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl = LocalTreeExplainer(model, X, feature_names, user_ids)
        r = expl.explain_one("0")
        for c in r.contributions:
            assert c.direction in ("supporting", "opposing")

    def test_save_csv(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl = LocalTreeExplainer(model, X, feature_names, user_ids)
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "contribs.csv")
        expl.save_csv(csv_path)
        assert os.path.isfile(csv_path)

        with open(csv_path, "r") as fh:
            header = fh.readline()
        assert "user_id" in header
        assert "feature_name" in header
        assert "contribution" in header

    def test_save_jsonl(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl = LocalTreeExplainer(model, X, feature_names, user_ids)
        tmpdir = tempfile.mkdtemp()
        jsonl_path = os.path.join(tmpdir, "explanations.jsonl")
        expl.save_jsonl(jsonl_path)
        assert os.path.isfile(jsonl_path)

        with open(jsonl_path, "r") as fh:
            lines = fh.readlines()
        assert len(lines) == len(user_ids)

    def test_threshold_affects_label(self):
        model, X, y = _make_tiny_model()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        user_ids = [str(i) for i in range(X.shape[0])]

        expl_low = LocalTreeExplainer(model, X, feature_names, user_ids, threshold=0.0)
        expl_high = LocalTreeExplainer(model, X, feature_names, user_ids, threshold=1.0)

        r_low = expl_low.explain_one("0")
        r_high = expl_high.explain_one("0")

        # With threshold=0.0, everything should be "coordinated_deception"
        assert r_low.predicted_label == "coordinated_deception"
        # With threshold=1.0, everything should be "normal"
        assert r_high.predicted_label == "normal"
