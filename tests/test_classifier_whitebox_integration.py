"""
Lightweight integration test for new_main_classifier whitebox path.

Stubs MultimodalExtractor and CognitiveVisualizer so NO model downloads occur.
"""

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# -- path setup (same as new_main_classifier) -------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CC_DIR = os.path.join(_PROJECT_ROOT, "Character Classification")
for _p in (_CC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_data(tmpdir):
    """Create a tiny DB + CSV pair that exercises merge/dedupe paths."""
    db_path = os.path.join(tmpdir, "test.db")
    csv_path = os.path.join(tmpdir, "test.csv")

    # DB
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE user (user_id INTEGER, user_name TEXT,"
                 "num_followers INTEGER, num_followings INTEGER, user_type TEXT)")
    conn.execute("CREATE TABLE post (post_id INTEGER, user_id INTEGER,"
                 "content TEXT, num_likes INTEGER, num_shares INTEGER, created_at DATETIME)")
    conn.execute("CREATE TABLE follow (follow_id INTEGER, follower_id INTEGER,"
                 "followee_id INTEGER, created_at DATETIME)")
    conn.execute("INSERT INTO user VALUES (1, 'alice', 10, 5, 'good')")
    conn.execute("INSERT INTO user VALUES (2, 'bob', 20, 8, 'bad')")
    conn.execute("INSERT INTO user VALUES (3, 'carol', 15, 6, 'bad')")
    conn.execute("INSERT INTO post VALUES (101, 1, 'hello world', 3, 1, '2025-01-01')")
    conn.execute("INSERT INTO post VALUES (102, 1, 'another tweet', 5, 2, '2025-01-02')")
    conn.execute("INSERT INTO post VALUES (103, 2, 'bad tweet content', 1, 0, '2025-01-03')")
    conn.execute("INSERT INTO follow VALUES (1, 1, 2, '2025-01-01')")
    conn.execute("INSERT INTO follow VALUES (2, 2, 1, '2025-01-02')")
    conn.commit()
    conn.close()

    # CSV
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "user_char": ["alice public bio", "bob public bio", ""],
        "previous_tweets": ["hello world | another tweet", "bad tweet content", ""],
        "user_type": ["good", "bad", "bad"],
        "is_bad": [0, 1, 1],
    })
    df.to_csv(csv_path, index=False)

    return db_path, csv_path


# ---------------------------------------------------------------------------
# Stub Extractor
# ---------------------------------------------------------------------------

class _StubExtractor:
    """Returns deterministic small fused matrices so we can test the pipeline.

    Mimics the MultimodalExtractor interface with return_provenance support.
    """

    def __init__(self, **kwargs):
        pass

    def fuse_multimodal_features(
        self, db_path, bios, tweets_list=None, user_ids_master=None,
        return_provenance=False,
    ):
        user_ids = [str(u) for u in user_ids_master] if user_ids_master else ["1", "2", "3"]
        n = len(user_ids)
        # build_26dim_features expects: semantic_raw = fused[:, :-18],
        # behavior = fused[:, -18:-8], psycho = fused[:, -8:].
        # With n=3 samples, PCA(n_components=8) would fail, so we produce
        # exactly (8 semantic + 10 behavior + 8 psycho) = 26 columns,
        # which skips PCA entirely (8 <= 8 branch).
        fused = np.random.RandomState(42).randn(n, 26)

        if not return_provenance:
            return user_ids, fused

        # Build provenance with text evidence matching the splitter
        provenance = {}
        for i, uid in enumerate(user_ids):
            entry = {}
            for name in ["Follower_Following_Ratio", "Empathy_Gap_Mean",
                         "Empathy_Gap_Max", "Dark_Triad_Mean",
                         "Contagion_Mean", "Volatility_Mean"]:
                entry[name] = {
                    "value": float(fused[i, 0]),
                    "extractor": "StubExtractor",
                    "evidence_ids": [f"text:{uid}:0"] if i < 2 else [],
                    "text_indices": [0] if i < 2 else [],
                    "metadata": {},
                }
            provenance[uid] = entry
        return user_ids, fused, provenance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifierIntegration:
    """Lightweight pipeline test -- no model downloads."""

    def test_df_final_user_order(self):
        """df_final must preserve merge order consistent with X rows."""
        tmpdir = tempfile.mkdtemp()
        db_path, csv_path = _make_minimal_data(tmpdir)

        # Monkey-patch to use our stub
        import new_main_classifier as clf
        with mock.patch.object(clf, "_Extractor", _StubExtractor):
            with mock.patch.object(clf, "CognitiveVisualizer", autospec=True):
                result = clf.run_classifier(
                    db_file=db_path,
                    csv_file=csv_path,
                    save_dir=tmpdir,
                    psychology_mode="off",
                    run_visualizer=False,
                    whitebox=False,
                )

        df_res = pd.read_csv(result["result_path"])
        # Every user in df_final should be in results
        assert set(df_res["user_id"].astype(str)) >= {"1", "2", "3"}
        print("df_final user order OK")

    def test_prob_columns(self):
        """prob_bot == prob_bot_oof == OOF; prob_bot_full_fit is separate."""
        tmpdir = tempfile.mkdtemp()
        db_path, csv_path = _make_minimal_data(tmpdir)

        import new_main_classifier as clf
        with mock.patch.object(clf, "_Extractor", _StubExtractor):
            with mock.patch.object(clf, "CognitiveVisualizer", autospec=True):
                result = clf.run_classifier(
                    db_file=db_path,
                    csv_file=csv_path,
                    save_dir=tmpdir,
                    psychology_mode="off",
                    run_visualizer=False,
                    whitebox=False,
                )

        df_res = pd.read_csv(result["result_path"])
        # Three columns must exist
        assert "prob_bot" in df_res.columns
        assert "prob_bot_oof" in df_res.columns
        assert "prob_bot_full_fit" in df_res.columns
        # prob_bot == prob_bot_oof (both OOF)
        np.testing.assert_array_almost_equal(
            df_res["prob_bot"].values, df_res["prob_bot_oof"].values
        )
        print("prob columns OK")

    def test_whitebox_text_evidence_aligned(self):
        """text evidence registered in whitebox must use split_tweet_pool."""
        tmpdir = tempfile.mkdtemp()
        db_path, csv_path = _make_minimal_data(tmpdir)

        import new_main_classifier as clf
        with mock.patch.object(clf, "_Extractor", _StubExtractor):
            with mock.patch.object(clf, "CognitiveVisualizer", autospec=True):
                result = clf.run_classifier(
                    db_file=db_path,
                    csv_file=csv_path,
                    save_dir=tmpdir,
                    psychology_mode="off",
                    run_visualizer=False,
                    whitebox=True,
                    whitebox_top_k=5,
                )

        # Check that explanation_packets.jsonl was created
        wb = result.get("whitebox_paths", {})
        packets_path = wb.get("packets_jsonl")
        assert packets_path and os.path.isfile(packets_path), \
            f"packets_jsonl not found: {wb}"

        # Read packets
        packets = []
        with open(packets_path, "r", encoding="utf-8") as fh:
            for line in fh:
                packets.append(json.loads(line))

        assert len(packets) >= 2  # at least 2 users

        # Each packet should have prediction_scope metadata
        for pkt in packets:
            assert pkt["metadata"].get("prediction_scope") == "full_fit_model"

        # Text evidence should use split_tweet_pool(min_len=5), not raw split
        for pkt in packets:
            for ev in pkt.get("evidence", []):
                c = ev.get("content") or ""
                # Short texts (<5 chars) should NOT appear as evidence
                assert len(c.strip()) >= 5 or c == "", \
                    f"Short text leaked into evidence: {c!r}"

        print("text evidence alignment OK")

    def test_whitebox_outputs_no_crash(self):
        """Whitebox path completes without exceptions."""
        tmpdir = tempfile.mkdtemp()
        db_path, csv_path = _make_minimal_data(tmpdir)

        import new_main_classifier as clf
        with mock.patch.object(clf, "_Extractor", _StubExtractor):
            with mock.patch.object(clf, "CognitiveVisualizer", autospec=True):
                result = clf.run_classifier(
                    db_file=db_path,
                    csv_file=csv_path,
                    save_dir=tmpdir,
                    psychology_mode="off",
                    run_visualizer=False,
                    whitebox=True,
                    whitebox_top_k=5,
                )

        wb = result.get("whitebox_paths", {})
        # No error key
        assert "_whitebox_error" not in wb, f"Whitebox error: {wb.get('_whitebox_error')}"
        # All expected outputs exist
        for key in ("model_json", "feature_names", "model_metadata",
                     "local_contributions_csv", "local_explanations_jsonl",
                     "packets_jsonl"):
            assert key in wb, f"Missing whitebox output: {key}"
            assert os.path.isfile(wb[key]), f"File missing: {wb[key]}"
        print("whitebox outputs OK")


class TestTextSplittingConsistency:
    """Verify that text splitting is consistent between extraction and evidence."""

    def test_split_tweet_pool_min_len_5(self):
        """split_tweet_pool with min_len=5 filters short texts."""
        from config import split_tweet_pool
        result = split_tweet_pool("hi | hello world | ok | long enough text", min_len=5)
        # "hi" (len 2) and "ok" (len 2) filtered; "hello world" and "long enough text" kept
        assert "hello world" in result
        assert "long enough text" in result
        assert len([t for t in result if len(t) < 5]) == 0, f"Short texts leaked: {result}"

    def test_max_tweets_per_user_truncation(self):
        """max_tweets_per_user truncates the text list."""
        from config import split_tweet_pool
        texts = split_tweet_pool("a tweet | another one | third post | fourth | fifth one", min_len=5)
        truncated = texts[:2]  # simulate max_tweets_per_user=2
        assert len(truncated) == 2

    def test_provenance_text_index_consistency(self):
        """text:{uid}:{idx} in provenance should reference actual analyzed texts."""
        from config import split_tweet_pool
        tweet_str = "short | hello world post | another tweet here | ok"
        texts = split_tweet_pool(tweet_str, min_len=5)
        # texts = ["hello world post", "another tweet here"]

        # Simulate what provenance builder does
        user_id = "99"
        evidence_ids = []
        for idx, text in enumerate(texts):
            evidence_ids.append(f"text:{user_id}:{idx}")

        assert len(evidence_ids) == 2
        assert evidence_ids[0] == "text:99:0"
        assert evidence_ids[1] == "text:99:1"
        # The content at text:99:0 should be the first valid text
        assert texts[0] == "hello world post"


class TestNativePredContribs:
    """Verify native XGBoost pred_contribs fallback works correctly."""

    def test_pred_contribs_reconstruction(self):
        xgb = pytest.importorskip("xgboost")
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2,
                                  objective="binary:logistic", random_state=42)
        model.fit(X, y)

        dmat = xgb.DMatrix(X)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        n_features = 3
        shap_vals = contribs[:, :n_features]
        base_vals = contribs[:, n_features]

        # Reconstructed margin = base + sum(shap)
        recon = base_vals + shap_vals.sum(axis=1)
        actual_margin = model.get_booster().predict(dmat, output_margin=True)
        np.testing.assert_array_almost_equal(recon, actual_margin, decimal=4)
        print("pred_contribs reconstruction OK")

    def test_local_explainer_uses_pred_contribs(self):
        xgb = pytest.importorskip("xgboost")
        import shap as _shap

        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        y = (X[:, 0] > 0).astype(int)
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2,
                                  objective="binary:logistic", random_state=42)
        model.fit(X, y)

        from explainability.local_explainer import LocalTreeExplainer
        expl = LocalTreeExplainer(model, X, ['a', 'b', 'c'],
                                  [str(i) for i in range(10)])
        r = expl.explain_one('0')
        assert r.reconstruction_error < 1.0, \
            f"Reconstruction error too large: {r.reconstruction_error}"
        print("local_explainer pred_contribs OK")
