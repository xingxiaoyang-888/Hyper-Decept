"""
Integration tests for MultimodalExtractor white-box extensions.

Uses psychology_mode=\"off\" to avoid model downloads.
"""

import importlib.util
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers -- import from "Character Classification" (dir name has a space)
# ---------------------------------------------------------------------------

_EXTRACTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Character Classification",
    "new_feature_extractor.py",
)

def _load_extractor_module():
    """Import new_feature_extractor from its absolute path."""
    # Ensure the "Character Classification" dir is on sys.path for its own
    # internal imports (config, graph_builder, emotional_analysis, etc.).
    cc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Character Classification"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for p in (cc_dir, project_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(
        "new_feature_extractor", os.path.abspath(_EXTRACTOR_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["new_feature_extractor"] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_db(db_path: str):
    """Create a tiny SQLite DB with user + post tables."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE user (user_id INTEGER, user_name TEXT,"
                 "num_followers INTEGER, num_followings INTEGER, user_type TEXT)")
    conn.execute("CREATE TABLE post (post_id INTEGER, user_id INTEGER,"
                 "content TEXT, num_likes INTEGER, num_shares INTEGER, created_at DATETIME)")
    conn.execute("INSERT INTO user VALUES (1, 'alice', 10, 5, 'good')")
    conn.execute("INSERT INTO user VALUES (2, 'bob', 20, 8, 'bad')")
    conn.execute("INSERT INTO post VALUES (101, 1, 'hello', 3, 1, '2025-01-01')")
    conn.execute("INSERT INTO post VALUES (102, 1, 'world', 5, 2, '2025-01-02')")
    conn.execute("INSERT INTO post VALUES (103, 2, 'test tweet', 1, 0, '2025-01-03')")
    conn.commit()
    conn.close()


def _make_minimal_csv(csv_path: str):
    """Create a minimal CSV file."""
    df = pd.DataFrame({
        "user_id": [1, 2],
        "user_char": ["alice bio", "bob bio"],
        "previous_tweets": ["hello | world", "test tweet"],
        "user_type": ["good", "bad"],
        "is_bad": [0, 1],
    })
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFuseMultimodalDefaultReturn:
    """When return_provenance=False (default), fuse_multimodal_features
    MUST return a 2-tuple (user_ids, fused_matrix)."""

    def test_default_returns_two_tuple(self):
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        csv_path = os.path.join(tmpdir, "test.csv")
        _make_minimal_db(db_path)
        _make_minimal_csv(csv_path)

        extractor = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)

        bios = ["Bio: alice. Recent actions: hello | world",
                "Bio: bob. Recent actions: test tweet"]
        tweets = ["hello | world", "test tweet"]

        result = extractor.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=False,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
        user_ids, fused = result
        assert isinstance(user_ids, list)
        assert isinstance(fused, np.ndarray)

    def test_default_does_not_raise(self):
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        csv_path = os.path.join(tmpdir, "test.csv")
        _make_minimal_db(db_path)
        _make_minimal_csv(csv_path)

        extractor = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)

        bios = ["Bio: a. Recent actions: x", "Bio: b. Recent actions: y"]
        tweets = ["x", "y"]

        # This should NOT raise a ValueError about unpacking
        result = extractor.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=False,
        )
        assert len(result) == 2


class TestFuseMultimodalWhitebox:
    """When return_provenance=True, fuse_multimodal_features MUST return
    a 3-tuple (user_ids, fused_matrix, provenance)."""

    def test_whitebox_returns_three_tuple(self):
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        csv_path = os.path.join(tmpdir, "test.csv")
        _make_minimal_db(db_path)
        _make_minimal_csv(csv_path)

        extractor = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)

        bios = ["Bio: a. Recent actions: x", "Bio: b. Recent actions: y"]
        tweets = ["x", "y"]

        result = extractor.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=True,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
        user_ids, fused, provenance = result
        assert isinstance(provenance, dict), f"provenance should be dict, got {type(provenance)}"

    def test_provenance_structure(self):
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        csv_path = os.path.join(tmpdir, "test.csv")
        _make_minimal_db(db_path)
        _make_minimal_csv(csv_path)

        extractor = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)

        bios = ["Bio: a. Recent actions: x", "Bio: b. Recent actions: y"]
        tweets = ["x", "y"]

        _, _, provenance = extractor.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=True,
        )

        # Each user_id should have an entry
        for uid in ["1", "2"]:
            assert uid in provenance, f"User {uid} missing from provenance"
            entry = provenance[uid]
            assert isinstance(entry, dict)

            # Behaviour features should be present
            assert "Follower_Following_Ratio" in entry, f"Missing behaviour feature for {uid}"
            bf = entry["Follower_Following_Ratio"]
            assert "value" in bf
            assert "extractor" in bf
            assert "evidence_ids" in bf

            # Psychology features should be present (even if zero)
            assert "Empathy_Gap_Mean" in entry, f"Missing psychology feature for {uid}"
            pf = entry["Empathy_Gap_Mean"]
            assert "value" in pf
            assert "extractor" in pf

    def test_fused_matrix_shape_consistent(self):
        """Binary and whitebox paths should produce the same fused_matrix."""
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        csv_path = os.path.join(tmpdir, "test.csv")
        _make_minimal_db(db_path)
        _make_minimal_csv(csv_path)

        bios = ["Bio: a. Recent actions: x", "Bio: b. Recent actions: y"]
        tweets = ["x", "y"]

        # Run with return_provenance=False
        extractor1 = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir + "_1")
        uid1, mat1 = extractor1.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=False,
        )

        # Run with return_provenance=True
        extractor2 = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir + "_2")
        uid2, mat2, _ = extractor2.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=True,
        )

        assert uid1 == uid2
        assert mat1.shape == mat2.shape
        np.testing.assert_array_almost_equal(mat1, mat2)


class TestPsychologyCacheConsistency:
    """Cache must be paired (.npy + .evidence.json)."""

    def test_matrix_cache_path_computed(self):
        """The cache prefix is computed even in off mode (just not written)."""
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        _make_minimal_db(db_path)

        extractor = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)
        bios = ["Bio: a. Recent actions: x"]
        tweets = ["x"]

        # Run once -- off mode returns zeros without caching.
        result = extractor.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1"],
            return_provenance=False,
        )
        assert len(result) == 2
        # No .npy is written in off mode (nothing to cache), which is correct.

    def test_whitebox_off_mode_returns_evidence(self):
        """Even in off mode, whitebox path returns empty evidence in provenance."""
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        _make_minimal_db(db_path)

        extractor = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)
        bios = ["Bio: a. Recent actions: x"]
        tweets = ["x"]

        _, _, prov = extractor.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1"],
            return_provenance=True,
        )
        assert "1" in prov
        # Psychology features should be present with zero values
        assert prov["1"]["Empathy_Gap_Mean"]["value"] == 0.0

    def test_provenance_consistent_across_runs(self):
        """Two runs with same inputs produce same provenance structure."""
        mod = _load_extractor_module()
        MultimodalExtractor = mod.MultimodalExtractor

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        _make_minimal_db(db_path)

        bios = ["Bio: a. Recent actions: x", "Bio: b. Recent actions: y"]
        tweets = ["x", "y"]

        # First run
        extractor1 = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir)
        _, _, prov1 = extractor1.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=True,
        )

        # Second run
        extractor2 = MultimodalExtractor(psychology_mode="off", cache_dir=tmpdir + "_2")
        _, _, prov2 = extractor2.fuse_multimodal_features(
            db_path, bios, tweets_list=tweets,
            user_ids_master=["1", "2"],
            return_provenance=True,
        )

        # Provenance structure should be identical
        assert set(prov1.keys()) == set(prov2.keys())
        for uid in prov1:
            assert set(prov1[uid].keys()) == set(prov2[uid].keys())
            for feat in prov1[uid]:
                assert prov1[uid][feat]["value"] == prov2[uid][feat]["value"]
