from argparse import Namespace
import json

import pandas as pd
import pytest

from data_processing import prepare_p2_smoke_data as preparation


class _Bundle:
    def __init__(self):
        self.core_users = pd.DataFrame({"user_id": ["1"]})
        self.boundary_users = pd.DataFrame({"user_id": []})
        self.labels = pd.DataFrame({"user_id": ["1"], "is_bad": [0]})
        self.follow_edges = pd.DataFrame(columns=["follower_id", "followee_id"])
        self.actions = pd.DataFrame(columns=["actor_id", "target_id"])
        self.relations = pd.DataFrame(columns=["source_id", "target_id"])
        self.posts = pd.DataFrame(columns=["post_id", "author_id"])

    def manifest(self):
        return {
            "dataset_kind": "twibot22_raw",
            "counts": {"core_users": 1},
            "warnings": [],
        }


class _Adapter:
    def __init__(self, *_args, **_kwargs):
        pass

    def load(self):
        return _Bundle()


def _args(tmp_path, *, mode):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "label.csv").write_text("id,label\n1,human\n", encoding="utf-8")
    core = tmp_path / "core_ids.txt"
    core.write_text("1\n", encoding="utf-8")
    return Namespace(
        twibot_dir=source,
        core_ids=core,
        output_dir=tmp_path / "bundle",
        expected_core_count=1,
        edge_chunksize=10,
        mode=mode,
        raw_audit=None,
        selection_audit=None,
    )


def test_formal_twibot_requires_audit_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "TwiBot22RawAdapter", _Adapter)
    with pytest.raises(ValueError, match="raw audit and selection audit"):
        preparation.prepare_twibot(_args(tmp_path, mode="formal"))


def test_formal_twibot_records_audit_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "TwiBot22RawAdapter", _Adapter)
    args = _args(tmp_path, mode="formal")
    args.raw_audit = tmp_path / "raw_audit.json"
    args.selection_audit = tmp_path / "selection_audit.json"
    args.raw_audit.write_text('{"status":"passed"}\n', encoding="utf-8")
    args.selection_audit.write_text('{"selected":1}\n', encoding="utf-8")

    manifest_path = preparation.prepare_twibot(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "hyperdecept.twibot22-formal.v1"
    assert manifest["mode"] == "formal"
    assert manifest["smoke_only"] is False
    assert len(manifest["formal_provenance"]["raw_audit"]["sha256"]) == 64
    assert len(manifest["formal_provenance"]["selection_audit"]["sha256"]) == 64


def test_smoke_twibot_remains_default_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "TwiBot22RawAdapter", _Adapter)
    args = _args(tmp_path, mode="smoke")

    manifest_path = preparation.prepare_twibot(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "hyperdecept.twibot22-smoke.v1"
    assert manifest["smoke_only"] is True
    assert "formal_provenance" not in manifest
