import json
import pandas as pd

from data_processing.materialize_coordination_bundle import FEATURE_COLUMNS, materialize


def test_materialized_fake_bundle_has_26d_and_availability_sidecar(tmp_path):
    pd.DataFrame({"user.id_str": ["u1", "u2"], "cib": ["campaign", None]}).to_csv(tmp_path / "users_ids.csv", index=False)
    out = tmp_path / "bundle"
    manifest = materialize("fake-accounts-activity", tmp_path, out)
    features = pd.read_csv(out / "features_26d.csv")
    availability = pd.read_csv(out / "feature_availability.csv")
    assert features.columns.tolist() == ["user_id", *FEATURE_COLUMNS]
    assert availability.columns.tolist() == ["node_id", *FEATURE_COLUMNS]
    assert manifest["feature_dim"] == 26
    assert manifest["llm_origin_labels"] is False
    assert json.loads((out / "manifest.json").read_text())["feature_dim"] == 26

