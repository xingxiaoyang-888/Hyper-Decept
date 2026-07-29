import torch

from data_processing.mgtab_raw_audit import (
    EXPECTED_FEATURES,
    RELATION_TYPES,
    audit_mgtab_directory,
    render_markdown,
)


def _write_tensor_bundle(root, *, invalid=False):
    features = torch.zeros((4, EXPECTED_FEATURES), dtype=torch.float32)
    features[:, :20] = torch.tensor([
        [0, 0, 1, 0.1, 0, 0.2, 0.3, 0.4, 1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0.2, 1, 0.3, 0.4, 0.5, 0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0.3, 0, 0.4, 0.5, 0.6, 1, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0, 1, 0, 1, 0],
        [1, 1, 1, 0.4, 1, 0.5, 0.6, 0.7, 0, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 1, 0, 1, 0, 1],
    ])
    features[:, 20:] = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    if invalid:
        features[0, 20] = float("nan")
    edge_index = torch.tensor([
        list(range(4)) + [0, 1, 2],
        [1, 2, 3, 0, 2, 3, 0],
    ], dtype=torch.int64)
    if invalid:
        edge_index[1, 0] = 9
    torch.save(edge_index, root / "edge_index.pt")
    torch.save(torch.arange(len(RELATION_TYPES)), root / "edge_type.pt")
    torch.save(torch.ones(len(RELATION_TYPES)), root / "edge_weight.pt")
    torch.save(features, root / "features.pt")
    torch.save(torch.tensor([0, 1, 0, 1]), root / "labels_bot.pt")
    torch.save(torch.tensor([0, 1, 2, 0]), root / "labels_stance.pt")


def test_audit_profiles_schema_without_exporting_rows(tmp_path):
    _write_tensor_bundle(tmp_path)
    report = audit_mgtab_directory(tmp_path, with_sha256=False)

    assert report["detected_kind"] == "mgtab_standard_tensor_bundle"
    assert report["summary"]["node_count"] == 4
    assert report["summary"]["edge_row_count"] == 7
    assert report["schema_contract"]["feature_columns"][0]["name"] == (
        "profile_use_background_image"
    )
    assert report["schema_contract"]["relation_types"]["6"] == "hashtag"
    assert report["label_distributions"]["bot"]["1"]["count"] == 2
    assert "tensor([[" not in str(report)
    assert "MGTAB 标准版字段" in render_markdown(report)


def test_audit_rejects_nonfinite_features_and_orphan_edges(tmp_path):
    _write_tensor_bundle(tmp_path, invalid=True)
    report = audit_mgtab_directory(tmp_path, with_sha256=False)
    codes = {item["code"] for item in report["findings"]}

    assert report["status"] == "fail"
    assert "invalid_edge_endpoint" in codes
    assert "nonfinite_numeric_value" in codes


def test_missing_bundle_is_reported_without_loading(tmp_path):
    report = audit_mgtab_directory(tmp_path, with_sha256=False)

    assert report["detected_kind"] == "incomplete_or_unknown"
    assert report["status"] == "fail"
    assert set(report["missing_required_files"])
