from pathlib import Path

import torch

from scripts.lorentz_warm_start import (
    apply_real_schema_warm_start,
    apply_uk_global_warm_start,
)


class _Curvature(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(-1.0))


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.common_curvature = _Curvature()
        self.input_scale_raw = torch.nn.Parameter(torch.tensor(-2.0))


class _SchemaEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metadata = (
            ("user", "tweet"),
            (
                ("user", "posts", "tweet"),
                ("user", "likes", "tweet"),
            ),
        )
        self.common_curvature = _Curvature()
        self.input_scale_raw = torch.nn.Parameter(torch.tensor(-2.0))
        self.input_projectors = torch.nn.ModuleDict({
            "user": torch.nn.Linear(2, 2),
            "tweet": torch.nn.Linear(2, 2),
        })
        layer = torch.nn.Module()
        layer.relations = torch.nn.ModuleDict({
            "user__posts__tweet": torch.nn.Linear(2, 2),
            "user__likes__tweet": torch.nn.Linear(2, 2),
        })
        layer.relation_gates = torch.nn.ParameterDict({
            "user__posts__tweet": torch.nn.Parameter(torch.tensor(0.0)),
            "user__likes__tweet": torch.nn.Parameter(torch.tensor(0.0)),
        })
        layer.fusion_gates = torch.nn.ModuleDict({
            "user": torch.nn.Linear(2, 1),
            "tweet": torch.nn.Linear(2, 1),
        })
        layer.norms = torch.nn.ModuleDict({
            "user": torch.nn.LayerNorm(2),
            "tweet": torch.nn.LayerNorm(2),
        })
        self.layers = torch.nn.ModuleList([layer])


def test_only_schema_agnostic_global_parameters_transfer(tmp_path: Path):
    checkpoint = tmp_path / "uk.pt"
    torch.save({
        "schema_version": "hypertrace.lorentz-pretrain.v1",
        "encoder": {
            "common_curvature.raw": torch.tensor(-3.0),
            "input_scale_raw": torch.tensor(1.5),
        },
    }, checkpoint)
    encoder = _Encoder()
    report = apply_uk_global_warm_start(encoder, checkpoint)
    assert report["transferred_keys"] == ["common_curvature.raw", "input_scale_raw"]
    assert encoder.common_curvature.raw.item() == -3.0
    assert encoder.input_scale_raw.item() == 1.5


def test_real_transfer_copies_only_observed_schema_overlap(tmp_path: Path):
    source = _SchemaEncoder()
    for parameter in source.parameters():
        parameter.data.fill_(7.0)
    checkpoint = tmp_path / "real.pt"
    torch.save({
        "schema_version": "hypertrace.real-operation-pretrain.v1",
        "source_operation": "honduras",
        "source_cutoff": "2019-12-01T00:00:00Z",
        "observed_node_types": ["user", "tweet"],
        "observed_edge_types": [["user", "posts", "tweet"]],
        "encoder": source.state_dict(),
    }, checkpoint)
    target = _SchemaEncoder()
    likes_before = target.layers[0].relations[
        "user__likes__tweet"
    ].weight.detach().clone()

    report = apply_real_schema_warm_start(target, checkpoint)

    assert report["common_edge_types"] == [["user", "posts", "tweet"]]
    assert torch.all(target.layers[0].relations[
        "user__posts__tweet"
    ].weight == 7.0)
    assert torch.equal(
        target.layers[0].relations["user__likes__tweet"].weight,
        likes_before,
    )
    assert not any("likes" in key for key in report["transferred_keys"])
