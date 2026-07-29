import importlib.util
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Character Classification"
    / "lorentz_hgt.py"
)


def _load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("lorentz_hgt_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_data():
    data = HeteroData()
    data["user"].x = torch.randn(5, 4)
    data["tweet"].x = torch.randn(3, 6)
    follows = ("user", "follows", "user")
    posts = ("user", "posts", "tweet")
    data[follows].edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [1, 2, 2, 3, 4]], dtype=torch.long
    )
    data[posts].edge_index = torch.tensor(
        [[0, 1, 3, 4], [0, 1, 1, 2]], dtype=torch.long
    )
    return data, follows, posts


def test_exp_log_roundtrip_and_manifold_constraint():
    module = _load_module()
    tangent = torch.randn(8, 5) * 0.15
    curvature = torch.tensor(0.7)
    point = module.expmap0(tangent, curvature)
    recovered = module.logmap0(point, curvature)
    norm = module.minkowski_dot(point, point).squeeze(-1)
    assert torch.allclose(recovered, tangent, atol=1e-5, rtol=1e-4)
    assert torch.allclose(norm, torch.full_like(norm, -1.0 / 0.7), atol=1e-5)


def test_lorentz_to_poincare_stays_inside_unit_ball():
    module = _load_module()
    point = module.expmap0(torch.randn(16, 4), torch.tensor(1.3))
    ball = module.lorentz_to_poincare(point, torch.tensor(1.3))
    assert torch.all(torch.linalg.vector_norm(ball, dim=-1) < 1.0)


def test_weighted_lorentz_centroid_stays_on_manifold_and_backpropagates():
    module = _load_module()
    curvature = torch.tensor(0.8)
    spatial = torch.randn(3, 4, requires_grad=True) * 0.1
    points = module.lorentz_from_spatial(spatial, curvature)
    weights = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    centroid = module.weighted_lorentz_centroid(
        points.unsqueeze(0), weights.unsqueeze(0), curvature, dim=1
    )
    norm = module.minkowski_dot(centroid, centroid).squeeze(-1)
    assert torch.allclose(norm, torch.tensor([-1.25]), atol=1e-5)
    centroid[..., 1:].sum().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_lorentz_prototype_classifier_uses_valid_learned_prototypes():
    module = _load_module()
    curvature = torch.tensor(1.2)
    classifier = module.LorentzPrototypeClassifier(5, 3)
    points = module.expmap0(torch.randn(7, 5) * 0.1, curvature)
    logits = classifier(points, curvature)
    prototypes = classifier.prototypes(curvature)
    prototype_norms = module.minkowski_dot(
        prototypes, prototypes
    ).squeeze(-1)
    assert logits.shape == (7, 3)
    assert torch.isfinite(logits).all()
    assert torch.allclose(
        prototype_norms,
        torch.full_like(prototype_norms, -1.0 / 1.2),
        atol=1e-5,
    )
    logits.sum().backward()
    assert classifier.prototype_spatial.grad is not None
    assert torch.isfinite(classifier.prototype_spatial.grad).all()


def test_intrinsic_hgt_supports_relation_masks_and_curvature_gradients():
    module = _load_module()
    data, follows, _ = _toy_data()
    model = module.IntrinsicLorentzHGT(
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        metadata=data.metadata(),
        dropout=0.0,
    )
    mask = torch.full((data[follows].edge_index.shape[1],), 0.8, requires_grad=True)
    embedding = model(
        data.x_dict,
        data.edge_index_dict,
        edge_mask_dict={follows: mask},
    )
    assert embedding.shape == (5, 8)
    assert torch.all(torch.linalg.vector_norm(embedding, dim=-1) < 1.0)
    loss = embedding[:, 0].sum()
    loss.backward()
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()
    assert mask.grad.abs().sum() > 0
    assert model.common_curvature.raw.grad is not None


def test_relation_specific_curvatures_are_exposed_for_audit():
    module = _load_module()
    data, follows, posts = _toy_data()
    model = module.IntrinsicLorentzHGT(
        hidden_dim=6,
        num_heads=2,
        num_layers=1,
        metadata=data.metadata(),
        dropout=0.0,
    )
    model(data.x_dict, data.edge_index_dict)
    metadata = model.geometry_metadata()
    assert metadata["geometry_backend"] == "intrinsic_lorentz"
    assert metadata["common_curvature"] < 0.0
    keys = metadata["relation_curvatures"]
    assert f"layer_0:{module.edge_type_name(follows)}" in keys
    assert f"layer_0:{module.edge_type_name(posts)}" in keys
    assert all(value < 0.0 for value in keys.values())
    assert metadata["edge_reliability_gate"] is True
    assert metadata["edge_reliability_attributes"] == [
        "base_weight",
        "multiplicity",
        "temporal_sync",
        "temporal_recency",
        "temporal_available",
    ]
    assert metadata["node_adaptive_self_neighbor_fusion"] is True
    audits = model._last_audits
    assert audits
    assert all(torch.all((audit.reliability >= 0) & (audit.reliability <= 1))
               for audit in audits.values())


def test_edge_attributes_change_reliability_and_temporal_mask_is_strict():
    module = _load_module()
    torch.manual_seed(5)
    curvature = torch.tensor(0.8)
    source = module.expmap0(torch.randn(3, 4) * 0.05, curvature)
    target = module.expmap0(torch.randn(3, 4) * 0.05, curvature)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    relation = module.LorentzRelationMessage(hidden_dim=4, num_heads=2)
    relation.eval()
    with torch.no_grad():
        relation.reliability_attribute_weights.fill_(0.75)

    neutral = {
        "base_weight": torch.ones(3),
        "multiplicity": torch.ones(3),
        "temporal_sync": torch.tensor([0.1, 0.5, 0.9]),
        "temporal_recency": torch.tensor([0.2, 0.6, 1.0]),
        "temporal_available": torch.zeros(3, dtype=torch.bool),
    }
    stronger = {
        **neutral,
        "base_weight": torch.tensor([1.0, 2.0, 4.0]),
        "multiplicity": torch.tensor([1.0, 3.0, 12.0]),
    }
    _, neutral_audit = relation(
        source, target, edge_index, curvature, edge_attributes=neutral
    )
    _, stronger_audit = relation(
        source, target, edge_index, curvature, edge_attributes=stronger
    )
    assert torch.all(stronger_audit.reliability >= neutral_audit.reliability)
    assert torch.any(stronger_audit.reliability > neutral_audit.reliability)

    hidden_temporal = {
        **neutral,
        "temporal_sync": torch.ones(3),
        "temporal_recency": torch.ones(3),
    }
    _, hidden_audit = relation(
        source, target, edge_index, curvature, edge_attributes=hidden_temporal
    )
    assert torch.allclose(hidden_audit.reliability, neutral_audit.reliability)

    available_temporal = {
        **hidden_temporal,
        "temporal_available": torch.ones(3, dtype=torch.bool),
    }
    _, available_audit = relation(
        source, target, edge_index, curvature, edge_attributes=available_temporal
    )
    assert torch.all(available_audit.reliability > hidden_audit.reliability)
    assert torch.all(
        (available_audit.reliability >= 0)
        & (available_audit.reliability <= 1)
    )


def test_misaligned_edge_reliability_attributes_are_rejected():
    module = _load_module()
    curvature = torch.tensor(1.0)
    points = module.expmap0(torch.randn(3, 4) * 0.05, curvature)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    relation = module.LorentzRelationMessage(hidden_dim=4, num_heads=2)
    try:
        relation(
            points,
            points,
            edge_index,
            curvature,
            edge_attributes={"multiplicity": torch.ones(3)},
        )
    except ValueError as error:
        assert "align with edge count" in str(error)
    else:
        raise AssertionError("misaligned edge attributes should be rejected")

    try:
        relation(
            points,
            points,
            edge_index,
            curvature,
            edge_attributes={"base_weight": torch.tensor([1.0, float("nan")])},
        )
    except ValueError as error:
        assert "finite values" in str(error)
    else:
        raise AssertionError("non-finite edge attributes should be rejected")


def test_masked_relation_with_zero_budget_is_numerically_stable():
    module = _load_module()
    data, follows, _ = _toy_data()
    model = module.IntrinsicLorentzHGT(
        hidden_dim=8,
        num_heads=1,
        num_layers=1,
        metadata=data.metadata(),
        dropout=0.0,
    )
    mask = torch.zeros(data[follows].edge_index.shape[1])
    embedding = model(
        data.x_dict,
        data.edge_index_dict,
        edge_mask_dict={follows: mask},
    )
    assert torch.isfinite(embedding).all()
