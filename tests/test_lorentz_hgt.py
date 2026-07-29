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
