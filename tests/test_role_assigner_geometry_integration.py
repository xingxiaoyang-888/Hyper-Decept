import importlib.util
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Character Classification"
    / "new_role_assigner.py"
)


def _load_role_assigner():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("new_role_assigner_m2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_hgt_mask_path_supports_gradient():
    module = _load_role_assigner()
    data = HeteroData()
    data["user"].x = torch.randn(4, 3)
    edge_type = ("user", "follows", "user")
    data[edge_type].edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    model = module.HyperRoleHGNN(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        metadata=data.metadata(),
    )
    model(data.x_dict, data.edge_index_dict)

    mask = torch.full((4,), 0.8, requires_grad=True)
    embedding = model(
        data.x_dict,
        data.edge_index_dict,
        edge_mask_dict={edge_type: mask},
    )
    loss = embedding.square().sum()
    loss.backward()
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()
