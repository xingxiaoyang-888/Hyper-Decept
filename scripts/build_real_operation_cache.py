"""Materialize one cutoff-safe real-operation graph cache exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrain_real_operation_graph import (  # noqa: E402
    load_training_graph,
    save_graph_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cutoff-quantile", type=float, default=0.8)
    parser.add_argument("--max-train-edges", type=int, default=200_000)
    parser.add_argument("--max-validation-edges", type=int, default=20_000)
    args = parser.parse_args()
    import torch
    graph, future_edges, audit = load_training_graph(
        args.bundle,
        cutoff_quantile=args.cutoff_quantile,
        device=torch.device(args.device),
        max_train_edges=args.max_train_edges,
        max_validation_edges=args.max_validation_edges,
        seed=args.seed,
    )
    result = save_graph_cache(
        args.cache_dir,
        graph=graph,
        future_edges=future_edges,
        audit=audit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
