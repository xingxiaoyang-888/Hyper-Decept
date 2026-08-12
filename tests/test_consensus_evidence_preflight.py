import numpy as np
import torch

from scripts.generate_consensus_evidence_preflight import _percentile, _to_numpy


def test_to_numpy_detaches_trainable_inference_output() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    output = parameter * 2

    converted = _to_numpy(output)

    np.testing.assert_array_equal(converted, np.array([2.0, 4.0], dtype=np.float32))
    assert converted.flags.writeable


def test_percentile_uses_the_frozen_reference_universe() -> None:
    all_scores = np.array([0.1, 0.2, 0.3, 0.99])
    all_ids = np.array(["a", "b", "c", "outside"])
    reference_mask = np.isin(all_ids, ["a", "b", "c"])
    reference_scores = all_scores[reference_mask]
    reference_ids = all_ids[reference_mask]

    assert _percentile(reference_scores, reference_ids, 2) == 1.0
