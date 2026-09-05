import pandas as pd
import torch

from scripts.pretrain_lorentz_graph import run_pretraining


def test_lorentz_pretraining_writes_checkpoint_and_resumes(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    pd.DataFrame({"user_id": ["u0", "u1", "u2"], **{
        f"Semantic_{i}": [float(i), float(i + 1), float(i + 2)] for i in range(8)
    }, "Follower_Following_Ratio": [1, 2, 3], "Action_Frequency": [1, 1, 2], **{
        name: [0.0, 0.1, 0.2] for name in ["Like_Ratio", "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio", "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio", "Empathy_Gap_Mean", "Empathy_Gap_Max", "Dark_Triad_Mean", "Dark_Triad_Max", "Contagion_Mean", "Contagion_Max", "Volatility_Mean", "Volatility_Max"]
    }}).to_csv(bundle / "features_26d.csv", index=False)
    pd.DataFrame({"node_id": ["u0", "u1", "u2"]}).to_csv(bundle / "nodes.csv", index=False)
    pd.DataFrame({"source_id": ["u0", "u1"], "target_id": ["u1", "u2"]}).to_csv(bundle / "edges.csv", index=False)
    feature_names = pd.read_csv(bundle / "features_26d.csv", nrows=0).columns.drop("user_id")
    availability = pd.DataFrame({"node_id": ["u0", "u1", "u2"]})
    for name in feature_names:
        availability[name] = name == "Action_Frequency"
    availability.to_csv(bundle / "feature_availability.csv", index=False)
    out = tmp_path / "out"
    result = run_pretraining(bundle_root=bundle, output_dir=out, device="cpu", max_steps=1, hidden_dim=8, num_heads=2)
    assert result["status"] == "passed"
    assert result["available_feature_values"] == 3
    assert (out / "last_checkpoint.pt").is_file()
    resumed = run_pretraining(
        bundle_root=bundle,
        output_dir=out,
        device="cpu",
        epochs=2,
        max_steps=2,
        hidden_dim=8,
        num_heads=2,
        resume=True,
    )
    assert resumed["resumed"] is True
    assert len(resumed["history"]) == 2
