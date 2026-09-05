import pandas as pd

from data_processing.coordination_adapter import (
    CoordinationCapabilities,
    load_fake_accounts,
    load_uk2019,
)


def test_uk_adapter_does_not_claim_text_or_llm(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    pd.DataFrame({"id": ["u1"], "cluster": [3]}).to_csv(
        extracted / "superspreader-nodes.csv", index=False
    )
    pd.DataFrame({"source": ["u1"], "target": ["u2"], "weight": [1.0]}).to_csv(
        extracted / "superspreader-edges.csv", index=False
    )
    bundle = load_uk2019(tmp_path)
    assert bundle.capabilities.text is False
    assert bundle.capabilities.llm_origin_labels is False
    assert bundle.capabilities.explicit_campaign_labels is False
    assert bundle.capabilities.explicit_cib_labels is False
    assert bundle.labels.loc[0, "label_task"] == "topology_cluster"
    assert bundle.labels.loc[0, "label_value"] == 3
    assert pd.isna(bundle.labels.loc[0, "campaign_id"])


def test_crypto_adapter_derives_user_user_participation_edges(tmp_path):
    sample = tmp_path / "extracted" / "samples"
    sample.mkdir(parents=True)
    pd.DataFrame({"user_id": ["u1", "u2"], "posts": [1, 2], "activity": [1, 2], "bounty_participant": [True, False]}).to_csv(sample / "users.tsv", sep="\t", index=False)
    pd.DataFrame({"thread_id": ["t1", "t1"], "comment_id": ["c1", "c2"], "user_id": ["u1", "u2"], "post_time": ["2020-01-01", "2020-01-01"], "post_html": ["a", "b"]}).to_csv(sample / "comments_raw.tsv", sep="\t", index=False)
    pd.DataFrame({"thread_id": [], "comment_id": [], "user_id": [], "post_time ": [], "post_tex": []}).to_csv(sample / "events.tsv", sep="\t", index=False)
    bundle = __import__("data_processing.coordination_adapter", fromlist=["load_crypto_campaign"]).load_crypto_campaign(tmp_path)
    assert len(bundle.edges) == 1
    assert bundle.edges.loc[0, "edge_type"] == "co_participates_campaign"


def test_fake_accounts_blank_cib_is_unknown(tmp_path):
    pd.DataFrame({"user.id_str": ["u1", "u2"], "cib": ["campaign", None]}).to_csv(
        tmp_path / "users_ids.csv", index=False
    )
    bundle = load_fake_accounts(tmp_path)
    assert bundle.labels["is_known"].tolist() == [True, False]
    assert bundle.capabilities.llm_origin_labels is False
