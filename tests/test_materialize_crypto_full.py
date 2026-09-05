import pandas as pd

from data_processing.materialize_crypto_full import materialize


def test_full_crypto_materializer_tracks_coverage_and_provenance(tmp_path):
    source = tmp_path / "source"
    labelled = source / "labeled"
    labelled.mkdir(parents=True)
    pd.DataFrame({
        "user_id": ["u1", "u2"], "bounty_participant": [True, False],
    }).to_csv(labelled / "users.tsv", sep="\t", index=False)
    pd.DataFrame({"thread_id": ["c1"], "comment_id": ["e1"], "user_id": ["u1"], "post_time ": ["2020-01-01"]}).to_csv(labelled / "events.tsv", sep="\t", index=False)
    rows = pd.DataFrame({
        "thread_id": ["c1", "c1"], "comment_id": ["a", "b"],
        "post_time": ["2020-01-01 01:00:00", "2020-01-01 02:00:00"],
        "user_id": ["u1", "u2"], "twitter_links": ["", "https://x"],
    })
    for filename in ("comments_participation.tsv", "comments_registration.tsv", "comments_other.tsv", "comments_author.tsv"):
        rows.to_csv(labelled / filename, sep="\t", index=False)
    out = tmp_path / "bundle"
    manifest = materialize(source, out, chunksize=1, edges_per_campaign=2)
    assert manifest["counts"]["labelled_activity_rows"] == 8
    assert manifest["coverage"]["known_user_activity"] == 1.0
    assert manifest["label_contract"]["not_available"] == ["bot_human", "cib_binary", "llm_origin"]
    assert pd.read_csv(out / "edges.csv")["evidence_id"].str.contains("comments_").all()

