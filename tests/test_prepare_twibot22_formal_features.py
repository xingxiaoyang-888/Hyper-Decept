import pandas as pd

from data_processing.prepare_twibot22_formal_features import (
    BEHAVIOR_COLUMNS,
    build_behavior_features,
    select_posts,
)


def test_select_posts_keeps_latest_cap_in_chronological_order():
    posts = pd.DataFrame({
        "author_id": ["u1", "u1", "u1", "u2"],
        "post_id": ["p2", "p1", "p3", "p4"],
        "content": ["two", "one", "three", "four"],
        "created_at": ["2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
    })
    selected = select_posts(posts, ["u1", "u2"], max_tweets_per_user=2)
    assert selected[selected.author_id == "u1"].post_id.tolist() == ["p2", "p3"]
    assert selected[selected.author_id == "u2"].post_id.tolist() == ["p4"]


def test_behavior_features_have_exact_contract_and_observed_values():
    core = pd.DataFrame({
        "user_id": ["u1", "u2"], "followers": [10, 0], "following": [4, 0],
    })
    actions = pd.DataFrame({
        "actor_id": ["u1", "u1", "u1"],
        "action_type": ["like", "retweet", "reply"],
        "event_time": ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", None],
    })
    posts = pd.DataFrame({
        "author_id": ["u1", "u1"],
        "content": ["https://example.test @a #b pic.twitter.com/x", "plain"],
    })
    result = build_behavior_features(core, actions, posts)
    assert result.columns.tolist() == BEHAVIOR_COLUMNS
    assert result.loc[0, "Follower_Following_Ratio"] == 2.0
    assert result.loc[0, "Action_Frequency"] == 3
    assert result.loc[0, "Like_Ratio"] == 1 / 3
    assert result.loc[0, "URL_Ratio"] == 0.5
    assert result.loc[1].eq(0).all()
