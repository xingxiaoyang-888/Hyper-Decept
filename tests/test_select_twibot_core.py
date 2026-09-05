import csv

import pytest

from data_processing.select_twibot_core import select_core_users


def _write(path, field, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", field])
        writer.writeheader()
        writer.writerows({"id": user_id, field: value} for user_id, value in rows)


def test_selection_is_deterministic_and_jointly_stratified(tmp_path):
    labels = tmp_path / "label.csv"
    splits = tmp_path / "split.csv"
    label_rows = []
    split_rows = []
    for split in ("train", "validation", "test"):
        for label in ("bot", "human"):
            for index in range(10):
                user_id = f"u-{split}-{label}-{index}"
                label_rows.append((user_id, label))
                split_rows.append((user_id, split))
    _write(labels, "label", reversed(label_rows))
    _write(splits, "split", split_rows)

    first, report = select_core_users(
        label_path=labels, split_path=splits, count=24, seed=7,
    )
    second, _ = select_core_users(
        label_path=labels, split_path=splits, count=24, seed=7,
    )

    assert first == second
    assert len(first) == len(set(first)) == 24
    assert {value["selected"] for value in report["strata"].values()} == {4}


def test_selection_changes_with_seed(tmp_path):
    labels = tmp_path / "label.csv"
    splits = tmp_path / "split.csv"
    rows = [(f"u{index}", "bot" if index % 2 else "human") for index in range(40)]
    _write(labels, "label", rows)
    _write(splits, "split", [(user_id, "train") for user_id, _ in rows])

    first, _ = select_core_users(
        label_path=labels, split_path=splits, count=10, seed=1,
    )
    second, _ = select_core_users(
        label_path=labels, split_path=splits, count=10, seed=2,
    )

    assert set(first) != set(second)


def test_selection_rejects_oversized_request(tmp_path):
    labels = tmp_path / "label.csv"
    splits = tmp_path / "split.csv"
    _write(labels, "label", [("u1", "bot")])
    _write(splits, "split", [("u1", "train")])

    with pytest.raises(ValueError, match="only 1 are eligible"):
        select_core_users(label_path=labels, split_path=splits, count=2, seed=1)
