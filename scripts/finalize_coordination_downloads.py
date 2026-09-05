"""Finalize and audit coordination downloads once range jobs complete."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "raw_datasets" / "coordination"
LOG = ROOT / "runtime" / "download_logs" / "finalize.log"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def wait_for(path: Path, expected: int, timeout: int = 86400) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.is_file() and path.stat().st_size == expected:
            return
        time.sleep(30)
    raise TimeoutError(f"timed out waiting for {path}")


def main() -> None:
    crypto = DATA / "crypto_campaign" / "Bitcointalk_Bounties_Altcoins_Dataset.zip"
    wait_for(crypto, 7_457_722_579)
    log(f"crypto sha256={sha256(crypto)}")
    with zipfile.ZipFile(crypto) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"corrupt Crypto archive member: {bad}")
        destination = DATA / "crypto_campaign" / "extracted_full"
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
        log(f"crypto extracted members={len(archive.namelist())}")

    uk = DATA / "uk2019" / "tweet-ids.csv.zip"
    with zipfile.ZipFile(uk) as archive:
        if archive.testzip():
            raise RuntimeError("corrupt UK archive")
        destination = DATA / "uk2019" / "extracted_full"
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
        log(f"uk extracted members={len(archive.namelist())}")

    fake = DATA / "fake_accounts" / "tweets.csv"
    with fake.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    log(f"fake tweets bytes={fake.stat().st_size} rows={rows} columns={header}")

    fox = DATA / "fox8" / "fox8_23_dataset.ndjson.gz"
    with gzip.open(fox, "rt", encoding="utf-8") as f:
        first = f.readline()
    json.loads(first)
    log(f"fox8 bytes={fox.stat().st_size} first_json_valid=true")
    report = {
        "crypto": {"bytes": crypto.stat().st_size, "sha256": sha256(crypto)},
        "uk_tweet_ids": {"bytes": uk.stat().st_size, "sha256": sha256(uk)},
        "fake_tweets": {"bytes": fake.stat().st_size, "sha256": sha256(fake), "rows": rows, "columns": header},
        "fox8": {"bytes": fox.stat().st_size, "sha256": sha256(fox)},
    }
    (ROOT / "runtime" / "p2_formal_package" / "audits" / "coordination_full_download_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    log("FINALIZED")


if __name__ == "__main__":
    main()

