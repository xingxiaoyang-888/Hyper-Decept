"""Create evidence-grounded persona records for TwiBot simulation agents.

This does not invent demographics.  It packages public biography, account
metrics, network degree, and sampled posts into the same structured profile
interface consumed by the DeepPersona chunking/RAG layer.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = []
    with args.csv.expanduser().resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            agent_id = int(row["user_id"])
            bio = str(row.get("user_char") or row.get("description") or "").strip()
            posts = ast.literal_eval(row.get("previous_tweets") or "[]")
            evidence_summary = bio
            if posts:
                evidence_summary += "\nObserved recent public posts: " + " | ".join(posts)
            profiles.append(
                {
                    "Profile Index": agent_id + 1,
                    "Demographic Information": {
                        "Public self-description": bio,
                    },
                    "Cultural and Social Context": {
                        "Source account ID": row.get("source_user_id", ""),
                        "Public username": row.get("username", ""),
                    },
                    "Observed Social Media Behavior": {
                        "Followers": int(row.get("followers_count") or 0),
                        "Following": int(row.get("following_count") or 0),
                        "Preserved internal follows": len(
                            ast.literal_eval(row.get("following_agentid_list") or "[]")
                        ),
                        "Recent public posts": posts,
                    },
                    "Summary": evidence_summary,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(profiles)} evidence-grounded profiles: {args.output}")


if __name__ == "__main__":
    main()
