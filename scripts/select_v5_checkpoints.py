"""Select formal v5 checkpoints using synthetic validation metrics only."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--honduras-freeze", type=Path, required=True)
    p.add_argument("--uae-freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    selected = []
    for source, manifest_path, target in (
        ("honduras", args.honduras_freeze, "uae"),
        ("uae", args.uae_freeze, "honduras"),
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = max(
            manifest["checkpoints"],
            key=lambda item: (item["best_validation_auprc"], -int(item["seed"]), item["held_out_scenario"]),
        )
        selected.append({**entry, "source_operation": source, "target_operation": target})
    result = {
        "schema_version": "hypertrace.v5-selected-checkpoints.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "maximum synthetic validation AUPRC; deterministic seed/scenario tie-break; no real external metric read",
        "real_external_metrics_consumed_for_selection": False,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
