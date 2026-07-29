"""
profile_chunker.py

Splits each profile in deeppersonal_agents.json into multiple semantic chunks,
flattening them into natural language text for subsequent embedding and vector retrieval.

Output: chunked_profiles.json (readable, used for inspecting chunk quality)
"""

import json
import os
import re
import argparse
from pathlib import Path

# Top-level sections to be treated as independent chunks
SECTION_MAPPING = {
    "demographic": "Demographic Information",
    "career": "Career and Work Identity",
    "values": "Core Values, Beliefs, and Philosophy",
    "lifestyle": "Lifestyle and Daily Routine",
    "social_context": "Cultural and Social Context",
    "interests": "Hobbies, Interests, and Lifestyle",
}

# Top-level fields in the profile that do not need to be chunked
SKIP_KEYS = {"Generated At", "Profile Index", "Summary"}


def section_slug(section_name: str) -> str:
    """Create stable metadata for official sections not known in advance."""
    return re.sub(r"[^a-z0-9]+", "_", section_name.lower()).strip("_")


def flatten_dict(d: dict, parent_key: str = "") -> list[tuple[str, str]]:
    NOISE_VALUES = {
        "none", "none;", "not applicable", "", "none mentioned",
        "not specified", "no", "not interested", "none; no",
        "none; not applicable",
    }

    items: list[tuple[str, str]] = []
    for k, v in d.items():
        full_key = f"{parent_key} → {k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, full_key))
        else:
            val_str = str(v).strip() if v is not None else ""
            if val_str.lower() not in NOISE_VALUES:
                items.append((full_key, val_str))
    return items


def build_section_text(section_name: str, section_data: dict) -> str:
    pairs = flatten_dict(section_data)
    if not pairs:
        return ""

    lines = [f"[{section_name}]"]
    for key, val in pairs:
        clean_key = key.replace("_", " ")
        lines.append(f"{clean_key}: {val}")
    return "\n".join(lines)


def chunk_profile(profile: dict, agent_id: int) -> list[dict]:
    """Splits a profile into multiple chunk dicts."""
    chunks: list[dict] = []

    # 1. Summary block (kept as is, this is the most valuable natural language personality description)
    summary = profile.get("Summary", "")
    if summary and summary.strip():
        chunks.append({
            "agent_id": agent_id,
            "section": "summary",
            "text": summary.strip(),
        })

    # 2. Known semantic blocks
    processed_keys = set()
    for section_key, json_key in SECTION_MAPPING.items():
        section_data = profile.get(json_key)
        if isinstance(section_data, dict):
            processed_keys.add(json_key)
            text = build_section_text(section_key, section_data)
            if text.strip():
                chunks.append({
                    "agent_id": agent_id,
                    "section": section_key,
                    "text": text,
                })

    # 3. Preserve future/variant DeepPersona sections instead of silently
    # dropping them when the official taxonomy evolves.
    for json_key, section_data in profile.items():
        if (
            json_key in SKIP_KEYS
            or json_key in processed_keys
            or not isinstance(section_data, dict)
        ):
            continue
        section_key = section_slug(json_key)
        text = build_section_text(section_key, section_data)
        if text.strip():
            chunks.append({
                "agent_id": agent_id,
                "section": section_key,
                "text": text,
            })

    return chunks


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=script_dir / "deeppersonal_agents.json"
    )
    parser.add_argument(
        "--output", type=Path, default=script_dir / "chunked_profiles.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    print(f"Read {len(profiles)} profiles")

    # Chunk one by one
    all_chunks: list[dict] = []
    for profile in profiles:
        idx = profile.get("Profile Index", 1) - 1  # Convert to 0-based agent_id
        chunks = chunk_profile(profile, idx)
        all_chunks.extend(chunks)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Print statistics
    print(f"\n[OK] Chunking completed: {len(profiles)} profiles -> {len(all_chunks)} chunks")
    print(f"  Output file: {output_path}")

    sections = sorted(set(c["section"] for c in all_chunks))
    print(f"  Chunk types: {', '.join(sections)}")

    print("\nChunk details:")
    for c in all_chunks:
        char_count = len(c["text"])
        line_count = c["text"].count("\n") + 1
        print(f"  [agent_{c['agent_id']}] {c['section']:15s}  {char_count:5d} chars, {line_count:2d} lines")
        if char_count == 0:
            print("    [WARN] Empty chunk, please check source data")


if __name__ == "__main__":
    main()
