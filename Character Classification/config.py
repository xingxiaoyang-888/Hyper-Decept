import ast
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PRESETS = {
    "agent72": (
        PROJECT_ROOT / "data" / "test_72.db",
        PROJECT_ROOT / "data" / "72agent_deeppersonal.csv",
    ),
    "test72": (
        PROJECT_ROOT / "data" / "test_72.db",
        PROJECT_ROOT / "data" / "72agent_deeppersonal.csv",
    ),
    "72": (
        PROJECT_ROOT / "data" / "test_72.db",
        PROJECT_ROOT / "data" / "72agent_deeppersonal.csv",
    ),
    "twibot120": (
        PROJECT_ROOT / "data" / "twibot_120_v5.db",
        PROJECT_ROOT / "data" / "twibot_120_multimodal_v5.csv",
    ),
    "twibot_120": (
        PROJECT_ROOT / "data" / "twibot_120_v5.db",
        PROJECT_ROOT / "data" / "twibot_120_multimodal_v5.csv",
    ),
    "twibot1000": (
        PROJECT_ROOT / "data" / "twibot_1000_v5.db",
        PROJECT_ROOT / "data" / "twibot_1000_multimodal_v5.csv",
    ),
    "twibot": (
        PROJECT_ROOT / "data" / "twibot_1000_v5.db",
        PROJECT_ROOT / "data" / "twibot_1000_multimodal_v5.csv",
    ),
    "twibot_1000": (
        PROJECT_ROOT / "data" / "twibot_1000_v5.db",
        PROJECT_ROOT / "data" / "twibot_1000_multimodal_v5.csv",
    ),
    "sim1000": (
        PROJECT_ROOT / "data" / "simu_db" / "test_1000_ver2.db",
        PROJECT_ROOT / "data" / "simu_db" / "test_1000_good_bad_random_bernoulli_.csv",
    ),
    "simulation": (
        PROJECT_ROOT / "data" / "simu_db" / "test_1000_ver2.db",
        PROJECT_ROOT / "data" / "simu_db" / "test_1000_good_bad_random_bernoulli_.csv",
    ),
    "sim": (
        PROJECT_ROOT / "data" / "simu_db" / "test_1000_ver2.db",
        PROJECT_ROOT / "data" / "simu_db" / "test_1000_good_bad_random_bernoulli_.csv",
    ),
}

DEFAULT_DATASET = "agent72"
DATASET_CHOICES = sorted(DATASET_PRESETS.keys())


def configure_utf8_streams() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def resolve_repo_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if candidate.is_absolute():
        return candidate
    repo_candidate = PROJECT_ROOT / candidate
    return repo_candidate if repo_candidate.exists() else candidate


def infer_dataset_key(db_path: Optional[str] = None, csv_path: Optional[str] = None) -> str:
    target = " ".join(str(p).lower() for p in (db_path, csv_path) if p)
    if "twibot_120" in target or "120_multimodal" in target:
        return "twibot120"
    if "twibot" in target:
        return "twibot1000"
    if "test_72" in target or "72agent" in target:
        return "agent72"
    if "test_1000_ver2" in target or "simu_db" in target:
        return "sim1000"
    return DEFAULT_DATASET


def resolve_dataset_paths(
    db_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Tuple[str, str]:
    db_path = os.getenv("AFG_DB_FILE", db_path)
    csv_path = os.getenv("AFG_CSV_FILE", csv_path)
    dataset = os.getenv("AFG_DATASET", dataset)

    dataset_key = (dataset or infer_dataset_key(db_path, csv_path)).lower()
    if dataset_key not in DATASET_PRESETS:
        dataset_key = DEFAULT_DATASET

    default_db, default_csv = DATASET_PRESETS[dataset_key]
    resolved_db = resolve_repo_path(db_path) if db_path else default_db
    resolved_csv = resolve_repo_path(csv_path) if csv_path else default_csv
    return str(resolved_db), str(resolved_csv)


def safe_name(value: Optional[str], default: str = "run") -> str:
    text = str(value or default).strip()
    cleaned = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    name = "".join(cleaned).strip("_")
    return name or default


def make_experiment_dir(
    base_dir: str,
    db_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    dataset: Optional[str] = None,
    run_name: Optional[str] = None,
    prefix: str = "run",
) -> str:
    dataset_key = safe_name(dataset or infer_dataset_key(db_path, csv_path))
    if run_name:
        folder_name = safe_name(run_name)
    else:
        db_stem = safe_name(Path(str(db_path)).stem if db_path else dataset_key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{prefix}_{dataset_key}_{db_stem}_{timestamp}"

    base = resolve_repo_path(base_dir) or Path(base_dir)
    run_dir = base / folder_name
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = base / f"{folder_name}_{counter}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def write_manifest(run_dir: str, **metadata) -> str:
    manifest_path = Path(run_dir) / "manifest.txt"
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(manifest_path)


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    column_set = set(columns)
    for name in candidates:
        if name in column_set:
            return name
    return None


def get_table_columns(conn: sqlite3.Connection) -> dict:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn,
    )["name"].tolist()
    return {
        table: [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
        for table in tables
    }


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_tweet_pool(value) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item).strip() for item in value if not is_missing(item))

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return " | ".join(
                    str(item).strip().replace(" | ", " ")
                    for item in parsed
                    if not is_missing(item) and str(item).strip()
                )
        except (ValueError, SyntaxError):
            pass
    return text


def split_tweet_pool(value, min_len: int = 0) -> List[str]:
    text = normalize_tweet_pool(value)
    if not text:
        return []
    parts = text.split(" | ") if " | " in text else [text]
    return [part.strip() for part in parts if len(part.strip()) > min_len]


def load_label_frame(csv_path: Optional[str], db_path: Optional[str] = None) -> pd.DataFrame:
    resolved_csv = resolve_repo_path(csv_path) if csv_path else None
    if resolved_csv and resolved_csv.exists():
        df = pd.read_csv(resolved_csv)
    else:
        raise FileNotFoundError(f"CSV label/text file not found: {csv_path}")

    if "user_id" not in df.columns:
        raise ValueError("CSV must contain a user_id column.")

    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)

    if "user_type" not in df.columns and "is_bad" in df.columns:
        df["user_type"] = df["is_bad"].map(lambda v: "bad" if int(v) == 1 else "good")
    if "is_bad" not in df.columns:
        if "user_type" not in df.columns:
            raise ValueError("CSV must contain user_type or is_bad.")
        df["is_bad"] = (
            df["user_type"].astype(str).str.lower().str.contains("bad|bot", regex=True).astype(int)
        )

    for column in ("name", "username", "description", "user_char", "previous_tweets"):
        if column not in df.columns:
            df[column] = ""

    if df["user_char"].fillna("").astype(str).str.strip().eq("").all():
        for fallback in ("description", "bio"):
            if fallback in df.columns:
                df["user_char"] = df[fallback].fillna("")
                break

    return df


def build_text_fields(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    bios: List[str] = []
    tweets_joined: List[str] = []
    summary_texts: List[str] = []

    for _, row in df.iterrows():
        bio = row.get("user_char", row.get("description", row.get("bio", "")))
        bio_text = "" if is_missing(bio) else str(bio)
        tweets = normalize_tweet_pool(row.get("previous_tweets", ""))
        bios.append(bio_text)
        tweets_joined.append(tweets)
        summary_texts.append(f"Bio: {bio_text}. Recent actions: {tweets}")

    return bios, tweets_joined, summary_texts


def label_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["user_id", "is_bad", "user_type", "name", "username"]
    return [column for column in preferred if column in df.columns]
