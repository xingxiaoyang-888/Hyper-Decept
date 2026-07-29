"""
HyperDecept-WB M1: Evidence Registry.

Loads observable evidence rows from SQLite databases and CSV files into
a unified in-memory store.  Label columns and hidden fields are never
registered as evidence.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import EvidenceRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table whitelist -- only these tables are scanned in register_db.
# ---------------------------------------------------------------------------
_ALLOWED_TABLES: set = {
    "user",
    "post",
    "follow",
    "like",
    "dislike",
    "comment",
    "comment_like",
    "comment_dislike",
    "trace",
    "agent_actions",
}

# ---------------------------------------------------------------------------
# Field whitelist for trace / agent_actions tables.
# Columns NOT in this set are skipped.
# ---------------------------------------------------------------------------
_TRACE_ALLOWED_FIELDS: set = {
    "user_id",
    "agent_name",
    "agent_id",
    "created_at",
    "timestamp",
    "ts",
    "action",
    "action_type",
    "info",
    "content",
    "post_id",
    "comment_id",
}

# ---------------------------------------------------------------------------
# Blocked field names (label leakage).  Also applies to CSV.
# ---------------------------------------------------------------------------
_BLOCKED_COLUMNS: set = {
    "user_type",
    "is_bad",
    "label",
    "ground_truth",
    "bot_label",
    "human_label",
    "split",
    "fold",
    "activity_level",
    "following_agentid_list",
    "strategy",
    "private_message",
    "task",
    "hidden_personality",
    "hidden_role",
}

# ---------------------------------------------------------------------------
# CSV allowed columns (explicit whitelist).  Everything else is skipped.
# ---------------------------------------------------------------------------
_CSV_ALLOWED_COLUMNS: set = {
    "user_id",
    "name",
    "username",
    "description",
    "bio",
    "previous_tweets",
    "following_list",
    "created_at",
    "user_char_excluded",  # placeholder -- user_char itself is blocked
}


def _is_csv_column_allowed(col_name: str) -> bool:
    """Return True if *col_name* should be registered from CSV."""
    lower = col_name.lower().strip()
    # blocked columns are never allowed
    if lower in _BLOCKED_COLUMNS:
        return False
    # user_char is also blocked (may contain hidden personality)
    if lower == "user_char":
        return False
    # explicitly allowed
    if lower in _CSV_ALLOWED_COLUMNS:
        return True
    # heuristic: observable text/biography fields
    if lower in ("description", "bio", "previous_tweets", "following_list",
                 "created_at", "user_char"):
        return lower != "user_char"  # double-check
    return False


# ---------------------------------------------------------------------------
# EvidenceRegistry
# ---------------------------------------------------------------------------

class EvidenceRegistry:
    """Thread-unsafe in-memory store of :class:`EvidenceRecord` rows."""

    def __init__(self) -> None:
        self._records: Dict[str, EvidenceRecord] = {}

    # -- registration -------------------------------------------------------

    def register_db(self, db_path: str) -> int:
        """Scan whitelisted tables in *db_path* and register observable rows.

        Returns the number of new records added.
        """
        db_path = str(Path(db_path).resolve())
        if not os.path.isfile(db_path):
            logger.warning("EvidenceRegistry: DB not found, skipping: %s", db_path)
            return 0

        added = 0
        conn = sqlite3.connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            tables = self._list_tables(conn)

            for table_name in tables:
                if table_name.lower() not in _ALLOWED_TABLES:
                    logger.debug("EvidenceRegistry: skipping unknown table %s", table_name)
                    continue
                columns = self._table_columns(conn, table_name)
                safe_columns = self._filter_columns(table_name, columns)
                if not safe_columns:
                    continue
                try:
                    added += self._register_table(conn, table_name, safe_columns)
                except Exception:
                    logger.warning(
                        "EvidenceRegistry: failed to read table %s, skipped.", table_name
                    )
        finally:
            conn.close()
        return added

    def register_csv(
        self,
        csv_path: str,
        user_id_column: str = "user_id",
        observable_aliases: Optional[Dict[str, str]] = None,
    ) -> int:
        """Register allowed observable columns from a CSV file.

        Uses an explicit whitelist; label and hidden columns are skipped
        even if they exist in the file. ``observable_aliases`` is an explicit
        dataset-contract mapping for ambiguous legacy names. For example,
        TwiBot's public description is stored as ``user_char`` and may be
        mapped to ``bio``; the default remains blocked.
        """
        csv_path = str(Path(csv_path).resolve())
        if not os.path.isfile(csv_path):
            logger.warning("EvidenceRegistry: CSV not found, skipping: %s", csv_path)
            return 0

        added = 0
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return 0

            aliases = {
                str(source): str(target)
                for source, target in (observable_aliases or {}).items()
            }
            allowed_fields = []
            for field in reader.fieldnames:
                source_name = field.lower().strip()
                target_name = aliases.get(field, field).lower().strip()
                if source_name in _BLOCKED_COLUMNS:
                    continue
                if _is_csv_column_allowed(target_name):
                    allowed_fields.append(field)

            for row_idx, row in enumerate(reader):
                user_id = str(row.get(user_id_column, "")).strip()
                if not user_id:
                    continue
                for field in allowed_fields:
                    value = row.get(field)
                    if value is None or str(value).strip() == "":
                        continue
                    normalized_field = aliases.get(field, field)
                    evidence_id = f"csv:{user_id}:{normalized_field}:{row_idx}"
                    record = EvidenceRecord(
                        evidence_id=evidence_id,
                        evidence_type="csv_field",
                        actor_id=user_id,
                        target_id=None,
                        content=str(value)[:2000],
                        timestamp=row.get("created_at"),
                        source_table="csv",
                        source_row_id=row_idx,
                        observed=True,
                        metadata={
                            "csv_path": csv_path,
                            "column": normalized_field,
                            "source_column": field,
                        },
                    )
                    self._records[evidence_id] = record
                    added += 1
        return added

    def register_texts(
        self,
        user_id: str,
        texts: List[str],
        post_ids: Optional[List[str]] = None,
    ) -> int:
        """Register per-user text evidence with stable ``text:{user_id}:{idx}`` ids.

        Parameters
        ----------
        user_id : str
            Owning user id.
        texts : list of str
            Public post/tweet texts.
        post_ids : list of str, optional
            If provided, maps 1:1 to *texts* as real post ids.  When absent
            the records use synthetic ``text:{user_id}:{idx}`` ids.

        Returns
        -------
        int : number of records added.
        """
        uid = str(user_id)
        added = 0
        for idx, text in enumerate(texts):
            if not text or not str(text).strip():
                continue
            if post_ids and idx < len(post_ids) and post_ids[idx]:
                evidence_id = f"post:{post_ids[idx]}"
            else:
                evidence_id = f"text:{uid}:{idx}"
            record = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type="text",
                actor_id=uid,
                target_id=None,
                content=str(text)[:2000],
                timestamp=None,
                source_table="text",
                source_row_id=idx,
                observed=True,
                metadata={"text_index": idx},
            )
            self._records[evidence_id] = record
            added += 1
        return added

    # -- query --------------------------------------------------------------

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """Return a single record by id, or *None*."""
        return self._records.get(str(evidence_id))

    def get_for_user(self, user_id: str) -> List[EvidenceRecord]:
        """Return all evidence records linked to *user_id*."""
        uid = str(user_id)
        results: List[EvidenceRecord] = []
        for rec in self._records.values():
            if rec.actor_id == uid or rec.target_id == uid:
                results.append(rec)
        return results

    def search(
        self,
        evidence_type: Optional[str] = None,
        actor_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> List[EvidenceRecord]:
        """Return records matching the supplied filters (all optional)."""
        results: List[EvidenceRecord] = []
        for rec in self._records.values():
            if evidence_type is not None and rec.evidence_type != evidence_type:
                continue
            if actor_id is not None and rec.actor_id != str(actor_id):
                continue
            if target_id is not None and rec.target_id != str(target_id):
                continue
            results.append(rec)
        return results

    def to_dict(self) -> dict:
        """Return all records as a dict keyed by evidence_id."""
        return {eid: rec.to_dict() for eid, rec in self._records.items()}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, evidence_id: str) -> bool:
        return str(evidence_id) in self._records

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _list_tables(conn: sqlite3.Connection) -> List[str]:
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            return [r[0] for r in rows if not str(r[0]).startswith("sqlite_")]
        except Exception:
            return []

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
        try:
            rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            return [r[1] for r in rows]
        except Exception:
            return []

    @staticmethod
    def _filter_columns(table_name: str, columns: List[str]) -> List[str]:
        """Return columns safe for evidence registration."""
        t = table_name.lower()
        if t in ("trace", "agent_actions"):
            return [
                c for c in columns
                if c.lower() in _TRACE_ALLOWED_FIELDS
                and c.lower() not in _BLOCKED_COLUMNS
            ]
        return [
            c for c in columns
            if c.lower() not in _BLOCKED_COLUMNS
        ]

    def _register_table(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        safe_columns: List[str],
    ) -> int:
        added = 0
        # Build column list with rowid for unique key
        col_str = ", ".join(f'"{c}"' for c in safe_columns)
        try:
            cursor = conn.execute(f"SELECT rowid, {col_str} FROM \"{table_name}\"")
        except Exception:
            return 0

        for row in cursor:
            row_dict = {safe_columns[i]: row[i + 1] for i in range(len(safe_columns))}
            rowid = row[0]
            record = self._row_to_record(table_name, rowid, row_dict)
            if record is not None:
                self._records[record.evidence_id] = record
                added += 1
        return added

    def _row_to_record(
        self,
        table_name: str,
        rowid: int,
        row: Dict[str, Any],
    ) -> Optional[EvidenceRecord]:
        """Map a DB row (dict) to an EvidenceRecord, or *None*."""
        t = table_name.lower()

        # -- actor_id -------------------------
        actor_id = None
        for col in ("user_id", "follower_id", "muter_id", "agent_name", "agent_id"):
            if col in row and row[col] is not None:
                actor_id = str(row[col])
                break

        # -- target_id ------------------------
        target_id = None
        for col in (
            "post_id", "followee_id", "mutee_id", "comment_id",
            "original_post_id", "product_id",
        ):
            if col in row and row[col] is not None:
                target_id = str(row[col])
                break

        # -- content --------------------------
        content = None
        for col in ("content", "info", "bio", "action"):
            if col in row and row[col] is not None:
                val = str(row[col])
                if val.strip():
                    content = val[:2000]
                    break

        # -- timestamp ------------------------
        timestamp = None
        for col in ("created_at", "timestamp", "ts"):
            if col in row and row[col] is not None:
                timestamp = str(row[col])
                break

        # -- build evidence_id using real primary keys --------------------
        evidence_id = self._build_evidence_id(table_name, row, rowid)

        # -- metadata -------------------------
        metadata: Dict[str, Any] = {
            "source_table": table_name,
            "source_rowid": rowid,
        }

        return EvidenceRecord(
            evidence_id=evidence_id,
            evidence_type=t,
            actor_id=actor_id,
            target_id=target_id,
            content=content,
            timestamp=timestamp,
            source_table=table_name,
            source_row_id=rowid,
            observed=True,
            metadata=metadata,
        )

    @staticmethod
    def _build_evidence_id(
        table_name: str,
        row: Dict[str, Any],
        rowid: int,
    ) -> str:
        """Build a stable string id from real primary keys, falling back to rowid."""
        t = table_name.lower()

        # Use real primary keys where available for stable, meaningful IDs
        if t == "user":
            uid = row.get("user_id", rowid)
            return f"user:{uid}"

        if t == "post":
            pid = row.get("post_id", rowid)
            return f"post:{pid}"

        if t == "follow":
            fid = row.get("follow_id", rowid)
            return f"follow:{fid}"

        if t == "like":
            lid = row.get("like_id", rowid)
            return f"like:{lid}"

        if t == "dislike":
            did = row.get("dislike_id", rowid)
            return f"dislike:{did}"

        if t == "comment":
            cid = row.get("comment_id", rowid)
            return f"comment:{cid}"

        if t == "comment_like":
            clid = row.get("comment_like_id", rowid)
            return f"comment_like:{clid}"

        if t == "comment_dislike":
            cdid = row.get("comment_dislike_id", rowid)
            return f"comment_dislike:{cdid}"

        if t in ("trace", "agent_actions"):
            a = row.get("user_id", row.get("agent_name", row.get("agent_id", "?")))
            return f"trace:{a}:{rowid}"

        # generic fallback (should not happen with table whitelist)
        return f"{t}:{rowid}"
