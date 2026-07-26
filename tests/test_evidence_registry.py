"""
Tests for explainability.evidence_registry.

Uses small in-memory / temp SQLite databases -- no external models needed.
"""

import os
import sqlite3
import tempfile
import csv

import pytest

from explainability.evidence_registry import EvidenceRegistry


# ------------------------------------------------------------------ helpers

def _make_temp_db(db_path: str, tables_spec: dict):
    """Create a SQLite file at *db_path* with the given table definitions.

    *tables_spec* maps table name -> list of (col_name, col_type) tuples.
    """
    conn = sqlite3.connect(db_path)
    for table, columns in tables_spec.items():
        col_defs = ", ".join(f"{c} {t}" for c, t in columns)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})")
    conn.commit()
    conn.close()


def _insert(conn, table: str, rows: list):
    """Insert rows into *table*.  Each row is a tuple of values."""
    placeholders = ", ".join("?" for _ in rows[0])
    conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
    conn.commit()


# ------------------------------------------------------------------ tests


class TestEvidenceRegistryBasic:
    def test_empty_registry(self):
        reg = EvidenceRegistry()
        assert len(reg) == 0
        assert reg.get("nonexistent") is None

    def test_register_minimal_db(self):
        """Basic DB with user + post tables."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT"), ("bio", "TEXT")],
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER"),
                     ("content", "TEXT"), ("created_at", "DATETIME")],
        })

        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "alice", "bio1"), (2, "bob", "bio2")])
        _insert(conn, "post", [(101, 1, "hello", "2025-01-01"),
                                (102, 2, "world", "2025-01-02")])
        conn.close()

        reg = EvidenceRegistry()
        added = reg.register_db(db_path)
        assert added > 0
        assert len(reg) > 0

    def test_user_evidence(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "alice"), (2, "bob")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        rec = reg.get("user:1")
        assert rec is not None
        assert rec.evidence_type == "user"
        assert rec.actor_id == "1"

    def test_follow_evidence(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "follow": [("follow_id", "INTEGER"), ("follower_id", "INTEGER"),
                       ("followee_id", "INTEGER"), ("created_at", "DATETIME")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "follow", [(1, 1, 2, "2025-01-01")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        rec = reg.get("follow:1")
        assert rec is not None
        assert rec.evidence_type == "follow"
        assert rec.actor_id == "1"
        assert rec.target_id == "2"

    def test_get_for_user(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER"), ("content", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "alice"), (2, "bob")])
        _insert(conn, "post", [(101, 1, "a"), (102, 1, "b"), (103, 2, "c")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        alice_ev = reg.get_for_user("1")
        assert len(alice_ev) >= 3  # user row + 2 posts

    def test_search(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER"), ("content", "TEXT")],
            "like": [("like_id", "INTEGER"), ("user_id", "INTEGER"), ("post_id", "INTEGER")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "post", [(101, 1, "hello")])
        _insert(conn, "like", [(1, 2, 101)])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        posts = reg.search(evidence_type="post")
        assert all(e.evidence_type == "post" for e in posts)

        by_actor = reg.search(actor_id="2")
        assert all(e.actor_id == "2" for e in by_actor)


class TestMissingTableTolerance:
    def test_nonexistent_db_does_not_crash(self):
        reg = EvidenceRegistry()
        n = reg.register_db("/nonexistent/path/to/db.sqlite")
        assert n == 0

    def test_missing_table_does_not_crash(self):
        """Tables referenced in code but absent in DB are simply skipped."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        # Only create 'user' table -- no post, follow, like, comment, trace.
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "alice")])
        conn.close()

        reg = EvidenceRegistry()
        n = reg.register_db(db_path)
        assert n > 0  # user row registered

    def test_missing_column_does_not_crash(self):
        """A table exists but some expected columns are absent."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER")],
            # content column intentionally missing
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "post", [(101, 1)])
        conn.close()

        reg = EvidenceRegistry()
        n = reg.register_db(db_path)
        assert n >= 1  # should still register without crashing

    def test_no_like_comment_trace_tables(self):
        """Simulate a DB with only user + post, no like/comment/trace."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER"), ("content", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "a")])
        _insert(conn, "post", [(101, 1, "hello")])
        conn.close()

        reg = EvidenceRegistry()
        n = reg.register_db(db_path)
        # Should register user + post rows without error
        assert n >= 2


class TestLabelExclusion:
    def test_user_type_not_registered(self):
        """user_type column must NOT appear in evidence content."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT"),
                     ("user_type", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "alice", "bad")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        rec = reg.get("user:1")
        assert rec is not None
        # user_type column value must NOT be evidence content
        if rec.content is not None:
            assert "bad" not in rec.content.lower()

    def test_is_bad_not_registered(self):
        """is_bad column must not appear."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("is_bad", "INTEGER")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, 1)])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        rec = reg.get("user:1")
        assert rec is not None
        # is_bad value (1) should not be exposed as content
        assert rec.content is None or "1" not in str(rec.content)


class TestCSVRegistration:
    def test_register_csv(self):
        csv_path = os.path.join(tempfile.mkdtemp(), "test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["user_id", "user_char", "previous_tweets", "user_type", "is_bad"])
            writer.writerow(["1", "A friendly person", "tweet1 | tweet2", "good", "0"])
            writer.writerow(["2", "Another person", "tweet3", "bad", "1"])

        reg = EvidenceRegistry()
        n = reg.register_csv(csv_path)
        assert n > 0

        # user_type and is_bad columns must NOT be registered
        for rec in reg._records.values():
            meta_col = rec.metadata.get("column", "")
            assert meta_col not in ("user_type", "is_bad")

    def test_csv_observable_fields(self):
        """Only observable fields (user_char, previous_tweets, etc.) are registered."""
        csv_path = os.path.join(tempfile.mkdtemp(), "test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["user_id", "description", "user_type"])
            writer.writerow(["1", "test desc", "bad"])

        reg = EvidenceRegistry()
        n = reg.register_csv(csv_path)
        assert n >= 1
        # user_type should NOT be in any evidence record metadata column
        for rec in reg._records.values():
            assert rec.metadata.get("column") != "user_type"


class TestIdConsistency:
    def test_all_ids_are_strings(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER"), ("content", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "a")])
        _insert(conn, "post", [(101, 1, "x")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)

        for eid, rec in reg._records.items():
            assert isinstance(eid, str)
            assert isinstance(rec.evidence_id, str)
            if rec.actor_id is not None:
                assert isinstance(rec.actor_id, str)
            if rec.target_id is not None:
                assert isinstance(rec.target_id, str)


class TestToDict:
    def test_to_dict(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "alice")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        d = reg.to_dict()
        assert "user:1" in d
        assert d["user:1"]["evidence_type"] == "user"


class TestUnknownTableExclusion:
    """Tables not in the whitelist must be skipped entirely."""

    def test_product_table_not_registered(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
            "product": [("product_id", "INTEGER"), ("product_name", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "user", [(1, "a")])
        _insert(conn, "product", [(1, "widget")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        # product table must NOT appear
        for rec in reg._records.values():
            assert rec.source_table != "product"

    def test_unknown_table_skipped(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "secret_plans": [("plan_id", "INTEGER"), ("detail", "TEXT")],
            "user": [("user_id", "INTEGER"), ("user_name", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "secret_plans", [(1, "top secret")])
        _insert(conn, "user", [(1, "alice")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        # Only user table should be registered
        for rec in reg._records.values():
            assert rec.source_table != "secret_plans"
            assert rec.source_table == "user"


class TestTraceFieldWhitelist:
    """trace/agent_actions tables only expose allowed fields."""

    def test_hidden_trace_fields_not_registered(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "trace": [("user_id", "INTEGER"), ("created_at", "DATETIME"),
                      ("action", "TEXT"), ("info", "TEXT"),
                      ("strategy", "TEXT"), ("private_message", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "trace", [(1, "2025-01-01", "post", "public info",
                                  "secret_plan", "private msg")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        for rec in reg._records.values():
            # strategy and private_message must NOT appear in content or metadata
            content = rec.content or ""
            assert "secret_plan" not in content
            assert "private msg" not in content
            # metadata should NOT contain strategy/private_message keys
            for k in rec.metadata:
                assert k not in ("strategy", "private_message")


class TestCSVUserCharExclusion:
    """user_char column must not be registered as CSV evidence."""

    def test_user_char_excluded_from_csv(self):
        csv_path = os.path.join(tempfile.mkdtemp(), "test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["user_id", "user_char", "description", "previous_tweets"])
            writer.writerow(["1", "hidden personality traits here", "public bio", "tweet1"])

        reg = EvidenceRegistry()
        reg.register_csv(csv_path)
        for rec in reg._records.values():
            content = rec.content or ""
            assert "hidden personality" not in content.lower()

    def test_description_still_registered(self):
        csv_path = os.path.join(tempfile.mkdtemp(), "test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["user_id", "description"])
            writer.writerow(["1", "public bio text"])

        reg = EvidenceRegistry()
        n = reg.register_csv(csv_path)
        assert n >= 1
        assert any("public bio text" in (rec.content or "") for rec in reg._records.values())


class TestDuplicateActions:
    """Identical actions (e.g. same user liking same post twice) must each
    produce a distinct evidence record keyed by their unique rowid."""

    def test_duplicate_likes_preserved(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "like": [("like_id", "INTEGER"), ("user_id", "INTEGER"),
                     ("post_id", "INTEGER"), ("created_at", "DATETIME")],
        })
        conn = sqlite3.connect(db_path)
        # Two likes from user 1 on post 101
        _insert(conn, "like", [(1, 1, 101, "2025-01-01"),
                                (2, 1, 101, "2025-01-02")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        # Both should exist as distinct records
        r1 = reg.get("like:1")
        r2 = reg.get("like:2")
        assert r1 is not None, "First like record should exist"
        assert r2 is not None, "Second like record should exist"
        assert r1.evidence_id != r2.evidence_id

    def test_duplicate_posts_preserved(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "post": [("post_id", "INTEGER"), ("user_id", "INTEGER"), ("content", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "post", [(101, 1, "first"), (102, 1, "second")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        assert reg.get("post:101") is not None
        assert reg.get("post:102") is not None


class TestRegisterTexts:
    """register_texts creates stable text:{uid}:{idx} evidence records."""

    def test_register_texts_basic(self):
        reg = EvidenceRegistry()
        n = reg.register_texts("42", ["hello world", "another tweet"])
        assert n == 2
        r = reg.get("text:42:0")
        assert r is not None
        assert r.content == "hello world"
        assert r.actor_id == "42"
        assert r.evidence_type == "text"

    def test_register_texts_with_post_ids(self):
        reg = EvidenceRegistry()
        n = reg.register_texts("42", ["tweet a", "tweet b"],
                               post_ids=["101", "102"])
        assert n == 2
        assert reg.get("post:101") is not None
        assert reg.get("post:102") is not None

    def test_register_texts_skips_empty(self):
        reg = EvidenceRegistry()
        n = reg.register_texts("42", ["hello", "", "  ", "world"])
        assert n == 2  # only non-empty strings counted


class TestAgentActionsContent:
    """agent_actions with content column must capture observable text."""

    def test_agent_actions_content_registered(self):
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "agent_actions": [("agent_name", "TEXT"), ("action_type", "TEXT"),
                              ("content", "TEXT"), ("strategy", "TEXT"),
                              ("task_id", "TEXT"), ("private_message", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "agent_actions", [
            ("agent_1", "post", "public tweet text", "hidden_strategy", "task_001", "secret")
        ])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        # Find the agent_actions record
        recs = reg.search(evidence_type="agent_actions")
        assert len(recs) >= 1
        rec = recs[0]
        # content must be captured
        assert rec.content is not None
        assert "public tweet text" in rec.content
        # Hidden fields must NOT appear
        content = rec.content or ""
        assert "hidden_strategy" not in content
        assert "task_001" not in content
        assert "secret" not in content

    def test_trace_info_still_registered(self):
        """trace.info should still be in content."""
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        _make_temp_db(db_path, {
            "trace": [("user_id", "INTEGER"), ("action", "TEXT"), ("info", "TEXT")],
        })
        conn = sqlite3.connect(db_path)
        _insert(conn, "trace", [(1, "post", "public action info")])
        conn.close()

        reg = EvidenceRegistry()
        reg.register_db(db_path)
        recs = reg.search(evidence_type="trace")
        assert len(recs) >= 1
        assert "public action info" in (recs[0].content or "")
