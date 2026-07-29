-- Read-only report transformation for the aggregate JSON emitted by
-- data_processing/mgtab_raw_audit.py. Values below are the reviewed aggregate
-- results from the 2026-07-29 MGTAB standard audit; no individual vectors are
-- included. This script is executable with SQLite and reproduces the bounded
-- report tables used by docs/mgtab_field_audit_artifact.json.

.mode json

CREATE TEMP TABLE audit_summary (
    node_count INTEGER,
    edge_count INTEGER,
    feature_dim INTEGER,
    high_risk_count INTEGER
);
INSERT INTO audit_summary VALUES (10199, 1700108, 788, 4);

CREATE TEMP TABLE relation_audit (
    id INTEGER,
    relation TEXT,
    direction TEXT,
    edge_rows INTEGER,
    duplicate_rows INTEGER,
    self_loops INTEGER,
    weight_range TEXT
);
INSERT INTO relation_audit VALUES
    (0, 'followers', 'directed', 308120, 0, 0, '1.0000–1.0000'),
    (1, 'friends', 'directed', 412575, 0, 0, '1.0000–1.0000'),
    (2, 'mention', 'directed', 114516, 0, 3648, '1.0000–1.0000'),
    (3, 'reply', 'directed', 223466, 197086, 159772, '1.0000–1.0000'),
    (4, 'quoted', 'directed', 77631, 54362, 42399, '1.0000–1.0000'),
    (5, 'url', 'undirected, stored once', 263800, 0, 0, '0.0253–1.0000'),
    (6, 'hashtag', 'undirected, stored once', 300000, 0, 0, '0.4577–1.0000');

CREATE TEMP TABLE label_audit (
    ordering INTEGER,
    task TEXT,
    value INTEGER,
    label TEXT,
    count INTEGER,
    share REAL
);
INSERT INTO label_audit VALUES
    (1, 'bot', 0, 'human', 7451, 0.7305618198),
    (2, 'bot', 1, 'bot', 2748, 0.2694381802),
    (3, 'stance', 0, 'neutral', 3776, 0.3702323757),
    (4, 'stance', 1, 'against', 3637, 0.3566035886),
    (5, 'stance', 2, 'support', 2786, 0.2731640357);

SELECT * FROM audit_summary;
SELECT * FROM relation_audit ORDER BY id;
SELECT * FROM label_audit ORDER BY ordering;
