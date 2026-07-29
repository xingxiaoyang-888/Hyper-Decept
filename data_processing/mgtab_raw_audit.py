"""Read-only field and integrity audit for the standard MGTAB tensor bundle.

The official release is a compact, anonymous graph representation rather than
the original Twitter records.  This module profiles the released tensors,
validates their alignment, and records modeling risks without modifying the
dataset or exporting individual feature vectors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


AUDIT_VERSION = 1
CHUNK_SIZE = 1024 * 1024
REQUIRED_FILES = (
    "edge_index.pt",
    "edge_type.pt",
    "edge_weight.pt",
    "features.pt",
    "labels_bot.pt",
    "labels_stance.pt",
)

RELATION_TYPES = {
    0: "followers",
    1: "friends",
    2: "mention",
    3: "reply",
    4: "quoted",
    5: "url",
    6: "hashtag",
}
EXPECTED_RELATION_COUNTS = {
    0: 308_120,
    1: 412_575,
    2: 114_516,
    3: 223_466,
    4: 77_631,
    5: 263_800,
    6: 300_000,
}
BOT_LABELS = {0: "human", 1: "bot"}
STANCE_LABELS = {0: "neutral", 1: "against", 2: "support"}

# Table 4 of the official paper defines dimensions 1-20 in this order.
PROPERTY_FEATURES = (
    ("profile_use_background_image", "boolean"),
    ("default_profile", "boolean"),
    ("verified", "boolean"),
    ("followers_count", "numerical"),
    ("default_profile_image", "boolean"),
    ("listed_count", "numerical"),
    ("statuses_count", "numerical"),
    ("friends_count", "numerical"),
    ("geo_enabled", "boolean"),
    ("favourites_count", "numerical"),
    ("created_at", "numerical"),
    ("screen_name_length", "numerical"),
    ("name_length", "numerical"),
    ("description_length", "numerical"),
    ("followers_friends_ratios", "numerical"),
    ("default_profile_background_color", "boolean"),
    ("default_profile_sidebar_fill_color", "boolean"),
    ("default_profile_sidebar_border_color", "boolean"),
    ("has_URL", "boolean"),
    ("profile_background_image_URL", "boolean"),
)

EXPECTED_NODES = 10_199
EXPECTED_FEATURES = 788
EXPECTED_EDGES = sum(EXPECTED_RELATION_COUNTS.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: torch.Tensor | int | float) -> int | float:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, int):
        return int(value)
    return float(value)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu").flatten()
    if values.numel() == 0:
        return {}
    probabilities = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], dtype=torch.float64
    )
    result = torch.quantile(values, probabilities)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p25", "median", "p75", "p90", "p99", "max"),
            result.tolist(),
        )
    }


def _distribution(values: torch.Tensor, names: dict[int, str]) -> dict[str, dict]:
    unique, counts = torch.unique(values, return_counts=True)
    total = int(values.numel())
    result = {}
    for value, count in zip(unique.tolist(), counts.tolist()):
        numeric = int(value)
        result[str(numeric)] = {
            "name": names.get(numeric, "unknown"),
            "count": int(count),
            "rate": float(count / total) if total else 0.0,
        }
    return result


def _tensor_profile(tensor: torch.Tensor) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "storage_bytes": int(tensor.numel() * tensor.element_size()),
    }
    if tensor.numel() == 0:
        return profile
    if tensor.is_floating_point():
        finite = torch.isfinite(tensor)
        profile.update({
            "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()),
            "nan_count": int(torch.isnan(tensor).sum()),
            "inf_count": int(torch.isinf(tensor).sum()),
            "zero_count": int((tensor == 0).sum()),
        })
        finite_values = tensor[finite]
        if finite_values.numel():
            profile.update({
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
                "std": float(finite_values.std(unbiased=False)),
            })
    else:
        profile.update({
            "min": int(tensor.min()),
            "max": int(tensor.max()),
            "zero_count": int((tensor == 0).sum()),
        })
    return profile


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(value)!r}")
    return value.detach().cpu()


def _finding(
    severity: str,
    code: str,
    title: str,
    evidence: dict[str, Any],
    risk: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "title": title,
        "evidence": evidence,
        "risk": risk,
        "recommendation": recommendation,
    }


def _relation_profile(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    relation_id: int,
    node_count: int,
) -> dict[str, Any]:
    mask = edge_type == relation_id
    source = edge_index[0, mask]
    target = edge_index[1, mask]
    weights = edge_weight[mask]
    keys = source.to(torch.int64) * node_count + target.to(torch.int64)
    _, multiplicities = torch.unique(keys, return_counts=True)
    unique_nodes = torch.unique(torch.cat((source, target)))
    profile = {
        "id": relation_id,
        "name": RELATION_TYPES[relation_id],
        "declared_direction": (
            "undirected" if relation_id in {5, 6} else "directed"
        ),
        "edge_rows": int(mask.sum()),
        "expected_edge_rows": EXPECTED_RELATION_COUNTS[relation_id],
        "unique_directed_pairs": int(multiplicities.numel()),
        "duplicate_edge_rows": int(multiplicities.sum() - multiplicities.numel()),
        "duplicated_pair_count": int((multiplicities > 1).sum()),
        "max_pair_multiplicity": int(multiplicities.max()) if multiplicities.numel() else 0,
        "self_loop_rows": int((source == target).sum()),
        "incident_node_count": int(unique_nodes.numel()),
        "weight": {
            "min": float(weights.min()) if weights.numel() else None,
            "max": float(weights.max()) if weights.numel() else None,
            "mean": float(weights.mean()) if weights.numel() else None,
            "unique_count": int(torch.unique(weights).numel()),
            "nonpositive_count": int((weights <= 0).sum()),
            "nonfinite_count": int((~torch.isfinite(weights)).sum()),
        },
    }
    if relation_id in {5, 6} and keys.numel():
        reverse_keys = target.to(torch.int64) * node_count + source.to(torch.int64)
        sorted_keys = torch.sort(keys).values
        positions = torch.searchsorted(sorted_keys, reverse_keys)
        bounded = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
        has_reverse = (positions < sorted_keys.numel()) & (
            sorted_keys[bounded] == reverse_keys
        )
        profile["reverse_row_coverage"] = {
            "rows_with_reverse_counterpart": int(has_reverse.sum()),
            "rows_without_reverse_counterpart": int((~has_reverse).sum()),
            "rate": float(has_reverse.float().mean()),
        }
    return profile


def _feature_profile(features: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = _tensor_profile(features)
    if features.ndim != 2 or not features.is_floating_point():
        return result
    finite = torch.isfinite(features)
    safe = torch.where(finite, features, torch.zeros_like(features))
    column_min = safe.min(dim=0).values
    column_max = safe.max(dim=0).values
    result.update({
        "all_zero_column_count": int(((column_min == 0) & (column_max == 0)).sum()),
        "constant_column_count": int((column_min == column_max).sum()),
        "all_zero_row_count": int((safe == 0).all(dim=1).sum()),
        "duplicate_full_row_count": int(
            features.shape[0] - torch.unique(features, dim=0).shape[0]
        ),
        "row_l2_norm_quantiles": _quantiles(torch.linalg.vector_norm(safe, dim=1)),
    })
    if features.shape[1] == EXPECTED_FEATURES:
        property_values = features[:, :20]
        tweet_values = features[:, 20:]
        boolean_audits = []
        numeric_audits = []
        for index, (name, kind) in enumerate(PROPERTY_FEATURES):
            values = property_values[:, index]
            item = {
                "index": index,
                "name": name,
                "kind": kind,
                "min": float(values.min()),
                "max": float(values.max()),
                "unique_count": int(torch.unique(values).numel()),
            }
            if kind == "boolean":
                item["invalid_boolean_count"] = int(
                    ((values != 0) & (values != 1)).sum()
                )
                boolean_audits.append(item)
            else:
                item["outside_unit_interval_count"] = int(
                    ((values < 0) | (values > 1)).sum()
                )
                numeric_audits.append(item)
        result["groups"] = {
            "property": {
                "shape": list(property_values.shape),
                "expected_semantics": "10 boolean + 10 MinMax-normalized numerical features",
                "boolean_fields": boolean_audits,
                "numerical_fields": numeric_audits,
            },
            "tweet_labse": {
                "shape": list(tweet_values.shape),
                "expected_semantics": "mean of each user's multilingual LaBSE tweet embeddings",
                "all_zero_row_count": int((tweet_values == 0).all(dim=1).sum()),
                "duplicate_row_count": int(
                    tweet_values.shape[0] - torch.unique(tweet_values, dim=0).shape[0]
                ),
                "min": float(tweet_values.min()),
                "max": float(tweet_values.max()),
            },
        }
    return result


def audit_mgtab_directory(
    root: Path,
    *,
    with_sha256: bool = True,
) -> dict[str, Any]:
    """Audit an extracted standard MGTAB directory without modifying it."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"MGTAB directory does not exist: {root}")

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    file_inventory = []
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file():
            continue
        file_inventory.append({
            "name": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            **({"sha256": sha256_file(path)} if with_sha256 else {}),
        })

    base: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "release": "MGTAB standard",
        "privacy": (
            "The report includes aggregate tensor statistics only; it exports no "
            "individual feature vectors or reconstructed account values."
        ),
        "source": {
            "repository": "https://github.com/GraphDetec/MGTAB",
            "paper": "https://arxiv.org/abs/2301.01123",
            "license": "CC BY-NC-ND 4.0",
        },
        "files": file_inventory,
        "missing_required_files": missing,
    }
    if missing:
        base.update({
            "detected_kind": "incomplete_or_unknown",
            "status": "fail",
            "findings": [_finding(
                "critical",
                "missing_required_files",
                "标准版张量文件不完整",
                {"missing": missing},
                "无法可靠构图或训练，张量之间也无法验证对齐。",
                "重新解压官方标准版，并在训练前重新运行本审计。",
            )],
        })
        return base

    tensors = {name: _load_tensor(root / name) for name in REQUIRED_FILES}
    edge_index = tensors["edge_index.pt"]
    edge_type = tensors["edge_type.pt"]
    edge_weight = tensors["edge_weight.pt"]
    features = tensors["features.pt"]
    bot_labels = tensors["labels_bot.pt"]
    stance_labels = tensors["labels_stance.pt"]

    findings: list[dict[str, Any]] = []
    node_count = int(features.shape[0]) if features.ndim >= 1 else 0
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0

    shape_errors = []
    if list(edge_index.shape[:1]) != [2] or edge_index.ndim != 2:
        shape_errors.append(f"edge_index shape={list(edge_index.shape)}")
    if edge_type.ndim != 1 or edge_type.numel() != edge_count:
        shape_errors.append(f"edge_type shape={list(edge_type.shape)}")
    if edge_weight.ndim != 1 or edge_weight.numel() != edge_count:
        shape_errors.append(f"edge_weight shape={list(edge_weight.shape)}")
    if features.ndim != 2:
        shape_errors.append(f"features shape={list(features.shape)}")
    if bot_labels.ndim != 1 or bot_labels.numel() != node_count:
        shape_errors.append(f"labels_bot shape={list(bot_labels.shape)}")
    if stance_labels.ndim != 1 or stance_labels.numel() != node_count:
        shape_errors.append(f"labels_stance shape={list(stance_labels.shape)}")
    if shape_errors:
        findings.append(_finding(
            "critical",
            "tensor_alignment_failure",
            "张量形状或长度不对齐",
            {"errors": shape_errors},
            "边、特征与标签可能指向不同记录，任何训练结果都不可信。",
            "停止训练，重新获取数据或核对上游转换脚本。",
        ))

    # Only run cross-tensor checks when the fundamental dimensions are usable.
    endpoints_valid = edge_index.ndim == 2 and edge_index.shape[0] == 2 and node_count > 0
    edge_domain = {}
    degree = torch.zeros(node_count, dtype=torch.int64)
    relation_profiles = []
    if endpoints_valid:
        negative = int((edge_index < 0).sum())
        too_large = int((edge_index >= node_count).sum())
        edge_domain = {
            "min_node_index": int(edge_index.min()) if edge_index.numel() else None,
            "max_node_index": int(edge_index.max()) if edge_index.numel() else None,
            "negative_endpoint_count": negative,
            "out_of_range_endpoint_count": too_large,
        }
        if negative or too_large:
            findings.append(_finding(
                "critical",
                "invalid_edge_endpoint",
                "边端点超出节点索引范围",
                edge_domain,
                "图消息会关联到不存在的节点或直接导致训练失败。",
                "停止训练并修复 edge_index 与 features 的节点映射。",
            ))
        else:
            degree = torch.bincount(edge_index.flatten(), minlength=node_count)

    allowed_relations = set(RELATION_TYPES)
    observed_relations = {int(value) for value in torch.unique(edge_type).tolist()}
    invalid_relations = sorted(observed_relations - allowed_relations)
    if invalid_relations:
        findings.append(_finding(
            "critical",
            "invalid_relation_type",
            "发现未定义的关系类型",
            {"invalid_values": invalid_relations},
            "HGT/RGCN 的关系参数无法与官方语义稳定对应。",
            "修复 edge_type，关系取值必须为 0-6。",
        ))
    elif endpoints_valid and edge_type.numel() == edge_count and edge_weight.numel() == edge_count:
        relation_profiles = [
            _relation_profile(
                edge_index, edge_type, edge_weight, relation_id, node_count
            )
            for relation_id in RELATION_TYPES
        ]

    bot_invalid = sorted(set(torch.unique(bot_labels).tolist()) - set(BOT_LABELS))
    stance_invalid = sorted(
        set(torch.unique(stance_labels).tolist()) - set(STANCE_LABELS)
    )
    if bot_invalid or stance_invalid:
        findings.append(_finding(
            "critical",
            "invalid_label_value",
            "标签存在未定义取值",
            {"bot": bot_invalid, "stance": stance_invalid},
            "监督目标语义不明确，评估指标可能被错误解释。",
            "在训练前固定标签字典并拒绝未知值。",
        ))

    nonfinite_features = (
        int((~torch.isfinite(features)).sum())
        if features.is_floating_point() else 0
    )
    nonfinite_weights = (
        int((~torch.isfinite(edge_weight)).sum())
        if edge_weight.is_floating_point() else 0
    )
    if nonfinite_features or nonfinite_weights:
        findings.append(_finding(
            "critical",
            "nonfinite_numeric_value",
            "特征或边权包含 NaN/Inf",
            {"feature_values": nonfinite_features, "edge_weights": nonfinite_weights},
            "会传播 NaN 梯度并破坏模型训练。",
            "定位源列并在进入模型前拒绝非有限值。",
        ))

    feature_audit = _feature_profile(features)
    tweet_zero_rows = int(
        feature_audit.get("groups", {})
        .get("tweet_labse", {})
        .get("all_zero_row_count", 0)
    )
    if tweet_zero_rows:
        findings.append(_finding(
            "medium",
            "zero_tweet_embeddings",
            "部分用户的 768 维推文向量全为零",
            {
                "rows": tweet_zero_rows,
                "rate": tweet_zero_rows / node_count if node_count else 0.0,
            },
            "这些节点实际只能依赖账号属性和图结构，文本模态缺失但没有显式缺失标记。",
            "增加 tweet_embedding_available 掩码，并按该掩码做分层评估。",
        ))

    duplicate_rows = sum(
        item.get("duplicate_edge_rows", 0)
        for item in relation_profiles
        if item["id"] in {3, 4}
    )
    if duplicate_rows:
        findings.append(_finding(
            "high",
            "repeated_reply_quote_edges",
            "reply/quoted 关系包含重复边记录",
            {
                "duplicate_rows": duplicate_rows,
                "relations": {
                    item["name"]: item["duplicate_edge_rows"]
                    for item in relation_profiles if item["id"] in {3, 4}
                },
            },
            "该发布包实际具有多重图语义；直接去重会丢失交互频率，直接保留则会隐式放大消息权重。",
            "在适配器中显式选择 preserve_multiedges 或 coalesce_with_count，并写入运行清单。",
        ))

    implicit_without_reverse = {
        item["name"]: item.get("reverse_row_coverage", {}).get(
            "rows_without_reverse_counterpart", 0
        )
        for item in relation_profiles if item["id"] in {5, 6}
    }
    if any(implicit_without_reverse.values()):
        findings.append(_finding(
            "high",
            "undirected_relations_stored_once",
            "URL/hashtag 被声明为无向关系，但文件中每条边只存一个方向",
            implicit_without_reverse,
            "若直接按有向 edge_index 送入 HGT，一半节点不会收到对称关系消息，结果依赖任意存储方向。",
            "构图时为 URL/hashtag 显式补反向边，并保持相同 PMI 权重。",
        ))

    isolated_nodes = int((degree == 0).sum()) if degree.numel() else 0
    if isolated_nodes:
        findings.append(_finding(
            "medium",
            "isolated_nodes",
            "标准版子图中存在孤立节点",
            {
                "count": isolated_nodes,
                "rate": isolated_nodes / node_count if node_count else 0.0,
            },
            "这些节点无法获得任何图消息，只能依赖自身特征。",
            "保留节点但单独报告其性能；不要为了方便静默删除。",
        ))

    findings.extend([
        _finding(
            "high",
            "no_released_split_or_timestamp",
            "标准版没有 split 文件或时间字段",
            {"official_loader": "7:2:1 random node split at runtime"},
            "无法做真正的时间外推测试；重新随机划分也可能让同一事件社区同时进入训练和测试。",
            "将 MGTAB 定位为事件内外部验证，并保存固定 split.csv、随机种子和划分哈希。",
        ),
        _finding(
            "high",
            "anonymous_precomputed_release",
            "发布包不包含原始用户 ID、原始推文或逐条证据",
            {
                "available": "anonymous node indices and precomputed features",
                "unavailable": ["original user IDs", "raw tweets", "timestamps", "campaign/role labels"],
            },
            "不能检查它与 TwiBot-22 的身份重叠，也不能用它直接验证角色保真、时间漂移或文本证据追溯。",
            "只将它用于 bot/stance 与多关系结构的外部验证；白盒证据追溯必须依赖 TwiBot-22 或仿真事件日志。",
        ),
    ])

    joint_distribution: dict[str, int] = {}
    if not bot_invalid and not stance_invalid and bot_labels.numel() == stance_labels.numel():
        for bot_value, bot_name in BOT_LABELS.items():
            for stance_value, stance_name in STANCE_LABELS.items():
                count = int(
                    ((bot_labels == bot_value) & (stance_labels == stance_value)).sum()
                )
                joint_distribution[f"{bot_name}|{stance_name}"] = count

    critical_count = sum(item["severity"] == "critical" for item in findings)
    high_count = sum(item["severity"] == "high" for item in findings)
    status = "fail" if critical_count else ("warning" if findings else "pass")
    base.update({
        "detected_kind": "mgtab_standard_tensor_bundle",
        "status": status,
        "summary": {
            "node_count": node_count,
            "edge_row_count": edge_count,
            "feature_dimension": int(features.shape[1]) if features.ndim == 2 else None,
            "relation_type_count": len(observed_relations),
            "critical_finding_count": critical_count,
            "high_finding_count": high_count,
        },
        "schema_contract": {
            "node_grain": "one expert-annotated Twitter account per row/index",
            "edge_grain": "one released relation row; reply/quoted may repeat the same pair",
            "feature_columns": [
                {"index": index, "name": name, "kind": kind}
                for index, (name, kind) in enumerate(PROPERTY_FEATURES)
            ] + [{
                "index_range": [20, 787],
                "name": "mean_tweet_LaBSE_embedding",
                "kind": "dense_embedding",
                "dimension": 768,
            }],
            "relation_types": {
                str(key): value for key, value in RELATION_TYPES.items()
            },
            "bot_labels": {str(key): value for key, value in BOT_LABELS.items()},
            "stance_labels": {
                str(key): value for key, value in STANCE_LABELS.items()
            },
        },
        "tensor_profiles": {
            name: (
                feature_audit if name == "features.pt" else _tensor_profile(tensor)
            )
            for name, tensor in tensors.items()
        },
        "label_distributions": {
            "bot": _distribution(bot_labels, BOT_LABELS),
            "stance": _distribution(stance_labels, STANCE_LABELS),
            "joint": joint_distribution,
        },
        "graph_integrity": {
            "edge_domain": edge_domain,
            "degree_quantiles_incident_rows": _quantiles(degree),
            "isolated_node_count": isolated_nodes,
            "relations": relation_profiles,
        },
        "capabilities": {
            "bot_labels": True,
            "stance_labels": True,
            "multi_relation_graph": True,
            "raw_text": False,
            "timestamps": False,
            "original_user_ids": False,
            "campaign_labels": False,
            "ground_truth_roles": False,
            "external_unlabelled_neighbors": False,
        },
        "expected_release_checks": {
            "node_count_matches": node_count == EXPECTED_NODES,
            "edge_count_matches": edge_count == EXPECTED_EDGES,
            "feature_dimension_matches": (
                features.ndim == 2 and features.shape[1] == EXPECTED_FEATURES
            ),
            "relation_counts_match": all(
                item["edge_rows"] == item["expected_edge_rows"]
                for item in relation_profiles
            ) if relation_profiles else False,
        },
        "findings": findings,
    })
    return base


def _format_count(value: int | float | None) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.4f}"
    return f"{int(value):,}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact technical audit report from the JSON manifest."""
    summary = report.get("summary", {})
    findings = report.get("findings", [])
    lines = [
        "# MGTAB 标准版字段与完整性审计",
        "",
        "## 技术结论",
        "",
        f"审计状态为 **{report.get('status', 'unknown')}**。数据文件可以作为多关系账号检测基准读取；"
        "但在接入双曲 HGT 前必须明确处理无向关系、重复交互边、缺失文本模态和固定数据划分。",
        "",
        f"- 节点：{_format_count(summary.get('node_count'))}",
        f"- 边记录：{_format_count(summary.get('edge_row_count'))}",
        f"- 特征：{_format_count(summary.get('feature_dimension'))} 维（20 维账号属性 + 768 维 LaBSE 推文向量）",
        f"- 严重问题：{_format_count(summary.get('critical_finding_count'))}；高风险建模事项：{_format_count(summary.get('high_finding_count'))}",
        "",
        "## 字段合同",
        "",
        "| 文件 | 粒度/形状 | 含义 |",
        "|---|---:|---|",
    ]
    profiles = report.get("tensor_profiles", {})
    meanings = {
        "features.pt": "节点特征；前 20 维为属性，后 768 维为用户推文 LaBSE 均值向量",
        "labels_bot.pt": "节点二分类标签：0=human，1=bot",
        "labels_stance.pt": "节点三分类标签：0=neutral，1=against，2=support",
        "edge_index.pt": "COO 边端点；第 0 行 source，第 1 行 target",
        "edge_type.pt": "与每条边对齐的关系类型 0-6",
        "edge_weight.pt": "与每条边对齐的权重；显式关系为 1，URL/hashtag 为共现权重",
    }
    for name in REQUIRED_FILES:
        shape = profiles.get(name, {}).get("shape")
        lines.append(f"| `{name}` | `{shape}` | {meanings[name]} |")

    lines.extend([
        "",
        "## 关系审计",
        "",
        "| ID | 关系 | 方向语义 | 边记录 | 重复记录 | 自环 | 权重范围 |",
        "|---:|---|---|---:|---:|---:|---|",
    ])
    for item in report.get("graph_integrity", {}).get("relations", []):
        weight = item["weight"]
        lines.append(
            f"| {item['id']} | {item['name']} | {item['declared_direction']} | "
            f"{item['edge_rows']:,} | {item['duplicate_edge_rows']:,} | "
            f"{item['self_loop_rows']:,} | {weight['min']:.4f}–{weight['max']:.4f} |"
        )

    lines.extend([
        "",
        "## 主要发现",
        "",
        "| 严重度 | 发现 | 影响与处理 |",
        "|---|---|---|",
    ])
    for item in findings:
        lines.append(
            f"| {item['severity']} | {item['title']} | {item['risk']} {item['recommendation']} |"
        )

    lines.extend([
        "",
        "## 对 HyperDecept 的可用边界",
        "",
        "- 可用：bot/stance 节点分类、多关系结构外部验证、关系级消融、几何解释测试。",
        "- 不可直接验证：真实时间漂移、原始文本证据追溯、用户身份跨数据集去重、团伙角色保真。",
        "- 推荐定位：将 MGTAB 作为独立外部验证集，而不是替代 TwiBot-22 的主要训练语料。",
        "- 训练前必须生成固定 `split.csv`，并记录随机种子、划分哈希、无向边补全策略和重复边聚合策略。",
        "",
        "## 方法与限制",
        "",
        "审计检查了文件哈希、张量形状和类型、标签域、有限值、特征范围、图端点、关系分布、"
        "重复边、自环、反向边覆盖和孤立节点。该标准发布包不含原始 ID、推文或时间戳，因此无法从"
        "本地张量独立复核专家标注过程、LaBSE 编码输入或数据采集时间。",
        "",
        "## 下一步",
        "",
        "1. 实现 MGTAB → HyperDecept 的只读适配器。",
        "2. 生成确定性的 `label.csv`、`split.csv` 与特征合同，不伪造原始 ID 或时间字段。",
        "3. 分别测试保留多重边和按计数聚合两种策略；URL/hashtag 都补反向边。",
        "4. 在报告指标中单列 52 个孤立节点和零推文向量节点。",
        "",
        "## 仍需确认",
        "",
        "- 团队最终采用纯外部测试，还是把 MGTAB 的训练分区加入联合训练。",
        "- 对 reply/quoted 的重复记录，采用多重边、计数权重还是二值化。",
        "",
        "来源：MGTAB 官方仓库与论文。完整机器可读证据见同目录 JSON manifest。",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the standard MGTAB tensor bundle without modifying it."
    )
    parser.add_argument("--mgtab-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip per-file SHA256 hashes for a faster audit.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = audit_mgtab_directory(
        args.mgtab_dir,
        with_sha256=not args.skip_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
