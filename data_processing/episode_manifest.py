"""Dataset planning and split auditing for HyperDecept joint training.

The statistical unit of the simulation corpus is an independently generated
episode, not an individual agent row.  This module records the provenance and
intended use of every real or synthetic graph before expensive server runs
start, and rejects split plans that leak identities, campaigns, scenarios, or
duplicate simulation seeds across evaluation boundaries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "hyperdecept.dataset-plan.v1"
PATH_CONTRACT = "hyperdecept.manifest-relative.v1"
DOMAINS = {"real", "synthetic"}
PURPOSES = {
    "real_primary",
    "real_external",
    "real_temporal_external",
    "simulation_main",
    "simulation_scale",
    "synthetic_external",
}
PARTITIONS = {"shared", "pool", "external_test", "scale_test"}
SPLIT_LEVELS = {"node", "episode", "scenario"}
LABEL_PROVENANCE = {"annotated", "observed", "generated", "inferred"}
DEFAULT_ATTACK_PHASES = (
    "benign_warmup",
    "coordination_onset",
    "propagation_escalation",
    "adaptive_evasion",
    "aftermath",
)


def _resolve_manifest_path(value: str, base_dir: Optional[Path]) -> str:
    """Resolve a declared local path against the manifest that owns it."""
    path = Path(str(value)).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path.resolve()) if base_dir is not None else str(path)


def _portable_manifest_path(value: str, base_dir: Path) -> str:
    """Serialize a local path relative to its manifest when possible."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    try:
        return Path(os.path.relpath(path, start=base_dir)).as_posix()
    except ValueError as exc:
        raise ValueError(
            "manifest paths must share a filesystem volume with the manifest; "
            f"cannot relativize {path} against {base_dir}"
        ) from exc


def _clean_identifier(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if any(char in text for char in "\r\n\t"):
        raise ValueError(f"{field_name} must not contain control whitespace")
    return text


def _relative_source_path(
    root: Path,
    scenario_id: str,
    num_agents: int,
    simulation_seed: int,
    purpose: str,
) -> str:
    folder = "main" if purpose == "simulation_main" else "scale"
    return str(
        root
        / folder
        / scenario_id
        / f"n{num_agents}"
        / f"seed_{simulation_seed}.db"
    )


@dataclass(frozen=True)
class EpisodeManifest:
    """Provenance and task contract for one graph episode or real snapshot."""

    episode_id: str
    dataset_name: str
    domain: str
    purpose: str
    partition: str
    split_level: str
    source_path: str
    identity_scope: str
    scenario_id: Optional[str] = None
    simulation_seed: Optional[int] = None
    num_agents: Optional[int] = None
    time_steps: Optional[int] = None
    attack_phases: Tuple[str, ...] = ()
    label_provenance: Mapping[str, str] = field(default_factory=dict)
    capabilities: Mapping[str, bool] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    generator_metadata: Mapping[str, str] = field(default_factory=dict)
    source_sha256: Optional[str] = None
    status: str = "planned"
    path_contract: str = PATH_CONTRACT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", _clean_identifier(self.episode_id, "episode_id")
        )
        object.__setattr__(
            self, "dataset_name", _clean_identifier(self.dataset_name, "dataset_name")
        )
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {sorted(DOMAINS)}")
        if self.purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {sorted(PURPOSES)}")
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {sorted(PARTITIONS)}")
        if self.split_level not in SPLIT_LEVELS:
            raise ValueError(f"split_level must be one of {sorted(SPLIT_LEVELS)}")
        if self.identity_scope not in {"dataset", "episode"}:
            raise ValueError("identity_scope must be 'dataset' or 'episode'")
        if self.status not in {"planned", "ready", "failed"}:
            raise ValueError("status must be planned, ready, or failed")
        if self.path_contract != PATH_CONTRACT:
            raise ValueError(f"unsupported path_contract: {self.path_contract}")
        if not str(self.source_path).strip():
            raise ValueError("source_path must not be empty")
        invalid_provenance = set(self.label_provenance.values()).difference(
            LABEL_PROVENANCE
        )
        if invalid_provenance:
            raise ValueError(
                f"unsupported label provenance: {sorted(invalid_provenance)}"
            )
        empty_artifacts = sorted(
            name for name, path in self.artifacts.items()
            if not str(name).strip() or not str(path).strip()
        )
        if empty_artifacts:
            raise ValueError(f"artifact paths must not be empty: {empty_artifacts}")
        if self.domain == "synthetic" and self.purpose in {
            "simulation_main", "simulation_scale"
        }:
            if not self.scenario_id:
                raise ValueError("synthetic episodes require scenario_id")
            if self.simulation_seed is None:
                raise ValueError("synthetic episodes require simulation_seed")
            if self.num_agents is None or self.num_agents <= 0:
                raise ValueError("synthetic episodes require positive num_agents")
            if self.identity_scope != "episode":
                raise ValueError("controlled synthetic identities must be episode-scoped")
        elif self.identity_scope != "dataset":
            raise ValueError("real and external benchmark identities must be dataset-scoped")
        if self.time_steps is not None and self.time_steps <= 0:
            raise ValueError("time_steps must be positive when provided")
        if self.source_sha256 is not None:
            digest = self.source_sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("source_sha256 must be a hexadecimal SHA-256 digest")
        if self.status == "ready" and not self.source_sha256:
            raise ValueError("ready episodes require source_sha256")

    def to_dict(self, *, base_dir: Optional[str | Path] = None) -> dict:
        value = asdict(self)
        value["attack_phases"] = list(self.attack_phases)
        value["label_provenance"] = dict(self.label_provenance)
        value["capabilities"] = dict(self.capabilities)
        value["artifacts"] = dict(self.artifacts)
        value["generator_metadata"] = dict(self.generator_metadata)
        if base_dir is not None:
            owner = Path(base_dir).expanduser().resolve()
            value["source_path"] = _portable_manifest_path(
                self.source_path, owner
            )
            value["artifacts"] = {
                name: _portable_manifest_path(path, owner)
                for name, path in self.artifacts.items()
            }
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Optional[str | Path] = None,
    ) -> "EpisodeManifest":
        data = dict(value)
        data["attack_phases"] = tuple(data.get("attack_phases") or ())
        if base_dir is not None:
            owner = Path(base_dir).expanduser().resolve()
            data["source_path"] = _resolve_manifest_path(
                str(data["source_path"]), owner
            )
            data["artifacts"] = {
                str(name): _resolve_manifest_path(str(path), owner)
                for name, path in dict(data.get("artifacts") or {}).items()
            }
        return cls(**data)

    def write(self, path: str | Path) -> Path:
        """Write a relocatable standalone manifest."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(base_dir=target.parent),
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def read(cls, path: str | Path) -> "EpisodeManifest":
        """Read paths relative to the directory containing the manifest."""
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle), base_dir=source.parent)


@dataclass(frozen=True)
class DatasetPlan:
    """Serializable collection of all real and simulated graph episodes."""

    plan_id: str
    episodes: Tuple[EpisodeManifest, ...]
    strategy: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _clean_identifier(self.plan_id, "plan_id")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        identifiers = [episode.episode_id for episode in self.episodes]
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate episode_id values: {duplicates}")
        signatures = [
            (
                episode.dataset_name,
                episode.scenario_id,
                episode.simulation_seed,
                episode.num_agents,
                episode.purpose,
            )
            for episode in self.episodes
            if episode.domain == "synthetic"
        ]
        duplicate_signatures = sorted({
            signature for signature in signatures if signatures.count(signature) > 1
        })
        if duplicate_signatures:
            raise ValueError(
                "duplicate synthetic scenario/seed/size episodes: "
                f"{duplicate_signatures}"
            )

    def to_dict(self, *, base_dir: Optional[str | Path] = None) -> dict:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "strategy": dict(self.strategy),
            "episodes": [
                episode.to_dict(base_dir=base_dir) for episode in self.episodes
            ],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Optional[str | Path] = None,
    ) -> "DatasetPlan":
        return cls(
            schema_version=str(value.get("schema_version", "")),
            plan_id=str(value["plan_id"]),
            strategy=dict(value.get("strategy") or {}),
            episodes=tuple(
                EpisodeManifest.from_dict(item, base_dir=base_dir)
                for item in value.get("episodes", [])
            ),
        )

    def write(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(base_dir=target.parent),
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def read(cls, path: str | Path) -> "DatasetPlan":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle), base_dir=source.parent)

    def summary(self) -> dict:
        synthetic = [episode for episode in self.episodes if episode.domain == "synthetic"]
        return {
            "episodes": len(self.episodes),
            "real_episodes": sum(episode.domain == "real" for episode in self.episodes),
            "synthetic_episodes": len(synthetic),
            "synthetic_agent_instances": sum(
                int(episode.num_agents or 0) for episode in synthetic
            ),
            "main_scenarios": sorted({
                episode.scenario_id for episode in synthetic
                if episode.purpose == "simulation_main"
            }),
            "scale_sizes": sorted({
                int(episode.num_agents) for episode in synthetic
                if episode.purpose == "simulation_scale"
                and episode.num_agents is not None
            }),
        }


@dataclass(frozen=True)
class SplitAuditReport:
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_plan_artifacts(
    plan: DatasetPlan,
    *,
    require_files: bool = False,
) -> SplitAuditReport:
    """Check that every episode declares the files needed by the trainer."""
    errors: List[str] = []
    warnings: List[str] = []
    for episode in plan.episodes:
        if episode.dataset_name == "mgtab":
            required = {
                "edge_index_pt",
                "edge_type_pt",
                "edge_weight_pt",
                "features_pt",
                "labels_bot_pt",
                "labels_stance_pt",
                "splits_csv",
                "adapter_manifest_json",
            }
        else:
            required = {"features_csv", "labels_csv"}
        if episode.purpose in {"simulation_main", "simulation_scale"}:
            required.update({"profiles_csv", "event_targets_csv", "episode_manifest"})
        elif episode.dataset_name != "mgtab":
            required.add("splits_csv")
        missing_contract = sorted(required.difference(episode.artifacts))
        if missing_contract:
            errors.append(
                f"{episode.episode_id} missing artifact declarations: {missing_contract}"
            )
        paths = {"source": episode.source_path, **dict(episode.artifacts)}
        for name, path in paths.items():
            if Path(path).exists():
                continue
            message = f"{episode.episode_id} {name} does not exist: {path}"
            if require_files or episode.status == "ready":
                errors.append(message)
            else:
                warnings.append(message)
        source = Path(episode.source_path)
        if (
            episode.source_sha256
            and source.is_file()
            and sha256_file(source) != episode.source_sha256
        ):
            errors.append(f"{episode.episode_id} source SHA-256 mismatch")
    return SplitAuditReport(tuple(sorted(set(errors))), tuple(sorted(set(warnings))))


def build_recommended_dataset_plan(
    *,
    simulation_root: str | Path,
    twibot_root: str | Path,
    mgtab_root: str | Path,
    fox8_root: Optional[str | Path] = None,
    botsim_root: Optional[str | Path] = None,
    scenarios: Sequence[str],
    main_seeds: Sequence[int] = (11, 22, 33, 44),
    anchor_num_agents: int = 2000,
    scale_scenarios: Optional[Sequence[str]] = None,
    scale_sizes: Sequence[int] = (500, 1000, 5000),
    scale_seeds: Sequence[int] = (101, 102, 103),
    time_steps: int = 50,
) -> DatasetPlan:
    """Build the recommended incomplete-factorial data-generation plan.

    All scenario families are generated at the anchor size.  Only two
    representative scenarios receive the expensive size sweep, preventing a
    wasteful scenario x seed x size Cartesian product.
    """
    scenario_ids = tuple(_clean_identifier(value, "scenario_id") for value in scenarios)
    if len(scenario_ids) < 3 or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenarios must contain at least three unique values")
    if anchor_num_agents <= 0 or time_steps <= 0:
        raise ValueError("anchor_num_agents and time_steps must be positive")
    if not main_seeds or len(set(main_seeds)) != len(main_seeds):
        raise ValueError("main_seeds must be non-empty and unique")
    if not scale_seeds or len(set(scale_seeds)) != len(scale_seeds):
        raise ValueError("scale_seeds must be non-empty and unique")
    if any(size <= 0 for size in scale_sizes):
        raise ValueError("scale_sizes must be positive")
    selected_scale = tuple(scale_scenarios or scenario_ids[:2])
    if not selected_scale or not set(selected_scale).issubset(scenario_ids):
        raise ValueError("scale_scenarios must be drawn from scenarios")

    episodes: List[EpisodeManifest] = [
        EpisodeManifest(
            episode_id="real:twibot22:official",
            dataset_name="twibot22",
            domain="real",
            purpose="real_primary",
            partition="shared",
            split_level="node",
            source_path=str(Path(twibot_root)),
            identity_scope="dataset",
            label_provenance={"bot": "annotated"},
            capabilities={
                "temporal": True,
                "raw_text": True,
                "external_neighbors": True,
                "ground_truth_roles": False,
            },
            artifacts={
                "core_ids": str(Path(twibot_root) / "derived" / "core_ids.txt"),
                "features_csv": str(
                    Path(twibot_root) / "derived" / "node_features_26d.csv"
                ),
                "labels_csv": str(Path(twibot_root) / "label.csv"),
                "splits_csv": str(Path(twibot_root) / "split.csv"),
            },
        ),
        EpisodeManifest(
            episode_id="real:mgtab:external",
            dataset_name="mgtab",
            domain="real",
            purpose="real_external",
            partition="shared",
            split_level="node",
            source_path=str(Path(mgtab_root)),
            identity_scope="dataset",
            label_provenance={"bot": "annotated"},
            capabilities={
                "temporal": False,
                # The standard release contains anonymous 20d profile features
                # plus 768d precomputed LaBSE vectors, not raw tweets or the
                # 400k unlabelled neighbors from MGTAB-large.
                "raw_text": False,
                "precomputed_text_embeddings": True,
                "external_neighbors": False,
                "ground_truth_roles": False,
            },
            artifacts={
                "edge_index_pt": str(Path(mgtab_root) / "edge_index.pt"),
                "edge_type_pt": str(Path(mgtab_root) / "edge_type.pt"),
                "edge_weight_pt": str(Path(mgtab_root) / "edge_weight.pt"),
                "features_pt": str(Path(mgtab_root) / "features.pt"),
                "labels_bot_pt": str(Path(mgtab_root) / "labels_bot.pt"),
                "labels_stance_pt": str(Path(mgtab_root) / "labels_stance.pt"),
                "splits_csv": str(
                    Path(mgtab_root) / "derived" / "split_seed42.csv"
                ),
                "adapter_manifest_json": str(
                    Path(mgtab_root)
                    / "derived"
                    / "adapter_manifest_seed42.json"
                ),
            },
            generator_metadata={
                "split_seed": "42",
                "multiedge_policy": "coalesce_with_count",
            },
        ),
    ]
    if fox8_root is not None:
        episodes.append(EpisodeManifest(
            episode_id="real:fox8-23:external",
            dataset_name="fox8-23",
            domain="real",
            purpose="real_temporal_external",
            partition="external_test",
            split_level="node",
            source_path=str(Path(fox8_root)),
            identity_scope="dataset",
            label_provenance={"bot": "annotated"},
            capabilities={
                "temporal": True,
                "raw_text": True,
                "external_neighbors": True,
                "ground_truth_roles": False,
            },
            artifacts={
                "features_csv": str(
                    Path(fox8_root) / "derived" / "node_features_26d.csv"
                ),
                "labels_csv": str(Path(fox8_root) / "label.csv"),
                "splits_csv": str(Path(fox8_root) / "split.csv"),
            },
        ))
    if botsim_root is not None:
        episodes.append(EpisodeManifest(
            episode_id="synthetic:botsim-24:external",
            dataset_name="botsim-24",
            domain="synthetic",
            purpose="synthetic_external",
            partition="external_test",
            split_level="node",
            source_path=str(Path(botsim_root)),
            identity_scope="dataset",
            label_provenance={"bot": "generated"},
            capabilities={
                "temporal": True,
                "raw_text": True,
                "ground_truth_roles": False,
                "ground_truth_campaigns": False,
            },
            artifacts={
                "features_csv": str(
                    Path(botsim_root) / "derived" / "node_features_26d.csv"
                ),
                "labels_csv": str(Path(botsim_root) / "label.csv"),
                "splits_csv": str(Path(botsim_root) / "split.csv"),
            },
        ))
    simulation_path = Path(simulation_root)
    for scenario_id in scenario_ids:
        for seed in main_seeds:
            episode_id = f"sim:{scenario_id}:n{anchor_num_agents}:s{seed}:main"
            source_path = _relative_source_path(
                simulation_path,
                scenario_id,
                anchor_num_agents,
                int(seed),
                "simulation_main",
            )
            source_prefix = str(Path(source_path).with_suffix(""))
            episodes.append(EpisodeManifest(
                episode_id=episode_id,
                dataset_name="deeppersona_oasis",
                domain="synthetic",
                purpose="simulation_main",
                partition="pool",
                split_level="scenario",
                source_path=source_path,
                identity_scope="episode",
                scenario_id=scenario_id,
                simulation_seed=int(seed),
                num_agents=anchor_num_agents,
                time_steps=time_steps,
                attack_phases=DEFAULT_ATTACK_PHASES,
                label_provenance={
                    "bot": "generated",
                    "role": "generated",
                    "campaign": "generated",
                    "attack_phase": "generated",
                    "next_action": "generated",
                },
                capabilities={
                    "temporal": True,
                    "raw_text": True,
                    "ground_truth_roles": True,
                    "ground_truth_campaigns": True,
                },
                artifacts={
                    "profiles_csv": f"{source_prefix}.csv",
                    "features_csv": f"{source_prefix}.features.csv",
                    "labels_csv": f"{source_prefix}.labels.csv",
                    "event_targets_csv": f"{source_prefix}.event_targets.csv",
                    "episode_manifest": f"{source_prefix}.manifest.json",
                },
            ))

    for scenario_id in selected_scale:
        for size in scale_sizes:
            if int(size) == anchor_num_agents:
                continue
            for seed in scale_seeds:
                episode_id = f"sim:{scenario_id}:n{int(size)}:s{seed}:scale"
                source_path = _relative_source_path(
                    simulation_path,
                    scenario_id,
                    int(size),
                    int(seed),
                    "simulation_scale",
                )
                source_prefix = str(Path(source_path).with_suffix(""))
                episodes.append(EpisodeManifest(
                    episode_id=episode_id,
                    dataset_name="deeppersona_oasis",
                    domain="synthetic",
                    purpose="simulation_scale",
                    partition="scale_test",
                    split_level="episode",
                    source_path=source_path,
                    identity_scope="episode",
                    scenario_id=scenario_id,
                    simulation_seed=int(seed),
                    num_agents=int(size),
                    time_steps=time_steps,
                    attack_phases=DEFAULT_ATTACK_PHASES,
                    label_provenance={
                        "bot": "generated",
                        "role": "generated",
                        "campaign": "generated",
                        "attack_phase": "generated",
                        "next_action": "generated",
                    },
                    capabilities={
                        "temporal": True,
                        "raw_text": True,
                        "ground_truth_roles": True,
                        "ground_truth_campaigns": True,
                    },
                    artifacts={
                        "profiles_csv": f"{source_prefix}.csv",
                        "features_csv": f"{source_prefix}.features.csv",
                        "labels_csv": f"{source_prefix}.labels.csv",
                        "event_targets_csv": f"{source_prefix}.event_targets.csv",
                        "episode_manifest": f"{source_prefix}.manifest.json",
                    },
                ))

    return DatasetPlan(
        plan_id="hyperdecept_joint_v1",
        episodes=tuple(episodes),
        strategy={
            "main_design": "scenario_x_seed_at_anchor_size",
            "scenario_evaluation": "leave_one_scenario_out",
            "scale_design": "representative_scenarios_only",
            "anchor_num_agents": anchor_num_agents,
            "time_steps": time_steps,
            "model_seeds": [7, 17, 27],
            "primary_real_metric": "AUPRC",
            "training_protocols": {
                "P1_external_holdout": (
                    "train TwiBot-22 + DeepPersona/OASIS; "
                    "test MGTAB, Fox8-23, and BotSim-24"
                ),
                "P2_multisource_real": (
                    "train TwiBot-22 train + MGTAB train + DeepPersona/OASIS; "
                    "evaluate official real test splits and external candidates"
                ),
            },
        },
    )


def leave_one_scenario_out_assignments(
    plan: DatasetPlan,
    held_out_scenario: str,
    validation_seed: Optional[int] = None,
) -> Dict[str, str]:
    """Assign main simulation episodes to train/validation/test for one fold."""
    main = [
        episode for episode in plan.episodes
        if episode.purpose == "simulation_main"
    ]
    scenarios = {episode.scenario_id for episode in main}
    if held_out_scenario not in scenarios:
        raise ValueError(f"unknown held_out_scenario: {held_out_scenario}")
    seen = [episode for episode in main if episode.scenario_id != held_out_scenario]
    seeds = sorted({int(episode.simulation_seed) for episode in seen})
    if not seeds:
        raise ValueError("no seen-scenario episodes are available")
    selected_validation_seed = seeds[-1] if validation_seed is None else validation_seed
    if selected_validation_seed not in seeds:
        raise ValueError("validation_seed is absent from seen scenarios")

    assignments: Dict[str, str] = {}
    for episode in plan.episodes:
        if episode.purpose == "simulation_main":
            if episode.scenario_id == held_out_scenario:
                assignments[episode.episode_id] = "test"
            elif episode.simulation_seed == selected_validation_seed:
                assignments[episode.episode_id] = "validation"
            else:
                assignments[episode.episode_id] = "train"
        elif episode.purpose in {
            "simulation_scale", "real_external", "real_temporal_external",
            "synthetic_external",
        }:
            assignments[episode.episode_id] = "test"
        else:
            assignments[episode.episode_id] = "shared"
    return assignments


def training_protocol_assignments(
    plan: DatasetPlan,
    *,
    protocol_id: str,
    held_out_scenario: str,
    validation_seed: Optional[int] = None,
) -> Dict[str, str]:
    """Combine LOSO simulation splits with one of the two real-data protocols."""
    if protocol_id not in {"P1_external_holdout", "P2_multisource_real"}:
        raise ValueError("unknown training protocol")
    assignments = leave_one_scenario_out_assignments(
        plan,
        held_out_scenario,
        validation_seed=validation_seed,
    )
    for episode in plan.episodes:
        if episode.dataset_name == "twibot22":
            assignments[episode.episode_id] = "shared"
        elif episode.dataset_name == "mgtab":
            assignments[episode.episode_id] = (
                "shared" if protocol_id == "P2_multisource_real" else "test"
            )
        elif episode.purpose in {"real_temporal_external", "synthetic_external"}:
            assignments[episode.episode_id] = "test"
    return assignments


def audit_episode_splits(
    plan: DatasetPlan,
    assignments: Mapping[str, str],
    *,
    user_ids_by_episode: Optional[Mapping[str, Iterable[str]]] = None,
    campaign_ids_by_episode: Optional[Mapping[str, Iterable[str]]] = None,
    require_scenario_holdout: bool = True,
) -> SplitAuditReport:
    """Audit episode assignments without pretending agents are independent.

    Real user IDs are dataset-scoped and therefore must not cross node-level
    train/validation/test extracts.  Synthetic user IDs are episode-scoped,
    because every simulation may legitimately reuse local IDs such as ``0``.
    """
    errors: List[str] = []
    warnings: List[str] = []
    known_ids = {episode.episode_id for episode in plan.episodes}
    unknown = sorted(set(assignments).difference(known_ids))
    missing = sorted(known_ids.difference(assignments))
    if unknown:
        errors.append(f"assignments contain unknown episodes: {unknown}")
    if missing:
        errors.append(f"assignments omit episodes: {missing}")
    invalid_splits = sorted(set(assignments.values()).difference(
        {"train", "validation", "test", "shared"}
    ))
    if invalid_splits:
        errors.append(f"unsupported split names: {invalid_splits}")

    episodes = {episode.episode_id: episode for episode in plan.episodes}
    signatures: Dict[Tuple[str, str, int, int], Tuple[str, str]] = {}
    for episode_id, split in assignments.items():
        episode = episodes.get(episode_id)
        if episode is None or episode.domain != "synthetic":
            continue
        signature = (
            episode.dataset_name,
            str(episode.scenario_id),
            int(episode.simulation_seed),
            int(episode.num_agents),
        )
        previous = signatures.get(signature)
        if previous is not None and previous[1] != split:
            errors.append(
                f"synthetic signature {signature} crosses {previous[1]} and {split}"
            )
        signatures[signature] = (episode_id, split)

    if require_scenario_holdout:
        train_scenarios = {
            episodes[episode_id].scenario_id
            for episode_id, split in assignments.items()
            if episode_id in episodes
            and split == "train"
            and episodes[episode_id].purpose == "simulation_main"
        }
        test_scenarios = {
            episodes[episode_id].scenario_id
            for episode_id, split in assignments.items()
            if episode_id in episodes
            and split == "test"
            and episodes[episode_id].purpose == "simulation_main"
        }
        overlap = sorted(train_scenarios.intersection(test_scenarios))
        if overlap:
            errors.append(f"main simulation scenarios cross train/test: {overlap}")

    def audit_scoped_ids(
        values_by_episode: Optional[Mapping[str, Iterable[str]]],
        label: str,
    ) -> None:
        if values_by_episode is None:
            warnings.append(f"{label} membership was not supplied; overlap not checked")
            return
        owners: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for episode_id, raw_values in values_by_episode.items():
            episode = episodes.get(episode_id)
            if episode is None:
                errors.append(f"{label} membership references unknown episode {episode_id}")
                continue
            split = assignments.get(episode_id)
            if split not in {"train", "validation", "test"}:
                continue
            for raw_value in raw_values:
                value = str(raw_value)
                scope = (
                    episode.dataset_name
                    if episode.identity_scope == "dataset"
                    else episode.episode_id
                )
                key = (scope, value)
                previous = owners.get(key)
                if previous is not None and previous[1] != split:
                    errors.append(
                        f"{label} {value!r} in scope {scope!r} crosses "
                        f"{previous[1]} and {split}"
                    )
                owners[key] = (episode_id, split)

    audit_scoped_ids(user_ids_by_episode, "user")
    audit_scoped_ids(campaign_ids_by_episode, "campaign")
    return SplitAuditReport(tuple(sorted(set(errors))), tuple(sorted(set(warnings))))


def _parse_csv_values(value: str, caster=str) -> list:
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan HyperDecept training datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="write a recommended plan")
    create.add_argument("--output", required=True)
    create.add_argument("--simulation-root", required=True)
    create.add_argument("--twibot-root", required=True)
    create.add_argument("--mgtab-root", required=True)
    create.add_argument("--fox8-root")
    create.add_argument("--botsim-root")
    create.add_argument(
        "--scenarios",
        default="leader_amplifier,bridge_infiltration,synchronized_boosting,persona_drift,adaptive_evasion",
    )
    create.add_argument("--main-seeds", default="11,22,33,44")
    create.add_argument("--anchor-num-agents", type=int, default=2000)
    create.add_argument("--scale-scenarios", default="leader_amplifier,adaptive_evasion")
    create.add_argument("--scale-sizes", default="500,1000,5000")
    create.add_argument("--scale-seeds", default="101,102,103")
    create.add_argument("--time-steps", type=int, default=50)
    validate = subparsers.add_parser("validate", help="validate and summarize a plan")
    validate.add_argument("--input", required=True)
    validate.add_argument("--require-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        plan = build_recommended_dataset_plan(
            simulation_root=args.simulation_root,
            twibot_root=args.twibot_root,
            mgtab_root=args.mgtab_root,
            fox8_root=args.fox8_root,
            botsim_root=args.botsim_root,
            scenarios=_parse_csv_values(args.scenarios),
            main_seeds=_parse_csv_values(args.main_seeds, int),
            anchor_num_agents=args.anchor_num_agents,
            scale_scenarios=_parse_csv_values(args.scale_scenarios),
            scale_sizes=_parse_csv_values(args.scale_sizes, int),
            scale_seeds=_parse_csv_values(args.scale_seeds, int),
            time_steps=args.time_steps,
        )
        output = plan.write(args.output)
        print(json.dumps({"plan": str(output), **plan.summary()}, ensure_ascii=False))
    else:
        plan = DatasetPlan.read(args.input)
        report = audit_plan_artifacts(plan, require_files=args.require_files)
        result = {
            **plan.summary(),
            "artifact_contract_valid": report.valid,
            "errors": list(report.errors),
            "warnings": list(report.warnings),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        report.raise_for_errors()


if __name__ == "__main__":
    main()
