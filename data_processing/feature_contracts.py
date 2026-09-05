"""Named, auditable feature contracts for HyperTrace detector experiments."""

from __future__ import annotations

from typing import Final


SEMANTIC_8: Final[tuple[str, ...]] = tuple(
    f"Semantic_{index}" for index in range(8)
)
BEHAVIOR_10: Final[tuple[str, ...]] = (
    "Follower_Following_Ratio",
    "Action_Frequency",
    "Like_Ratio",
    "Retweet_Ratio",
    "Reply_Ratio",
    "Temporal_Entropy",
    "URL_Ratio",
    "Mention_Ratio",
    "Hashtag_Ratio",
    "Media_Ratio",
)
OBSERVABLE_18: Final[tuple[str, ...]] = SEMANTIC_8 + BEHAVIOR_10
OBSERVABLE_17_NO_TEMPORAL: Final[tuple[str, ...]] = tuple(
    name for name in OBSERVABLE_18 if name != "Temporal_Entropy"
)

FEATURE_CONTRACTS: Final[dict[str, tuple[str, ...]]] = {
    "observable18": OBSERVABLE_18,
    "observable17_no_temporal": OBSERVABLE_17_NO_TEMPORAL,
}


def resolve_feature_contract(name: str) -> tuple[str, ...]:
    """Resolve a stable contract name without silently accepting typos."""
    try:
        return FEATURE_CONTRACTS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature contract {name!r}; expected one of "
            f"{sorted(FEATURE_CONTRACTS)}"
        ) from exc
