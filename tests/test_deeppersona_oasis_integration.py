"""Regression tests for the DeepPersona -> OASIS loading chain."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = PROJECT_ROOT / "MultiAgent4Collusion-master"
DEEPPERSONA_ROOT = PROJECT_ROOT / "deeppersona_ai"
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(DEEPPERSONA_ROOT))

import generate_simulation_csv as persona_adapter
import rag_engine
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform


def test_example_config_paths_do_not_depend_on_launch_directory():
    import importlib.util

    runner_path = (
        FRAMEWORK_ROOT
        / "scripts" / "twitter_gpt_example" / "twitter_simulation_large.py"
    )
    spec = importlib.util.spec_from_file_location("deeppersona_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)

    resolved = runner.resolve_framework_data_paths({
        "csv_path": "our_twitter_sim/False_Business_0.csv",
        "db_path": "our_twitter_sim/False_Business_0.db",
    })
    assert Path(resolved["csv_path"]).parent == FRAMEWORK_ROOT / "our_twitter_sim"
    assert Path(resolved["db_path"]).parent == FRAMEWORK_ROOT / "our_twitter_sim"


def _profiles(count=5):
    return [
        {"Summary": f"Agent {index} follows economic and social policy news."}
        for index in range(count)
    ]


def test_persona_adapter_populates_prompt_and_recommender_fields(monkeypatch):
    profiles = _profiles()
    monkeypatch.setattr(persona_adapter, "NUM_BAD_LEADER", 1)
    monkeypatch.setattr(persona_adapter, "NUM_BAD_MEMBER", 1)
    monkeypatch.setattr(persona_adapter, "NUM_BAD", 1)
    frame = persona_adapter.build_agent_dataframe(
        profiles,
        tweet_pool={"good": [], "bad": []},
    )

    assert len(frame) == len(profiles)
    assert frame["user_char"].str.strip().ne("").all()
    assert frame["description"].equals(frame["user_char"])
    assert set(frame["user_type"]) == {
        "good", "bad", "bad_leader", "bad_member"
    }


def test_runtime_persona_retrieval_is_agent_scoped_and_includes_summary():
    profiles = _profiles()
    chunks = [
        {"agent_id": index, "section": "summary", "text": profile["Summary"]}
        for index, profile in enumerate(profiles)
    ]
    index = rag_engine.PersonaIndex(chunks_by_agent={
        item["agent_id"]: [item] for item in chunks
    })
    memories = rag_engine.get_agent_specific_memory(
        collection=index,
        agent_id=0,
        current_topic="currency exchange rates and economic policy",
        top_k=3,
    )

    assert len(index.chunks_by_agent) == len(profiles)
    assert memories
    assert all(memory["agent_id"] == 0 for memory in memories)
    assert any(memory["section"] == "summary" for memory in memories)


def test_oasis_creates_agents_with_deeppersona_profiles(tmp_path, monkeypatch):
    profiles = _profiles()
    monkeypatch.setattr(persona_adapter, "NUM_BAD_LEADER", 1)
    monkeypatch.setattr(persona_adapter, "NUM_BAD_MEMBER", 1)
    monkeypatch.setattr(persona_adapter, "NUM_BAD", 1)
    frame = persona_adapter.build_agent_dataframe(
        profiles,
        tweet_pool={"good": [], "bad": []},
    )
    csv_path = tmp_path / "deeppersona_agents.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")

    async def load_agents():
        channel = Channel()
        platform = Platform(":memory:", channel, recsys_type="twitter")
        graph = await generate_agents(
            agent_info_path=str(csv_path),
            twitter_channel=channel,
            inference_channel=Channel(),
            detection_inference_channel=None,
            start_time=0,
            recsys_type="twitter",
            twitter=platform,
            num_agents=len(frame),
            cfgs=[{
                "model_type": "llama-3",
                "num": len(frame),
                "temperature": 0.0,
            }],
            is_openai_model=False,
        )
        return platform, list(graph.get_agents())

    platform, agents = asyncio.run(load_agents())
    assert len(agents) == len(profiles)

    first_agent = agents[0][1]
    persona = first_agent.user_info.profile["other_info"]["user_profile"]
    assert persona == profiles[0]["Summary"].strip()
    assert persona in first_agent.system_message.content
    assert platform.db.execute("SELECT COUNT(*) FROM user").fetchone()[0] == len(
        profiles
    )
    assert platform.db.execute(
        "SELECT bio = user_char FROM user WHERE user_id = 0"
    ).fetchone()[0] == 1
