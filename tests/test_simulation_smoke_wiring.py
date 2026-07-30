import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rag_engine_honors_episode_chunks_environment(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps([
        {"agent_id": 403, "section": "summary", "text": "episode persona"}
    ]), encoding="utf-8")
    monkeypatch.setenv("DEEP_PERSONA_CHUNKS_PATH", str(chunks))
    module = _load_module("episode_rag_engine", ROOT / "deeppersona_ai/rag_engine.py")
    index = module.load_vector_store()
    assert index.chunks_by_agent[403][0]["text"] == "episode persona"


def test_inference_manager_uses_declared_endpoint(monkeypatch):
    module = _load_module(
        "episode_inference_manager",
        ROOT / "MultiAgent4Collusion-master/oasis/inference/inference_manager.py",
    )
    endpoints = []

    class FakeThread:
        def __init__(self, *, server_url, **kwargs):
            endpoints.append(server_url)

    monkeypatch.setattr(module, "InferenceThread", FakeThread)
    manager = object.__new__(module.InferencerManager)
    manager.threads = {}
    manager._initialize_threads(
        [{"host": "127.0.0.1", "ports": [11434]}],
        "qwen3.5:9b",
        "openai",
        [],
    )
    assert endpoints == ["http://127.0.0.1:11434/v1"]
    assert set(manager.threads) == {11434}
