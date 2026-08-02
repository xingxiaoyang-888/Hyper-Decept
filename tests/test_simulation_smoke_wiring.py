import importlib.util
import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = ROOT / "MultiAgent4Collusion-master"
for value in (str(ROOT), str(FRAMEWORK_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)


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


def test_inference_manager_supports_multiple_slots_per_endpoint(monkeypatch):
    module = _load_module(
        "parallel_inference_manager",
        ROOT / "MultiAgent4Collusion-master/oasis/inference/inference_manager.py",
    )
    endpoints = []

    class FakeThread:
        def __init__(self, *, server_url, **kwargs):
            endpoints.append(server_url)

    monkeypatch.setattr(module, "InferenceThread", FakeThread)
    manager = object.__new__(module.InferencerManager)
    manager.threads = {}
    manager.parallel_per_endpoint = 1
    manager.max_tokens = 128
    manager.workers_by_port = module.defaultdict(list)
    manager._initialize_threads(
        [{"host": "127.0.0.1", "ports": [11434], "parallel": 4}],
        "qwen",
        "openai",
        [],
    )
    assert endpoints == ["http://127.0.0.1:11434/v1"] * 4
    assert len(manager.threads) == 4
    assert len(manager.workers_by_port[11434]) == 4


def test_busy_timed_out_inference_slot_is_not_reused(monkeypatch):
    module = _load_module(
        "timeout_safe_inference_manager",
        ROOT / "MultiAgent4Collusion-master/oasis/inference/inference_manager.py",
    )
    memory = module.SharedMemory(Busy=True, last_active=0)
    thread = type("FakeThread", (), {"shared_memory": memory})()
    manager = object.__new__(module.InferencerManager)
    manager.threads = {8000: thread}
    manager.workers_by_port = module.defaultdict(list, {8000: [8000]})
    manager.port_manager = type(
        "FakePortManager",
        (),
        {"get_ports_for_agent": lambda self, agent_id: [8000]},
    )()
    manager.lock = asyncio.Lock()
    manager.timeout = 1
    monkeypatch.setattr(module.time, "time", lambda: 10.0)

    selected_thread, worker_id = asyncio.run(
        manager._find_available_thread(agent_id=7)
    )

    assert selected_thread is None
    assert worker_id is None
    assert thread.shared_memory is memory
    assert memory.timeout_warned is True


def test_p2_smoke_runner_declares_three_sources_and_separate_validation():
    source = (ROOT / "scripts" / "run_p2_smoke.py").read_text(encoding="utf-8")
    assert 'import json' in source
    assert '--twibot-manifest' in source
    assert '--mgtab-manifest' in source
    assert '--simulation-manifest' in source
    assert '--report-dir' in source
    assert 'node_split="train"' in source
    assert 'node_split="validation"' in source
    assert 'trainer.train_step(twibot_train, synthetic_batch)' in source
    assert 'trainer.train_step(mgtab_train, synthetic_batch)' in source
    assert 'evaluate_bot_batch(\n            model, twibot_validation' in source
    assert 'evaluate_bot_batch(\n            model, mgtab_validation' in source
    assert 'build_p2_smoke_report(' in source


def test_p2_smoke_initializes_cuda_before_resetting_peak_stats(monkeypatch):
    module = _load_module(
        "p2_smoke_cuda_init",
        ROOT / "scripts/run_p2_smoke.py",
    )
    calls = []

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "set_device",
        lambda device: calls.append(("set_device", device)),
    )
    monkeypatch.setattr(
        module.torch,
        "empty",
        lambda size, *, device: calls.append(("empty", size, str(device))),
    )
    monkeypatch.setattr(
        module.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", device)),
    )

    module._initialize_cuda_memory_tracking(module.torch.device("cuda:0"))

    assert calls == [
        ("set_device", 0),
        ("empty", 0, "cuda:0"),
        ("reset", 0),
    ]
