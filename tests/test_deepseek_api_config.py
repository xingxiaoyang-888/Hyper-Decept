import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "MultiAgent4Collusion-master/oasis/inference/inference_thread.py"
    spec = importlib.util.spec_from_file_location("deepseek_thread_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deepseek_uses_secret_from_environment(monkeypatch):
    module = _load_module()
    calls = []

    class FakeBackend:
        _client = object()

    def fake_create(**kwargs):
        calls.append(kwargs)
        return FakeBackend()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setattr(module.ModelFactory, "create", fake_create)
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: kwargs)
    module.InferenceThread(
        model_path="openai",
        server_url="https://api.deepseek.com/v1",
        model_type="deepseek-chat",
        max_tokens=128,
    )
    assert calls[0]["api_key"] == "test-only-key"
    assert calls[0]["url"] == "https://api.deepseek.com/v1"
    assert calls[0]["model_config_dict"]["response_format"] == {
        "type": "json_object"
    }
