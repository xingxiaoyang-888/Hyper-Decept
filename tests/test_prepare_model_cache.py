import json

from scripts.prepare_model_cache import prepare_cache


def test_prepare_cache_writes_a_reproducible_model_manifest(tmp_path, monkeypatch):
    def fake_loader(kind, model_name):
        return object()

    monkeypatch.setattr("scripts.prepare_model_cache._load_model", fake_loader)
    manifest = prepare_cache(
        tmp_path / "hf",
        specs=(
            ("text_embedding", "sentence_transformers", "test/text"),
            ("dark_triad_nli", "transformers", "test/nli"),
        ),
    )
    saved = json.loads(
        (tmp_path / "hf" / "model_cache_manifest.json").read_text(encoding="utf-8")
    )
    assert saved == manifest
    assert len(saved["models"]) == 2
    assert all(item["status"] == "ready" for item in saved["models"])
