"""Pre-download and verify all Hugging Face models used by feature extraction.

Run this once on the server through the configured network/proxy.  Training
then uses ``TRANSFORMERS_OFFLINE=1`` and ``HF_HUB_OFFLINE=1`` so an experiment
cannot silently change because a model was updated or an API became available.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


MODEL_SPECS = (
    ("text_embedding", "sentence_transformers", "all-mpnet-base-v2"),
    ("empathy_emotion", "transformers", "SamLowe/roberta-base-go_emotions"),
    ("empathy_perplexity", "transformers", "gpt2"),
    ("dark_triad_nli", "transformers", "facebook/bart-large-mnli"),
    ("contagion_embedding", "sentence_transformers", "all-mpnet-base-v2"),
    ("volatility_emotion", "transformers", "SamLowe/roberta-base-go_emotions"),
    ("dark_triad_nli_fallback", "transformers", "microsoft/deberta-v3-base-mnli"),
    ("empathy_spacy", "spacy", "en_core_web_sm"),
)


def _load_model(kind: str, model_name: str):
    if kind == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    if kind == "spacy":
        import spacy

        try:
            return spacy.load(model_name)
        except OSError:
            # spaCy models are installed as packages rather than HF snapshots.
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name]
            )
            return spacy.load(model_name)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return {"tokenizer": tokenizer, "model": model}


def prepare_cache(cache_dir: str | Path, specs: Iterable[tuple] = MODEL_SPECS) -> dict:
    target = Path(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = []
    previous_home = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(target)
    try:
        for purpose, kind, model_name in specs:
            loaded = _load_model(kind, model_name)
            del loaded
            results.append({
                "purpose": purpose,
                "loader": kind,
                "model_name": model_name,
                "status": "ready",
            })
    finally:
        if previous_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = previous_home
    manifest = {
        "schema_version": "hyperdecept.model-cache.v1",
        "cache_dir": str(target.resolve()),
        "models": results,
        "transformers_offline_after_prepare": True,
    }
    (target / "model_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HyperDecept HF model cache")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--proxy",
        help="Optional HTTP(S) proxy URL used only while downloading models",
    )
    args = parser.parse_args()
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    print(json.dumps(prepare_cache(args.cache_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
