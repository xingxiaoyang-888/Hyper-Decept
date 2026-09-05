"""Resumable parallel HTTP Range downloader for large public datasets."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import urllib.request


def fetch(url: str, output: Path, total: int, chunk_size: int, workers: int, retries: int, proxy: str | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    state_path = part.with_suffix(part.suffix + ".json")
    chunks = [(start, min(total - 1, start + chunk_size - 1)) for start in range(0, total, chunk_size)]
    state = {str(start): False for start, _ in chunks}
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text())
            for key in state:
                state[key] = bool(previous.get(key, False))
        except Exception:
            pass
    with part.open("ab") as handle:
        handle.truncate(total)

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    ) if proxy else urllib.request.build_opener()

    def one(item):
        start, end = item
        key = str(start)
        if state[key]:
            return key, True
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                with opener.open(request, timeout=120) as response:
                    payload = response.read()
                    expected = end - start + 1
                    if len(payload) != expected:
                        raise RuntimeError(f"range {start}-{end}: got {len(payload)}, expected {expected}")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                with part.open("r+b") as handle:
                    handle.seek(start)
                    handle.write(payload)
                return key, True
            except Exception:
                if attempt + 1 == retries:
                    raise
                time.sleep(min(2 ** attempt, 20))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in chunks]
        done = 0
        for future in as_completed(futures):
            key, ok = future.result()
            state[key] = ok
            done += 1
            if done % max(1, workers) == 0 or done == len(chunks):
                state_path.write_text(json.dumps(state, sort_keys=True))
                print(f"completed_chunks={done}/{len(chunks)}", flush=True)
    if not all(state.values()):
        raise RuntimeError("download incomplete")
    part.replace(output)
    state_path.unlink(missing_ok=True)
    print(f"completed={output} bytes={output.stat().st_size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total", required=True, type=int)
    parser.add_argument("--chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    fetch(args.url, args.output, args.total, args.chunk_size, args.workers, args.retries, args.proxy)


if __name__ == "__main__":
    main()
