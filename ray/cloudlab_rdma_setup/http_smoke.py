#!/usr/bin/env python3
"""Send one real cached EVQA request to the deployed pipeline."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import requests


def main() -> None:
    data_root = Path(os.environ.get("DATA_ROOT", "/mydata"))
    url = os.environ.get("PIPELINE_URL", "http://10.10.1.2:8000/")
    cache = data_root / "ds_test.pkl"
    if not cache.exists():
        raise RuntimeError(
            f"{cache} does not exist. Run `cluster.sh warm-locust` first."
        )

    with cache.open("rb") as handle:
        requests_cache = pickle.load(handle)
    payload = requests_cache[0][0]
    started = time.time()
    response = requests.post(
        url,
        data=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "x-requestid": "CLOUDLAB-SMOKE",
        },
        timeout=240,
    )
    response.raise_for_status()
    result = response.json()
    if result and result[0] == "error":
        raise RuntimeError(f"Pipeline returned an error: {result}")
    print(
        f"HTTP_SMOKE_OK elapsed={time.time() - started:.3f}s "
        f"bytes={len(response.content)} result={result[:1]}"
    )


if __name__ == "__main__":
    main()

