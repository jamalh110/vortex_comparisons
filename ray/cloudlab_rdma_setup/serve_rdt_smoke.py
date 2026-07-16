#!/usr/bin/env python3
"""Validate the patched Ray Serve 2.56 nested-NIXL compatibility path."""

from __future__ import annotations

import importlib.util
import os
from collections import deque
from pathlib import Path

import ray
import torch
from ray import serve


def runtime_env() -> dict:
    spec = importlib.util.find_spec("nixl_cu12")
    if spec is None or spec.origin is None:
        raise RuntimeError("nixl_cu12 is not installed")
    site = Path(spec.origin).parent.parent
    libs = site / "nixl_cu12.libs"
    return {
        "env_vars": {
            "UCX_MODULE_DIR": str(libs / "ucx"),
            "UCX_NET_DEVICES": os.environ.get(
                "UCX_NET_DEVICES", "mlx5_0:1"
            ),
            "UCX_IB_GID_INDEX": os.environ.get("UCX_IB_GID_INDEX", "3"),
            "UCX_TLS": os.environ.get(
                "UCX_TLS", "rc_x,cuda_copy,self"
            ),
            "LD_LIBRARY_PATH": (
                f"{libs}:/usr/local/cuda-12.4/lib64:/usr/local/lib"
            ),
        }
    }


RUNTIME_ENV = runtime_env()
GPU_OPTIONS = {
    "num_gpus": 1,
    "resources": {"deploy_abcd": 1},
    "runtime_env": RUNTIME_ENV,
}


@serve.deployment(ray_actor_options=GPU_OPTIONS)
class Producer:
    def __init__(self):
        self.refs = deque(maxlen=8)

    def __call__(self):
        tensor = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        expected = float(tensor.sum())
        ref = ray.put(tensor, _tensor_transport="nixl")
        self.refs.append(ref)
        return {"ref": ref, "expected": expected}


@serve.deployment(ray_actor_options=GPU_OPTIONS)
class Consumer:
    def __call__(self, payload):
        tensor = ray.get(payload["ref"])
        torch.cuda.synchronize()
        actual = float(tensor.sum())
        assert tensor.is_cuda
        assert abs(actual - payload["expected"]) < 1e-2
        return actual, ray.util.get_node_ip_address()


@serve.deployment
class Coordinator:
    def __init__(self, producer, consumer):
        self.producer = producer
        self.consumer = consumer

    async def __call__(self):
        payload = await self.producer.remote()
        return await self.consumer.remote(payload)


def main() -> None:
    ray.init(address="auto")
    app = Coordinator.bind(Producer.bind(), Consumer.bind())
    handle = serve.run(
        app,
        name="serve-rdt-smoke",
        route_prefix=None,
    )
    result, consumer_ip = handle.remote().result(timeout_s=90)
    print(f"SERVE_RDT_SMOKE_OK consumer={consumer_ip} sum={result}")
    serve.delete("serve-rdt-smoke")
    ray.shutdown()


if __name__ == "__main__":
    main()

