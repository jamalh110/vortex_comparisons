#!/usr/bin/env python3
"""Cross-node GPU NIXL smoke test for the running Ray cluster."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import ray
import torch


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


@ray.remote(num_gpus=1, resources={"deploy_abcd": 1}, runtime_env=RUNTIME_ENV)
class Producer:
    @ray.method(tensor_transport="nixl")
    def produce(self):
        tensor = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        return tensor, float(tensor.sum())

    def node(self):
        return ray.util.get_node_ip_address()


@ray.remote(
    enable_tensor_transport=True,
    num_gpus=1,
    resources={"deploy_abcd": 1},
    runtime_env=RUNTIME_ENV,
)
class Consumer:
    def consume(self, payload):
        tensor, expected = payload
        torch.cuda.synchronize()
        actual = float(tensor.sum())
        assert tensor.is_cuda
        assert abs(actual - expected) < 1e-2
        return actual

    def node(self):
        return ray.util.get_node_ip_address()


def main() -> None:
    ray.init(address="auto")
    producer = Producer.remote()
    consumer = Consumer.remote()
    source = ray.get(producer.node.remote())
    destination = ray.get(consumer.node.remote())
    if source == destination:
        raise RuntimeError(
            "Producer and consumer landed on the same node; ensure two "
            "deploy_abcd GPUs are free before running this probe."
        )
    result = ray.get(
        consumer.consume.remote(producer.produce.remote()),
        timeout=90,
    )
    print(f"RDT_SMOKE_OK source={source} destination={destination} sum={result}")
    ray.shutdown()


if __name__ == "__main__":
    main()

