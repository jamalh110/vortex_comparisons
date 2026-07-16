#!/usr/bin/env python3
"""Render a portable Serve config from the repository templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("baseline", "rdma"), required=True)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--hf-datasets-cache", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.input.read_text())
    application = config["applications"][0]
    env = application.setdefault("runtime_env", {}).setdefault("env_vars", {})
    env.update(
        {
            "PYTHONPATH": (
                f"{args.pipeline_root}:"
                f"{args.work_root}/FLMR/third_party/ColBERT:"
                f"{args.work_root}/FLMR"
            ),
            "PATH": (
                "/usr/local/cuda-12.4/bin:/usr/local/bin:/usr/bin:/bin"
            ),
            "LD_LIBRARY_PATH": (
                "/usr/local/cuda-12.4/lib64:/usr/local/lib"
            ),
            "DATA_ROOT": args.data_root,
            "LOG_ROOT": args.log_root,
            "HF_HOME": args.hf_home,
            "HF_DATASETS_CACHE": args.hf_datasets_cache,
        }
    )

    if args.mode == "rdma":
        nixl_libs = (
            "/usr/local/lib/python3.10/dist-packages/nixl_cu12.libs"
        )
        env.update(
            {
                "LD_LIBRARY_PATH": (
                    f"{nixl_libs}:/usr/local/cuda-12.4/lib64:"
                    "/usr/local/lib"
                ),
                "UCX_MODULE_DIR": f"{nixl_libs}/ucx",
                "UCX_NET_DEVICES": "mlx5_0:1",
                "UCX_IB_GID_INDEX": "3",
                "UCX_TLS": "rc_x,cuda_copy,self",
                "RAY_rdt_fetch_fail_timeout_milliseconds": "180000",
            }
        )

    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()

