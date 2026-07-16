#!/usr/bin/env python3
"""Apply the pinned-stack compatibility patches used by this experiment.

The patches are intentionally strict. If a future package version no longer
contains the expected source, stop and reassess instead of modifying it
blindly.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    source = path.read_text()
    if marker in source:
        print(f"already patched: {path}")
        return
    if old not in source:
        raise RuntimeError(f"Expected block for {marker!r} not found in {path}")
    backup = path.with_suffix(path.suffix + ".pre_cloudlab_rdma")
    if not backup.exists():
        backup.write_text(source)
    path.write_text(source.replace(old, new, 1))
    print(f"patched: {path}")


def package_file(module: str, relative: str = "") -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot locate Python module {module!r}")
    base = Path(spec.origin)
    return base.parent / relative if relative else base


def patch_nixl_wrapper() -> None:
    path = package_file("nixl")
    source = path.read_text()
    marker = "Prefer the torch-matched backend, then fall back"
    if marker in source:
        print(f"already patched: {path}")
        return

    start = source.index("def _load_cuda_backend() -> str:")
    end = source.index("\n\n_pkg =", start)
    replacement = '''def _load_cuda_backend() -> str:
    cuda_major = _get_torch_cuda_major()
    # Prefer the torch-matched backend, then fall back. The tested stack uses
    # cu118 PyTorch with the nixl-cu12 backend and an NVIDIA 550 driver.
    candidates = []
    if cuda_major is not None:
        candidates.append(cuda_major)
    for major in (12, 13, 11):
        if major not in candidates:
            candidates.append(major)
    errors = []
    for major in candidates:
        mod_name = f"nixl_cu{major}"
        try:
            return importlib.import_module(mod_name).__name__
        except ModuleNotFoundError as exc:
            if exc.name != mod_name:
                raise
            errors.append(mod_name)
    raise ImportError(f"No NIXL CUDA backend found; tried {errors}")
'''
    backup = path.with_suffix(path.suffix + ".pre_cloudlab_rdma")
    if not backup.exists():
        backup.write_text(source)
    path.write_text(source[:start] + replacement + source[end:])
    print(f"patched: {path}")


def patch_ray_nixl_listener() -> None:
    path = package_file(
        "ray.experimental.rdt.nixl_tensor_transport",
    )
    old = '        agent_config = nixl_agent_config(backends=["UCX"])'
    new = '''        agent_config = nixl_agent_config(
            backends=["UCX"],
            enable_listen_thread=True,
            listen_port=0,
        )'''
    replace_once(path, old, new, "enable_listen_thread=True")


def patch_serve_replicas() -> None:
    path = package_file("ray.serve._private.deployment_info")
    old = '''            self._cached_actor_def = ray.remote(
                type(
                    self.actor_name,
                    (ReplicaActor,),
                    dict(ReplicaActor.__dict__),
                )
            )'''
    new = '''            # CloudLab RDT patch: receiving Serve replicas need Ray's
            # system tensor-transport concurrency groups.
            self._cached_actor_def = ray.remote(enable_tensor_transport=True)(
                type(
                    self.actor_name,
                    (ReplicaActor,),
                    dict(ReplicaActor.__dict__),
                )
            )'''
    replace_once(
        path,
        old,
        new,
        "ray.remote(enable_tensor_transport=True)(",
    )


def patch_flmr() -> None:
    work_root = Path(os.environ.get("WORK_ROOT", "/users/jamalh11/workspace"))
    indexing = work_root / "FLMR/flmr/indexing.py"
    searching = work_root / "FLMR/flmr/searching.py"

    index_old = '''        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=doc_maxlen,
            total_visible_gpus=nranks if use_gpu else 0,
        )'''
    index_new = '''        try:
            config = ColBERTConfig(
                nbits=nbits,
                doc_maxlen=doc_maxlen,
                total_visible_gpus=nranks if use_gpu else 0,
            )
        except TypeError:
            config = ColBERTConfig(
                nbits=nbits,
                doc_maxlen=doc_maxlen,
                gpus=nranks if use_gpu else 0,
            )'''
    replace_once(indexing, index_old, index_new, "except TypeError:")

    search_old = '''        config = ColBERTConfig(
            total_visible_gpus=total_visible_gpus,
        )'''
    search_new = '''        try:
            config = ColBERTConfig(
                total_visible_gpus=total_visible_gpus,
            )
        except TypeError:
            config = ColBERTConfig(
                gpus=total_visible_gpus,
            )'''
    replace_once(searching, search_old, search_new, "except TypeError:")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flmr-only", action="store_true")
    args = parser.parse_args()

    patch_flmr()
    if not args.flmr_only:
        patch_nixl_wrapper()
        patch_ray_nixl_listener()
        patch_serve_replicas()


if __name__ == "__main__":
    main()

