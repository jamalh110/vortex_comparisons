# Ray Serve, RDT, and CloudLab Notes

## Scope

This directory contains the Ray implementations of the Vortex pipelines. The
current CloudLab experiment is a five-node pipeline1 deployment:

| Node | Data IP | Role |
|---|---:|---|
| node0 | 10.10.1.1 | Locust only |
| node1 | 10.10.1.2 | Ray head and one StepB replica |
| node2 | 10.10.1.3 | Ray worker and one StepB replica |
| node3 | 10.10.1.4 | Ray worker and one StepB replica |
| node4 | 10.10.1.5 | Four A30 `1g.6gb` MIG devices; StepA, StepD, and StepE |

The data network is `enp65s0np0`, MTU 9000, backed by `mlx5_0`. It is
200-Gb/s Ethernet using RoCEv2, not InfiniBand. GID index 3 is the IPv4 RoCEv2
GID. A node1-to-node2 `ib_write_bw` test reached about 151 Gb/s.

## The Ray Serve/RDT problem

Ray 2.56 supports Ray Direct Transport (RDT) natively for Ray Core actors:

```python
@ray.remote
class Producer:
    @ray.method(tensor_transport="nixl")
    def produce(self):
        return cuda_tensor

@ray.remote(enable_tensor_transport=True)
class Consumer:
    def consume(self, tensor):
        ...
```

Ray Serve does not expose this on `DeploymentHandle`. A deployment method is
called inside Serve's `ReplicaActor.handle_request` wrapper, so applying
`@ray.method(tensor_transport="nixl")` to the user deployment method does not
mark the actual Ray actor task. Serve 2.56 also creates `ReplicaActor` without
`enable_tensor_transport=True` and rejects that key in `ray_actor_options`.

Returning a CUDA tensor directly from a deployment therefore uses normal
serialization. A CPU-only Ingress replica can then attempt CUDA
deserialization and crash with SIGBUS.

## Current working compatibility path

`pipeline1/main_rdma.py` and `pipeline1/StepD_rdma.py` publish nested refs:

```python
ref = ray.put(cuda_payload, _tensor_transport="nixl")
return {"rdt_ref": ref}
```

The receiving deployment resolves the nested ref with `ray.get`. The runtime
patch in `cloudlab_rdma_setup/runtime_patch.py` enables tensor transport on
Serve's internal replica actors. Producers retain a bounded queue of refs
because Serve's nested response does not reliably keep the inner RDT ref alive
under load.

This path is functional, but it is a compatibility workaround rather than
native Serve RDT.

## Why the compatibility path is slower

The original CPU/object-store run achieved approximately:

- 93.4 req/s steady state
- 340 ms median
- 344 ms average

The all-boundary RDT workaround achieved approximately:

- 60.5 req/s steady state
- 500 ms median
- 567 ms average

The fabric is not the bottleneck. The workaround adds:

1. One NIXL `ray.put` per request at every producing stage, even when Serve
   computed a batch.
2. An outer Serve response plus an inner RDT ObjectRef.
3. Explicit downstream `ray.get` calls; StepD currently fetches StepA and
   StepB separately.
4. CUDA synchronization and NIXL registration/metadata work for small objects.
5. RDT on local node4 MIG boundaries where network RDMA offers little benefit.

Only StepB-to-StepD is a large cross-node boundary. The other NIXL boundaries
mostly pay fixed control costs without using the 200-Gb/s fabric.

## Potential solutions

### Recommended: Serve ingress plus Ray Core stage actors

Use Serve only for HTTP parsing and response handling. Run StepA/B/D/E as Ray
Core actors with native RDT producer methods and
`enable_tensor_transport=True` consumers. Add an explicit microbatcher per
stage. This removes nested refs, the Serve runtime patch, redundant routing,
and manual keepalive queues.

### Fastest incremental option: hybrid transport

Restore the baseline Serve/object-store path for StepA-to-StepD and
StepD-to-StepE. Use NIXL only for the large cross-node StepB-to-StepD payload.
This is the most likely short path back to baseline performance.

### Additional optimizations

- Fetch independent StepA and StepB inputs concurrently.
- Publish one RDT object per batch and demultiplex by batch index.
- Reuse registered sender and receiver buffers or NIXL memory pools.
- Remove CUDA synchronizations that Ray's NIXL metadata extraction already
  performs.
- Try CUDA IPC for replicas sharing the same MIG device; retain NIXL/RoCE for
  cross-node transfers.
- Compare five-minute runs after a warm-up period, not a cold one-minute run.

Do not assume RDMA must improve this workload. The transferred payload is small
relative to the available network bandwidth, so fixed transport and scheduling
latency can dominate.

## Reproducibility

Use `cloudlab_rdma_setup/README.md`. The package pins the known-good software
stack, prepares data, patches the two Ray 2.56 limitations, configures MIG,
starts the cluster, validates verbs and RDT, deploys Serve, and runs Locust.

The short operational sequence is:

```text
check-ssh -> base -> OFED/reboot -> GPU/reboot -> python -> faiss
-> MIG/reboot-if-requested -> prepare-data -> sync-data
-> start -> verbs -> rdt -> deploy -> warm-locust -> smoke-http -> locust
```

`/users` and `/mydata` are node-local. Always run `sync-code` after edits and
`sync-data` after rebuilding artifacts. Package upgrades can overwrite the
NIXL/Ray source patches; rerun `cluster.sh patch` and restart Ray afterward.

