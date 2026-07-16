# CloudLab Ray + RoCE + NIXL Setup

This package recreates the five-node pipeline1 experiment on Wisconsin
CloudLab `d7525` machines. It is self-contained except for the
`vortex_comparisons` repository, public package/model downloads, and the
CloudLab reservation itself.

Read `../agents.md` before changing the RDT implementation. Ray Serve 2.56
does not natively expose RDT; the all-Serve RDMA mode uses a compatibility
patch and is slower than the baseline.

## Tested layout

| Alias | Data IP | Purpose | GPU |
|---|---:|---|---|
| node0 | 10.10.1.1 | Locust and orchestration | none |
| node1 | 10.10.1.2 | Ray head, StepB | A30 |
| node2 | 10.10.1.3 | StepB | A30 |
| node3 | 10.10.1.4 | StepB | A30 |
| node4 | 10.10.1.5 | StepA, StepD, three StepE replicas | A30, 4x `1g.6gb` MIG |

The control NIC has a public `128.105.x.x` address. The experiment data NIC is
`enp65s0np0` on `10.10.1.0/24`, MTU 9000. `mlx5_0` is a 200-Gb/s Ethernet
device, so this is RoCEv2 rather than InfiniBand. The IPv4 RoCEv2 GID is index
3.

## Tested software

- Ubuntu 22.04.2, Python 3.10.12
- MLNX OFED 23.10-3.2.2.0
- NVIDIA driver 550.163.01 and CUDA toolkit 12.4
- PyTorch 2.1.2+cu118
- Ray 2.56.0
- NIXL/nixl-cu12 1.3.1
- protobuf 4.25.9
- transformers 4.44.2, tokenizers 0.19.1, datasets 2.21.0
- NumPy 1.26.4
- Faiss 1.9.0 built with GPU support for compute capability 8.0
- FLMR commit `c5db04b5d4e288bd9d3c8594ad285f70c1aa8831`

Do not casually upgrade these packages. In particular, protobuf 7.x breaks
this Ray Serve installation.

## Before running

1. Reserve five `d7525` nodes on one experiment LAN.
2. Clone `vortex_comparisons` on node0 at
   `/users/<user>/vortex_comparisons`.
3. Edit `cluster.env` for the new username, node aliases, and IP addresses.
4. Ensure node0 can SSH to every alias without a password.

`/users` and `/mydata` are local disks in this experiment, not shared NFS.
The scripts therefore synchronize code and stream data explicitly.

Generate an orchestration key if needed:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

Install only the public key in each node's `authorized_keys`. Do not copy a
private key into this repository.

Run all cluster commands from this directory on node0:

```bash
cd /users/jamalh11/vortex_comparisons/ray/cloudlab_rdma_setup
source cluster.env
```

## Complete installation order

### 1. Check access and synchronize the repository

```bash
./cluster.sh check-ssh
./cluster.sh sync-code
```

### 2. Install base packages

```bash
./cluster.sh base
```

### 3. Install Mellanox OFED

First check whether the CloudLab image already has the tested OFED:

```bash
ssh node1 ofed_info -s
```

If it reports `MLNX_OFED_LINUX-23.10-3.2.2.0`, skip installation. Otherwise:

```bash
./cluster.sh ofed
for node in node1 node2 node3 node4; do ssh "$node" sudo reboot; done
```

Wait for all nodes to return before continuing.

### 4. Install NVIDIA driver and CUDA

```bash
./cluster.sh gpu
for node in node1 node2 node3 node4; do ssh "$node" sudo reboot; done
```

Verify `nvidia-smi` and `/usr/local/cuda-12.4/bin/nvcc` on every GPU node.

### 5. Install the pinned Python stack and FLMR

```bash
./cluster.sh sync-code
./cluster.sh python
```

This also applies:

- the FLMR `total_visible_gpus`/`gpus` compatibility change;
- NIXL's cu118-PyTorch to nixl-cu12 fallback;
- the Ray NIXL listen-thread change;
- the Ray Serve replica tensor-transport change.

The package patches are version-specific. `runtime_patch.py` creates backups
next to modified files and fails if the expected Ray/NIXL source is absent.

### 6. Build Faiss GPU

```bash
./cluster.sh faiss
```

This builds CMake if necessary, then builds Faiss 1.9.0 on all four GPU nodes.
Expect this step to take substantial CPU time.

### 7. Configure MIG

```bash
./cluster.sh mig
ssh node4 nvidia-smi -L
```

On a fresh node, the first command enables MIG mode and exits with a reboot
message. Reboot node4 and run `./cluster.sh mig` again to create the instances.

The start script discovers MIG UUIDs dynamically. Do not copy UUIDs from an
old reservation.

### 8. Prepare and distribute data

Build the models, EVQA parquet tables, image tree, step checkpoints, and
ColBERT index on node1:

```bash
./cluster.sh prepare-data
```

This uses Hugging Face because the old Azure blob hostname is unavailable.
The image downloads are large. If the `inat.zip` download is corrupt, remove
the partial archive and retry; `prepare_data.py` also prints a `zip -FF`
repair command.

Stream the completed data tree from node1 to all other nodes:

```bash
./cluster.sh sync-data
```

Required final paths include:

```text
/mydata/PreFLMR_ViT-L/
/mydata/clip-vit-large-patch14/
/mydata/EVQA/models/
/mydata/EVQA/EVQA_data/
/mydata/EVQA/EVQA_passages/
/mydata/EVQA/google-landmark/
/mydata/EVQA/inat/
/mydata/EVQA/index/EVQA_train_split/indexes/EVQA_PreFLMR_ViT-L.nbits=8/
```

Locust creates `/mydata/ds_test.pkl` on node0 the first time it imports
`locustfile.py`. The cache is approximately 6.8 GB.

## Validate the fabric

Before Ray, verify:

```bash
ssh node1 'rdma link; show_gids; ip link show enp65s0np0'
./cluster.sh verbs
```

Expected properties:

- `mlx5_0/1` is ACTIVE;
- link layer is Ethernet;
- GID index 3 maps to `10.10.1.x` and says RoCE v2;
- MTU is 9000;
- `ib_write_bw` is around 150 Gb/s for the tested 64-KiB transfer.

PFC was disabled on the tested reservation. That is not ideal for a congested
lossless RoCE fabric, but the isolated verbs and NIXL tests succeeded.

## Start and validate Ray RDT

```bash
./cluster.sh start
./cluster.sh rdt
./cluster.sh serve-rdt
```

Run both RDT smoke tests before the pipeline because each reserves two full
`deploy_abcd` GPUs. A successful result contains:

```text
RDT_SMOKE_OK source=10.10.1.x destination=10.10.1.y
SERVE_RDT_SMOKE_OK consumer=10.10.1.y
```

Ray environment variables are set by `start_ray.sh`:

```text
UCX_NET_DEVICES=mlx5_0:1
UCX_IB_GID_INDEX=3
UCX_MODULE_DIR=<site-packages>/nixl_cu12.libs/ucx
UCX_TLS=rc_x,ud_x,tcp,sm,self,cuda_copy,cuda_ipc
RAY_rdt_fetch_fail_timeout_milliseconds=180000
```

## Deploy pipeline1

Baseline, recommended for current performance:

```bash
./cluster.sh deploy baseline
```

Experimental all-Serve NIXL workaround:

```bash
./cluster.sh deploy rdma
```

Both commands shut down the existing Serve controller before deployment. This
forces code reload and ensures the configured `0.0.0.0:8000` HTTP binding is
used. `render_serve_config.py` injects the paths and environment from
`cluster.env`, so a new username does not require editing the YAML templates.

Check:

```bash
ssh node1 serve status
./cluster.sh warm-locust
./cluster.sh smoke-http
```

The HTTP smoke uses a real pickled EVQA request; a plain `curl /` is not a
valid application test.

## Run Locust

Five-minute, 32-user run:

```bash
./cluster.sh locust 32 5m locust_32
```

The wrapper prebuilds the Locust cache and clears stage timing logs before
starting the measured run.

CSV files are written on node0 under
`/users/jamalh11/ray_experiment_logs/`.

Known results:

| Mode | Steady throughput | Median | Average |
|---|---:|---:|---:|
| Baseline Serve/object store | ~93.4 req/s | 340 ms | 344 ms |
| All-boundary Serve/NIXL workaround | ~60.5 req/s | 500 ms | 567 ms |

The RDMA result proves functionality, not a performance win. See
`../agents.md` for the causes and recommended native Ray Core actor design.

Stop everything cleanly with:

```bash
./cluster.sh stop
```

## Troubleshooting

### NIXL imports the wrong CUDA backend

PyTorch reports CUDA 11.8, while the available NIXL wheel is cu12. Re-run:

```bash
./cluster.sh patch
```

### RDT times out after 60 seconds

Confirm the receiving actor has `enable_tensor_transport=True`. For Serve,
confirm the `runtime_patch.py` Serve patch was applied before starting the
Serve controller. Also confirm Ray workers inherited all UCX variables.

### Serve replicas cannot import `main_rdma`

Ensure `PYTHONPATH` in `serve_config_best_mig_rdma.yaml` points to the local
pipeline1 and FLMR paths on every node, then run `sync-code` and deploy again.

### Serve fails in protobuf descriptors

```bash
sudo python3 -m pip install --force-reinstall protobuf==4.25.9
```

Restart the Ray cluster afterward.

### RDT refs disappear under load

Nested RDT refs inside Serve responses are not a native supported path. The
current workaround keeps bounded owner-side ref queues. If
`ReferenceCountingAssertionError` returns, use the baseline or move the
pipeline stages to native Ray Core actors.

