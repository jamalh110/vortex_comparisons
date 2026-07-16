#!/usr/bin/env bash
# Start one node in the tested pipeline1 Ray layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"
ROLE="${1:-}"

NIXL_SITE="$(python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("nixl_cu12")
if spec is None or spec.origin is None:
    raise SystemExit("nixl_cu12 is not installed")
print(Path(spec.origin).parent.parent)
PY
)"
NIXL_LIBS="${NIXL_SITE}/nixl_cu12.libs"

export PATH="/usr/local/cuda-12.4/bin:/usr/local/bin:${PATH}"
export PYTHONPATH="${WORK_ROOT}/FLMR/third_party/ColBERT:${WORK_ROOT}/FLMR:${PYTHONPATH:-}"
export UCX_NET_DEVICES="${RDMA_DEVICE}:${RDMA_PORT}"
export UCX_IB_GID_INDEX="${ROCE_GID_INDEX}"
export UCX_MODULE_DIR="${NIXL_LIBS}/ucx"
export UCX_TLS="${UCX_TLS:-rc_x,ud_x,tcp,sm,self,cuda_copy,cuda_ipc}"
export LD_LIBRARY_PATH="${NIXL_LIBS}:/usr/local/cuda-12.4/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export NCCL_IB_HCA="${RDMA_DEVICE}"
export NCCL_SOCKET_IFNAME="${RDMA_NETDEV}"
export RAY_rdt_fetch_fail_timeout_milliseconds=180000

mkdir -p "${LOG_ROOT}"
chmod -R a+rwX "${LOG_ROOT}"
ray stop --force 2>/dev/null || true
sleep 2

DATA_IP="$(hostname -I | awk '{for (i=1;i<=NF;i++) if ($i ~ /^10\\.10\\.1\\./) {print $i; exit}}')"
[[ -n "${DATA_IP}" ]] || {
  echo "Could not locate a 10.10.1.x data-plane address." >&2
  exit 1
}

case "${ROLE}" in
  head)
    ray start --head \
      --port=6379 \
      --dashboard-host=0.0.0.0 \
      --metrics-export-port=8080 \
      --node-ip-address="${HEAD_IP}" \
      --resources='{"deploy_abcd": 1}' \
      --num-gpus=1
    ;;
  stepb)
    ray start \
      --address="${HEAD_IP}:6379" \
      --node-ip-address="${DATA_IP}" \
      --resources='{"deploy_abcd": 1}' \
      --num-gpus=1
    ;;
  mig)
    mapfile -t mig_uuids < <(
      nvidia-smi -L |
        sed -n 's/.*UUID: \\(MIG-[^)]*\\)).*/\\1/p'
    )
    [[ "${#mig_uuids[@]}" -eq 4 ]] || {
      echo "Expected four MIG UUIDs, found ${#mig_uuids[@]}." >&2
      nvidia-smi -L >&2
      exit 1
    }
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${mig_uuids[*]}")"
    ray start \
      --address="${HEAD_IP}:6379" \
      --node-ip-address="${DATA_IP}" \
      --resources='{"deploy_e": 4}' \
      --num-gpus=4
    ;;
  *)
    echo "Usage: $0 {head|stepb|mig}" >&2
    exit 2
    ;;
esac

ray status

