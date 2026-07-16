#!/usr/bin/env bash
# Cluster-wide orchestration. Run from node0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"
ACTION="${1:-}"

ray_nodes=(${RAY_NODES})
all_nodes=(${ALL_NODES})

run_parallel() {
  local command="$1"
  shift
  local pids=()
  for node in "$@"; do
    echo "[${node}] ${command}"
    ssh -o BatchMode=yes "${node}" "${command}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

sync_code() {
  for node in "${ray_nodes[@]}"; do
    ssh "${node}" "mkdir -p '${REPO_ROOT}'"
    rsync -az \
      --exclude '.git/' \
      --exclude '__pycache__/' \
      "${REPO_ROOT}/" "${node}:${REPO_ROOT}/"
  done
}

check_ssh() {
  for node in "${all_nodes[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${node}" \
      'echo "$(hostname) data=$(hostname -I)"'
  done
}

bootstrap_setup() {
  for node in "${ray_nodes[@]}"; do
    tar -C "${SETUP_ROOT}" -cf - . |
      ssh "${node}" \
        "mkdir -p '${SETUP_ROOT}' && tar -C '${SETUP_ROOT}' -xf -"
  done
}

install_base() {
  bootstrap_setup
  run_parallel \
    "bash '${SETUP_ROOT}/setup_node.sh' base" \
    "${all_nodes[@]}"
  sync_code
}

install_ofed() {
  run_parallel \
    "bash '${SETUP_ROOT}/setup_node.sh' ofed" \
    "${ray_nodes[@]}"
  echo "Reboot node1-node4 before continuing."
}

install_gpu() {
  run_parallel \
    "bash '${SETUP_ROOT}/setup_node.sh' gpu" \
    "${ray_nodes[@]}"
  echo "Reboot node1-node4 before continuing."
}

install_python() {
  run_parallel \
    "bash '${SETUP_ROOT}/setup_node.sh' python ray" \
    "${ray_nodes[@]}"
  ssh "${LOCUST_NODE}" \
    "bash '${SETUP_ROOT}/setup_node.sh' python locust"
}

install_faiss() {
  run_parallel \
    "bash '${SETUP_ROOT}/setup_node.sh' faiss ray" \
    "${ray_nodes[@]}"
}

patch_runtime() {
  run_parallel \
    "sudo WORK_ROOT='${WORK_ROOT}' python3 '${SETUP_ROOT}/runtime_patch.py'" \
    "${ray_nodes[@]}"
}

setup_mig() {
  ssh "${MIG_NODE}" "bash '${SETUP_ROOT}/setup_mig.sh'"
}

start_cluster() {
  patch_runtime
  ssh "${HEAD_NODE}" "bash '${SETUP_ROOT}/start_ray.sh' head"
  for node in node2 node3; do
    ssh "${node}" "bash '${SETUP_ROOT}/start_ray.sh' stepb"
  done
  ssh "${MIG_NODE}" "bash '${SETUP_ROOT}/start_ray.sh' mig"
  ssh "${HEAD_NODE}" ray status
}

stop_cluster() {
  ssh "${HEAD_NODE}" "serve shutdown -y 2>/dev/null || true"
  run_parallel "ray stop --force 2>/dev/null || true" "${ray_nodes[@]}"
}

clear_logs() {
  run_parallel \
    "mkdir -p '${LOG_ROOT}'; find '${LOG_ROOT}' -maxdepth 1 -type f -delete" \
    "${ray_nodes[@]}"
}

validate_verbs() {
  local port=18515
  ssh node2 "pkill -x ib_write_bw 2>/dev/null || true"
  ssh -f node2 \
    "ib_write_bw -d '${RDMA_DEVICE}' -x '${ROCE_GID_INDEX}' -F \
      --report_gbits -p '${port}' >/tmp/ib_write_bw_server.log 2>&1"
  sleep 2
  ssh node1 \
    "ib_write_bw -d '${RDMA_DEVICE}' -x '${ROCE_GID_INDEX}' -F \
      --report_gbits -p '${port}' -n 2000 '${NODE2_IP}'"
  ssh node2 "cat /tmp/ib_write_bw_server.log"
}

validate_rdt() {
  echo "This probe needs two free deploy_abcd GPUs. Stop Serve first."
  ssh "${HEAD_NODE}" \
    "PYTHONUNBUFFERED=1 \
     UCX_NET_DEVICES='${RDMA_DEVICE}:${RDMA_PORT}' \
     UCX_IB_GID_INDEX='${ROCE_GID_INDEX}' \
     python3 '${SETUP_ROOT}/rdt_smoke.py'"
}

validate_serve_rdt() {
  echo "This probe needs two free deploy_abcd GPUs. Stop pipeline Serve first."
  ssh "${HEAD_NODE}" \
    "PYTHONUNBUFFERED=1 \
     UCX_NET_DEVICES='${RDMA_DEVICE}:${RDMA_PORT}' \
     UCX_IB_GID_INDEX='${ROCE_GID_INDEX}' \
     python3 '${SETUP_ROOT}/serve_rdt_smoke.py'"
}

prepare_data() {
  ssh "${HEAD_NODE}" \
    "DATA_ROOT='${DATA_ROOT}' WORK_ROOT='${WORK_ROOT}' \
     HF_HOME='${HF_HOME}' HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \
     PYTHONPATH='${WORK_ROOT}/FLMR/third_party/ColBERT:${WORK_ROOT}/FLMR' \
     python3 '${SETUP_ROOT}/prepare_data.py'"
}

sync_data() {
  local source="${HEAD_NODE}"
  for target in node0 node2 node3 node4; do
    echo "Streaming /mydata from ${source} to ${target}"
    ssh "${target}" "sudo mkdir -p '${DATA_ROOT}'"
    ssh "${source}" \
      "sudo tar -C '${DATA_ROOT}' -cf - PreFLMR_ViT-L clip-vit-large-patch14 EVQA" |
      ssh "${target}" "sudo tar -C '${DATA_ROOT}' -xf -"
    ssh "${target}" \
      "sudo chown -R '${CLOUDLAB_USER}:$(id -gn)' '${DATA_ROOT}'"
  done
}

deploy() {
  local mode="${2:-baseline}"
  local config
  local rendered="/tmp/cloudlab_serve_${mode}.yaml"
  case "${mode}" in
    baseline) config="serve_config_best_mig.yaml" ;;
    rdma) config="serve_config_best_mig_rdma.yaml" ;;
    *)
      echo "Mode must be baseline or rdma." >&2
      exit 2
      ;;
  esac
  sync_code
  patch_runtime
  python3 "${SETUP_ROOT}/render_serve_config.py" \
    "${PIPELINE_ROOT}/${config}" \
    "${rendered}" \
    --mode "${mode}" \
    --pipeline-root "${PIPELINE_ROOT}" \
    --work-root "${WORK_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --hf-home "${HF_HOME}" \
    --hf-datasets-cache "${HF_DATASETS_CACHE}"
  scp -q "${rendered}" "${HEAD_NODE}:${rendered}"
  ssh "${HEAD_NODE}" \
    "cd '${PIPELINE_ROOT}' &&
     serve shutdown -y 2>/dev/null || true;
     sleep 3;
     cd '${PIPELINE_ROOT}' &&
     serve deploy '${rendered}'"

  for _ in $(seq 1 90); do
    local status
    status="$(ssh "${HEAD_NODE}" 'timeout 8 serve status 2>/dev/null' || true)"
    if echo "${status}" | grep -q 'status: DEPLOY_FAILED'; then
      echo "${status}"
      return 1
    fi
    if echo "${status}" | grep -q 'status: RUNNING' &&
      ! echo "${status}" |
        grep -Eq 'status: (DEPLOYING|UPDATING|UNHEALTHY)|STARTING:' &&
      [[ "$(echo "${status}" | grep -c 'RUNNING: 3')" -ge 2 ]] &&
      echo "${status}" | grep -q 'RUNNING: 16'; then
      echo "${status}"
      curl -fsS --max-time 5 "http://${HEAD_IP}:8000/" >/dev/null || true
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for Serve." >&2
  return 1
}

run_locust() {
  local users="${2:-32}"
  local duration="${3:-5m}"
  local label="${4:-locust_${users}_$(date +%Y%m%d_%H%M%S)}"
  local output="${USER_HOME}/ray_experiment_logs/${label}"
  warm_locust
  clear_logs
  ssh "${LOCUST_NODE}" "mkdir -p '${USER_HOME}/ray_experiment_logs'"
  ssh "${LOCUST_NODE}" \
    "cd '${PIPELINE_ROOT}' &&
     DATA_ROOT='${DATA_ROOT}' \
     HF_HOME='${HF_HOME}' \
     HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \
     RAY_SERVE_HOSTS='http://${HEAD_IP}:8000,http://${NODE2_IP}:8000,http://${NODE3_IP}:8000,http://${MIG_IP}:8000' \
     PYTHONPATH='${WORK_ROOT}/FLMR/third_party/ColBERT:${WORK_ROOT}/FLMR' \
     locust -f locustfile.py \
       --users '${users}' \
       --spawn-rate 4 \
       --headless \
       --run-time '${duration}' \
       -H 'http://${HEAD_IP}:8000' \
       --csv '${output}'"
}

warm_locust() {
  ssh "${LOCUST_NODE}" \
    "cd '${PIPELINE_ROOT}' &&
     DATA_ROOT='${DATA_ROOT}' \
     HF_HOME='${HF_HOME}' \
     HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \
     PYTHONPATH='${WORK_ROOT}/FLMR/third_party/ColBERT:${WORK_ROOT}/FLMR' \
     python3 -c 'import locustfile; print(\"LOCUST_CACHE_READY\", len(locustfile.bytes_to_send))'"
}

smoke_http() {
  ssh "${LOCUST_NODE}" \
    "DATA_ROOT='${DATA_ROOT}' \
     PIPELINE_URL='http://${HEAD_IP}:8000/' \
     python3 '${SETUP_ROOT}/http_smoke.py'"
}

case "${ACTION}" in
  check-ssh) check_ssh ;;
  sync-code) sync_code ;;
  base) install_base ;;
  ofed) install_ofed ;;
  gpu) install_gpu ;;
  python) install_python ;;
  faiss) install_faiss ;;
  patch) patch_runtime ;;
  mig) setup_mig ;;
  start) start_cluster ;;
  stop) stop_cluster ;;
  clear-logs) clear_logs ;;
  verbs) validate_verbs ;;
  rdt) validate_rdt ;;
  serve-rdt) validate_serve_rdt ;;
  prepare-data) prepare_data ;;
  sync-data) sync_data ;;
  deploy) deploy "$@" ;;
  warm-locust) warm_locust ;;
  smoke-http) smoke_http ;;
  locust) run_locust "$@" ;;
  *)
    cat >&2 <<'EOF'
Usage:
  cluster.sh check-ssh
  cluster.sh sync-code
  cluster.sh base
  cluster.sh ofed
  cluster.sh gpu
  cluster.sh python
  cluster.sh faiss
  cluster.sh patch
  cluster.sh mig
  cluster.sh prepare-data
  cluster.sh sync-data
  cluster.sh start
  cluster.sh stop
  cluster.sh clear-logs
  cluster.sh verbs
  cluster.sh rdt
  cluster.sh serve-rdt
  cluster.sh deploy {baseline|rdma}
  cluster.sh warm-locust
  cluster.sh smoke-http
  cluster.sh locust [users] [duration] [label]
EOF
    exit 2
    ;;
esac

