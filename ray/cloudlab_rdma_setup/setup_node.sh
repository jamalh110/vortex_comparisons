#!/usr/bin/env bash
# Per-node installer for Ubuntu 22.04 CloudLab d7525 nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"

ACTION="${1:-}"
ROLE="${2:-ray}" # ray | locust

base() {
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-transport-https build-essential ca-certificates curl git git-lfs gnupg \
    libblas-dev liblapack-dev libssl-dev python3-dev python3-pip rsync software-properties-common \
    swig tar unzip wget zip
  sudo -H python3 -m pip install --upgrade pip setuptools wheel packaging
  sudo mkdir -p \
    "${DATA_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
    "${WORK_ROOT}" "${LOG_ROOT}"
  sudo chown -R "${CLOUDLAB_USER}:$(id -gn)" \
    "${DATA_ROOT}" "${WORK_ROOT}" "${LOG_ROOT}"
  chmod -R a+rwX "${LOG_ROOT}"
  echo 'vm.overcommit_memory=1' |
    sudo tee /etc/sysctl.d/99-vortex-ray.conf >/dev/null
  sudo sysctl --system >/dev/null
}

ofed() {
  local archive="MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-x86_64.tgz"
  local directory="${archive%.tgz}"
  cd /tmp
  wget -c "https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/${archive}"
  tar -xzf "${archive}"
  cd "${directory}"
  sudo ./mlnxofedinstall
  echo "OFED installed. Reboot this node, then verify with: ofed_info -s"
}

gpu() {
  cd /tmp
  wget -q -N \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    nvidia-driver-550 cuda-toolkit-12-4
  echo "NVIDIA driver and CUDA toolkit installed. Reboot before continuing."
}

python_stack() {
  export HF_HOME HF_DATASETS_CACHE
  if [[ "${ROLE}" == "ray" ]]; then
    sudo -H python3 -m pip install --no-cache-dir \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu118
    # Install ColBERT dependencies first, then restore the tested pins below.
    sudo -H python3 -m pip install --no-cache-dir colbert-ai==0.2.22
    sudo -H python3 -m pip install --no-cache-dir \
      -r "${SCRIPT_DIR}/requirements-ray.txt"
  elif [[ "${ROLE}" == "locust" ]]; then
    sudo -H python3 -m pip install --no-cache-dir \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cpu
    sudo -H python3 -m pip install --no-cache-dir colbert-ai==0.2.22
    sudo -H python3 -m pip install --no-cache-dir \
      -r "${SCRIPT_DIR}/requirements-locust.txt"
  else
    echo "Unknown role: ${ROLE}" >&2
    exit 2
  fi

  mkdir -p "${WORK_ROOT}"
  if [[ ! -d "${WORK_ROOT}/FLMR/.git" ]]; then
    git clone https://github.com/aliciayuting/FLMR.git "${WORK_ROOT}/FLMR"
  fi
  git -C "${WORK_ROOT}/FLMR" fetch --all
  git -C "${WORK_ROOT}/FLMR" checkout c5db04b5d4e288bd9d3c8594ad285f70c1aa8831
  sudo -H python3 -m pip install --no-deps -e "${WORK_ROOT}/FLMR"
  sudo -H python3 -m pip install --no-deps -e "${WORK_ROOT}/FLMR/third_party/ColBERT"

  # Ray Serve requires protobuf 4.x on this tested stack.
  sudo -H python3 -m pip install --no-cache-dir --force-reinstall protobuf==4.25.9

  if [[ "${ROLE}" == "ray" ]]; then
    sudo WORK_ROOT="${WORK_ROOT}" python3 "${SCRIPT_DIR}/runtime_patch.py"
  else
    WORK_ROOT="${WORK_ROOT}" \
      python3 "${SCRIPT_DIR}/runtime_patch.py" --flmr-only
  fi

  python3 - <<'PY'
import importlib.metadata as metadata
import torch, transformers, tokenizers, datasets, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("datasets", datasets.__version__)
print("numpy", numpy.__version__)
for package in ("ray", "nixl", "nixl-cu12", "protobuf"):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        pass
import flmr
print("flmr", flmr.__file__)
PY
}

cmake_install() {
  local version="3.31.0"
  if command -v cmake >/dev/null; then
    local current
    current="$(cmake --version | awk 'NR==1 {print $3}')"
    if dpkg --compare-versions "${current}" ge 3.24; then
      echo "CMake ${current} is sufficient."
      return
    fi
  fi
  cd "${WORK_ROOT}"
  wget -c "https://github.com/Kitware/CMake/releases/download/v${version}/cmake-${version}.tar.gz"
  tar -xzf "cmake-${version}.tar.gz"
  cd "cmake-${version}"
  ./bootstrap --parallel="$(nproc)"
  make -j"$(nproc)"
  sudo make install
}

faiss_install() {
  [[ "${ROLE}" == "ray" ]] || {
    echo "Faiss GPU is only installed on Ray GPU nodes."
    return
  }
  cmake_install
  sudo -H python3 -m pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true
  cd "${WORK_ROOT}"
  rm -rf faiss
  git clone --branch v1.9.0 --depth 1 https://github.com/facebookresearch/faiss.git
  cd faiss
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_STANDARD=20 \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_RAFT=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 \
    -DCMAKE_CUDA_ARCHITECTURES=80
  cmake --build build --target faiss -j"$(( $(nproc) - 1 ))"
  sudo cmake --install build
  cmake --build build --target swigfaiss -j"$(nproc)"
  cd build/faiss/python
  sudo python3 setup.py install
  python3 - <<'PY'
import faiss
print("faiss", faiss.__version__, "GPUs", faiss.get_num_gpus())
assert faiss.get_num_gpus() >= 1
PY
}

verify() {
  echo "host=$(hostname)"
  python3 --version
  ofed_info -s || true
  rdma link || true
  show_gids || true
  nvidia-smi -L || true
  /usr/local/cuda-12.4/bin/nvcc --version || true
  python3 -c 'import ray, torch, nixl; print("ray", ray.__version__, "torch", torch.__version__, "nixl", nixl.__file__)' \
    || true
}

case "${ACTION}" in
  base) base ;;
  ofed) ofed ;;
  gpu) gpu ;;
  python) python_stack ;;
  cmake) cmake_install ;;
  faiss) faiss_install ;;
  verify) verify ;;
  *)
    echo "Usage: $0 {base|ofed|gpu|python|cmake|faiss|verify} [ray|locust]" >&2
    exit 2
    ;;
esac

