#!/usr/bin/env bash
# Configure the A30 node with four 1g.6gb MIG instances.
set -euo pipefail

count="$(nvidia-smi -L | awk '/MIG 1g.6gb/ {count++} END {print count+0}')"
if [[ "${count}" -eq 4 ]]; then
  echo "Four 1g.6gb MIG devices already exist."
  nvidia-smi -L
  exit 0
fi

mode="$(
  nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader |
    awk 'NR==1 {print $1}'
)"
if [[ "${mode}" != "Enabled" ]]; then
  sudo nvidia-smi -i 0 -mig 1
  echo "MIG mode was enabled. Reboot node4, then rerun this script."
  exit 3
fi

echo "Resetting GPU 0 MIG configuration."
sudo nvidia-smi mig -i 0 -dci || true
sudo nvidia-smi mig -i 0 -dgi || true
sudo nvidia-smi mig -i 0 \
  -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb \
  -C

count="$(nvidia-smi -L | awk '/MIG 1g.6gb/ {count++} END {print count+0}')"
[[ "${count}" -eq 4 ]] || {
  echo "Expected four MIG devices, found ${count}." >&2
  exit 1
}
nvidia-smi -L

