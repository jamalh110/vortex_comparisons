#!/bin/bash

# GPU memory log file
FINAL_GPU_LOG="$HOME/raylogs/gpu_util.dat"
TEMP_GPU_FILE=$(mktemp)

# DCGM log file
DCGM_LOG="$HOME/raylogs/dcgm_log.dat"

# Monitoring intervals
MEMORY_INTERVAL=5  # seconds for nvidia-smi sampling
DCGM_INTERVAL=5    # seconds for DCGM sampling

# Compute DCGM sampling interval in milliseconds
DCGM_SAMPLING_INTERVAL=$((DCGM_INTERVAL * 1000))

# Write header for GPU memory log
echo "Timestamp, GPU_ID, Total_Memory_MB, Used_Memory_MB, Free_Memory_MB" > "$TEMP_GPU_FILE"

# Start DCGM monitoring in the background with unbuffered output
echo "Starting DCGM monitoring (logs to $DCGM_LOG)..."
stdbuf -oL dcgmi dmon -e 203,204,1001,1002,1003,1004,1005,155 -d "$DCGM_SAMPLING_INTERVAL" > "$DCGM_LOG" 2>&1 &

# Capture the DCGM process ID so we can terminate it on exit
DCGM_PID=$!

# Function to handle cleanup when the script is terminated (e.g., via Ctrl+C)
cleanup() {
    echo "Stopping DCGM monitoring..."
    kill -SIGTERM "$DCGM_PID"
    sleep 1  # Give dcgmi time to flush its output
    sync     # Ensure data is written to disk

    echo "Saving collected GPU memory usage to $FINAL_GPU_LOG..."
    mv "$TEMP_GPU_FILE" "$FINAL_GPU_LOG"

    echo "Logs saved to $FINAL_GPU_LOG and $DCGM_LOG."
    exit 0
}

# Trap Ctrl+C (SIGINT) to call cleanup function
trap cleanup SIGINT

echo "Logging GPU memory usage every $MEMORY_INTERVAL seconds. DCGM monitoring every $DCGM_INTERVAL second."
echo "Press Ctrl+C to stop and save logs."

# Main loop: log GPU memory usage using nvidia-smi
while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits | \
    while IFS=',' read -r GPU_ID TOTAL USED FREE; do
        echo "$TIMESTAMP, $GPU_ID, $TOTAL, $USED, $FREE" >> "$TEMP_GPU_FILE"
    done
    sleep $MEMORY_INTERVAL
done