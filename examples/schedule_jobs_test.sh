#!/bin/bash

# List of Python commands to run
commands=(
    "python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
	"python test_gpu_util.py"
    # Add more commands as needed
)

# Number of GPUs
num_gpus=4

# Function to run a command on a specific GPU
run_command() {
    local cmd=$1
    local gpu_id=$2
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
}

# Main loop to schedule jobs
i=0
for cmd in "${commands[@]}"; do
    gpu_id=$((i % num_gpus))
    run_command "$cmd" $gpu_id
    ((i++))

    # If we've scheduled as many jobs as GPUs, wait for one to finish
    if (( i % num_gpus == 0 )); then
        wait -n
    fi
done

# Wait for all remaining jobs to finish
wait

echo "All commands have been executed."