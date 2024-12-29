#!/bin/bash

# List of Python commands to run
commands=(
    "cd /run/determined/workdir/home/gsplat/examples"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/DVD/airplants \
	--result_dir results/NVS/naive_baseline/DVD_airplants \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
    "python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/DVD/cozy2room \
	--result_dir results/NVS/naive_baseline/DVD_cozy2room \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/DVD/factory \
	--result_dir results/NVS/naive_baseline/DVD_factory \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/DVD/hotdog \
	--result_dir results/NVS/naive_baseline/DVD_hotdog \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/DVD/tanabata \
	--result_dir results/NVS/naive_baseline/DVD_tanabata \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/DVD/vender \
	--result_dir results/NVS/naive_baseline/DVD_vender \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/FFD/airplants \
	--result_dir results/NVS/naive_baseline/FFD_airplants \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
    "python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/FFD/cozy2room \
	--result_dir results/NVS/naive_baseline/FFD_cozy2room \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/FFD/factory \
	--result_dir results/NVS/naive_baseline/FFD_factory \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/FFD/hotdog \
	--result_dir results/NVS/naive_baseline/FFD_hotdog \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/FFD/tanabata \
	--result_dir results/NVS/naive_baseline/FFD_tanabata \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/FFD/vender \
	--result_dir results/NVS/naive_baseline/FFD_vender \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/TV/airplants \
	--result_dir results/NVS/naive_baseline/TV_airplants \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
    "python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/TV/cozy2room \
	--result_dir results/NVS/naive_baseline/TV_cozy2room \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/TV/factory \
	--result_dir results/NVS/naive_baseline/TV_factory \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/TV/hotdog \
	--result_dir results/NVS/naive_baseline/TV_hotdog \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/TV/tanabata \
	--result_dir results/NVS/naive_baseline/TV_tanabata \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
	"python simple_trainer_mcmc_naive_baseline.py \
	--data_dir data/sci_nerf/NVS/TV/vender \
	--result_dir results/NVS/naive_baseline/TV_vender \
	--init_scale 1 \
	--disable_viewer \
	--cap_max 100_000 --novel_view"
)

# Number of GPUs
num_gpus=4

# Function to run a command on a specific GPU
run_command() {
    local cmd="$1"
    local gpu_id="$2"
    CUDA_VISIBLE_DEVICES=$gpu_id bash -c "$cmd" &
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
