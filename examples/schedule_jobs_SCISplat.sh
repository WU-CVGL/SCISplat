#!/bin/bash

# List of Python commands to run
commands=(
    "cd /run/determined/workdir/home/gsplat/examples"
    "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/NVS/SCISplat/airplants/ \
	--result_dir results/2024_11_28/NVS/vggsfm_airplants \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000 --debug --novel_view"
	"python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/NVS/SCISplat/hotdog/ \
	--result_dir results/2024_11_28/NVS/vggsfm_hotdog \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 3000 --debug --novel_view"
	"python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/NVS/SCISplat/tanabata/ \
	--result_dir results/2024_11_28/NVS/vggsfm_tanabata \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000 --debug --novel_view"
	"python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/NVS/SCISplat/factory/ \
	--result_dir results/2024_11_28/NVS/vggsfm_factory \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000 --debug --novel_view"
	"python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/NVS/SCISplat/vender/ \
	--result_dir results/2024_11_28/NVS/vggsfm_vender \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000 --debug --novel_view"
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
