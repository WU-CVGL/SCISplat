import subprocess
import threading
import queue
import time


# Function to execute a command on a specific GPU
def run_command_on_gpu(command, gpu_id):
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    process = subprocess.Popen(command, shell=True, env=env)
    process.wait()


# Function to manage GPU execution
def gpu_worker(gpu_id, command_queue):
    while not command_queue.empty():
        command = command_queue.get()
        print(f"Running on GPU {gpu_id}: {command}")
        run_command_on_gpu(command, gpu_id)
        command_queue.task_done()


def main(commands):
    num_gpus = 4
    command_queue = queue.Queue()

    # Enqueue all commands
    for cmd in commands:
        command_queue.put(cmd)

    # Start a thread for each GPU
    threads = []
    for gpu_id in range(num_gpus):
        thread = threading.Thread(target=gpu_worker, args=(gpu_id, command_queue))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("All commands have been executed.")


if __name__ == "__main__":
    # Example list of commands to run
    command_list = [
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/airplants_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_airplants_1000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 1000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/airplants_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_airplants_5000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 5000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/airplants_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_airplants_10000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/hotdog_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_hotdog_1000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 1000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/hotdog_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_hotdog_5000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 5000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/hotdog_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_hotdog_10000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/cozy2room_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_cozy2room_1000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 1000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/cozy2room_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_cozy2room_5000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 5000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/cozy2room_nearest_qf8_shared_pts4096 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_cozy2room_10000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/tanabata_nearest_qf1_pts11000 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_tanabata_1000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 1000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/tanabata_nearest_qf1_pts11000 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_tanabata_5000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 5000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/tanabata_nearest_qf1_pts11000 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_tanabata_10000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/factory_nearest_qf8_shared_pts4096_no-converged \
	--result_dir results/2024_9_20/ablation_points/vggsfm_factory_1000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 1000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/factory_nearest_qf8_shared_pts4096_no-converged \
	--result_dir results/2024_9_20/ablation_points/vggsfm_factory_5000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 5000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/factory_nearest_qf8_shared_pts4096_no-converged \
	--result_dir results/2024_9_20/ablation_points/vggsfm_factory_10000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/vender_nearest_qf1_shared_11000 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_vender_1000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 1000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/vender_nearest_qf1_shared_11000 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_vender_5000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 5000",
        "python simple_trainer_mcmc_sci.py \
	--data_dir data/sci_nerf/ablation_study/initial_points/vender_nearest_qf1_shared_11000 \
	--result_dir results/2024_9_20/ablation_points/vggsfm_vender_10000pts_MCMC \
	--init_scale 1 \
	--disable_viewer \
	--pose_opt \
	--pose_refine \
	--pose_opt_lr 5e-4 \
	--cap_max 100_000 \
	--batch_size 8 \
	--downsample_cap 10000",
        # Add more commands as needed
    ]

    main(command_list)
