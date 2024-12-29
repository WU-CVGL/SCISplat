import time
import os
import torch


def get_gpu_id():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_id = torch.cuda.get_device_name(current_device)
        return current_device, gpu_id
    else:
        return None, "No GPU available"


def main():
    gpu_id, gpu_name = get_gpu_id()
    if gpu_id is not None:
        print(f"Running on GPU ID: {gpu_id} ({gpu_name})")
    else:
        print(gpu_name)
    time.sleep(1)


if __name__ == "__main__":
    main()
