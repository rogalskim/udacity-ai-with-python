import torch


def prepare_device(use_gpu: bool) -> torch.device:
    gpu_count = torch.cuda.device_count()
    print(f"{gpu_count} CUDA-capable GPUs found.")

    if not use_gpu or gpu_count < 1:
        print("Local CPU selected for calculations.")
        return torch.device("cpu")

    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    name = torch.cuda.get_device_name(device_id)
    capability = torch.cuda.get_device_capability(device_id)
    print(f"Using {name} GPU with CUDA {capability[0]}.{capability[1]} capability.")
    return device
