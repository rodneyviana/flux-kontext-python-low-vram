import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature

def get_num_cuda_devices():
    """Get the number of CUDA devices available."""
    return torch.cuda.device_count()

def get_gpu_total_memory(device_index):
    """Get the total memory (in bytes) of a specific CUDA device."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    return memory_info.total

def get_gpu_available_memory(device_index):
    """Get the available memory (in bytes) of a specific CUDA device."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    return memory_info.free

def get_gpu_temperature(device_index):
    """Get the temperature (in degrees Celsius) of a specific CUDA device."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    temperature = nvmlDeviceGetTemperature(handle, 0)
    return temperature

def garbage_collect(device_index=None):
    """Perform garbage collection on a specific CUDA device or all devices."""
    gc.collect()
    if device_index is not None:
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
    else:
        # Clear cache for all available devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()