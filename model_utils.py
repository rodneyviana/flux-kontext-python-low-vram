import PIL
import PIL.Image
import numpy as np
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from diffusers.hooks import apply_group_offloading
from cuda_utils import get_num_cuda_devices, get_gpu_total_memory, get_gpu_available_memory, get_gpu_temperature, garbage_collect

def get_cuda_index(device: str | torch.device = "cuda:0") -> int:
    if device == None:
        return 0
    # if device is a string and starts with "cuda:"
    if isinstance(device, str) and device.find("cuda:") > -1:
        num_devices = get_num_cuda_devices()
        index = int(device.split(":")[1])
        if index >= num_devices:
            print(f"Invalid CUDA index {index}, using 0")
            index = 0
        return index
    else:
        if isinstance(device, torch.device) and device.index is not None:
            return device.index
        return 0

def is_low_memory(device: str = "cuda:0") -> bool:
    cuda_index = get_cuda_index(device)
    gpu_total_memory = get_gpu_total_memory(cuda_index) / 1024 / 1024 / 1024
    if gpu_total_memory < 40:
        print(f"Low memory detected on device {device} with {gpu_total_memory}GB of memory")
        return True
    return False

def apply_low_memory(pipe: FluxKontextPipeline, device: str = "cuda:0"):
    device_id = get_cuda_index(device)
    print(f"Using CUDA device index: {device_id}")
    is_low_memory_device = is_low_memory(device)

    if is_low_memory_device:
        print("Applying low memory settings for FLUX")
        apply_group_offloading(
            pipe.text_encoder, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
        print(f"Applied group offloading to text_encoder on device {device}")
        apply_group_offloading(
            pipe.text_encoder_2, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
        print(f"Applied group offloading to text_encoder_2 on device {device}")
        apply_group_offloading(
            pipe.transformer,
            offload_type="leaf_level",
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            use_stream=True,
        )
        print(f"Applied group offloading to transformer on device {device}")
        apply_group_offloading(
            pipe.vae, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
        print(f"Applied group offloading to vae on device {device}")
    else:
        print("Low memory settings not required for this device")
        pipe.to(device)

_pipe = None
def load_model(device: str = "cuda:0") -> FluxKontextPipeline:
    global _pipe
    if _pipe is not None:
        print("Model already loaded, returning existing pipeline")
        return _pipe
    cuda_index = get_cuda_index(device)
    print(f"Loading model on CUDA device index: {cuda_index}")
    _pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    apply_low_memory(_pipe, device=device)
    return _pipe

def run_prompt(pipe: FluxKontextPipeline, prompt: str, input_image: PIL.Image.Image | np.ndarray, guidance_scale: float = 2.5, num_inference_steps: int = 28):
    if pipe is None:
        raise ValueError("Pipeline is not loaded. Please load the model first.")
    print(f"Running prompt: {prompt} with guidance scale: {guidance_scale} and {num_inference_steps} steps")
    image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    return image
