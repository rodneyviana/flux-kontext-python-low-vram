import datetime
import fnmatch
import glob
from uuid import uuid4
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import gc
import psutil
import os
import subprocess
from cuda_utils import garbage_collect
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, AutoencoderKL, FluxPipeline, FluxInpaintPipeline
from compel import Compel, ReturnedEmbeddingsType
import torch
import yaml
import time
from diffusers.hooks import apply_group_offloading
from schedulers import PipeScheduler, SchedulersByModel
import json
from dotenv import load_dotenv
from metadata_helper import get_model_by_name, get_lora_by_name
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1, get_weighted_text_embeddings_sdxl
from cuda_util import get_num_cuda_devices, get_gpu_total_memory, get_gpu_temperature, get_gpu_available_memory

load_dotenv()

def force_free_system_memory():
    """Force the system to free cached memory and buffers"""
    try:
        print("  🔄 Forcing system memory cleanup...")
        
        # Method 1: Sync and drop caches (requires sudo, may not work)
        try:
            # Drop page cache, dentries and inodes
            subprocess.run(['sudo', 'sync'], check=False, capture_output=True)
            subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=False, capture_output=True)
            print("    ✓ System caches dropped")
        except:
            print("    ⚠️  Could not drop system caches (need sudo)")
        
        # Method 2: Force Python to release memory back to OS
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            print("    ✓ malloc_trim executed")
        except:
            print("    ⚠️  Could not execute malloc_trim")
        
        # Method 3: Set memory management environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        
    except Exception as e:
        print(f"    ⚠️  System memory cleanup error: {e}")

def aggressive_memory_cleanup():
    """More aggressive memory cleanup including system-level operations"""
    print("\n🔥 Performing AGGRESSIVE memory cleanup...")
    
    # Show memory before cleanup
    print_detailed_memory_status("BEFORE aggressive cleanup")
    
    # Step 1: Clear all Python objects
    collected = gc.collect()
    print(f"  ✓ Python GC collected {collected} objects")
    
    # Step 2: Clear CUDA memory for all devices
    if torch.cuda.is_available():
        print(f"  🔧 Aggressively clearing CUDA cache for {torch.cuda.device_count()} devices...")
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                before_cleanup = get_gpu_available_memory(i) / 1024 / 1024 / 1024
                
                # Clear cached memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Try to reset memory stats
                try:
                    torch.cuda.reset_peak_memory_stats(i)
                    torch.cuda.reset_accumulated_memory_stats(i)
                except:
                    pass
                
                after_cleanup = get_gpu_available_memory(i) / 1024 / 1024 / 1024
                freed = after_cleanup - before_cleanup
                print(f"    GPU {i}: Freed {freed:.2f}GB ({before_cleanup:.2f}GB → {after_cleanup:.2f}GB available)")
    
    # Step 3: Force multiple garbage collection passes
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            print(f"  ✓ GC pass {i+1}: collected {collected} objects")
    
    # Step 4: Force system memory cleanup
    force_free_system_memory()
    
    # Step 5: Wait a moment for system to process
    import time
    time.sleep(1)
    
    # Step 6: Final garbage collection
    collected = gc.collect()
    if collected > 0:
        print(f"  ✓ Final GC: collected {collected} objects")
    
    # Show final memory status
    print_detailed_memory_status("AFTER aggressive cleanup")
    print("🔥 Aggressive memory cleanup completed!\n")

def get_usable_ram_gb():
    """Calculate usable RAM in GB, accounting for Linux memory management"""
    ram = psutil.virtual_memory()
    
    # Method 1: Use available directly (conservative)
    available_direct = ram.available / 1024 / 1024 / 1024
    
    # Method 2: Calculate based on total - used (more optimistic)
    available_calculated = (ram.total - ram.used) / 1024 / 1024 / 1024
    
    # Method 3: Use total - used but leave some safety margin (30% of total for safety)
    safety_margin = ram.total * 0.3  # 30% safety margin
    available_with_margin = (ram.total - ram.used - safety_margin) / 1024 / 1024 / 1024
    
    # Method 4: Check if we have enough for basic operation (be very conservative)
    min_required = 8.0  # Minimum 8GB required for FLUX offloading
    
    print(f"  🔍 RAM analysis:")
    print(f"    Available (direct): {available_direct:.1f}GB")
    print(f"    Available (calculated): {available_calculated:.1f}GB") 
    print(f"    Available (with margin): {available_with_margin:.1f}GB")
    
    # Use the most conservative approach when available is very low
    if available_direct < min_required:
        print(f"  ⚠️  WARNING: Only {available_direct:.1f}GB RAM available, need at least {min_required}GB")
        
        # Try aggressive cleanup first
        print("  🔥 Attempting aggressive memory cleanup...")
        aggressive_memory_cleanup()
        
        # Re-check after cleanup
        ram_after = psutil.virtual_memory()
        available_after = ram_after.available / 1024 / 1024 / 1024
        print(f"  📊 RAM after cleanup: {available_after:.1f}GB available")
        
        if available_after < min_required:
            return available_after  # Return what we have, let caller decide
        else:
            return available_after
    
    # If calculated is much higher than available, use calculated with margin
    if available_calculated > available_direct * 1.5 and available_with_margin > 0:
        usable = max(available_with_margin, available_direct)
        print(f"  🔍 Using calculated with margin ({usable:.1f}GB)")
    else:
        usable = available_direct
        print(f"  🔍 Using direct available ({usable:.1f}GB)")
    
    return usable

def print_detailed_memory_status(label: str = ""):
    """Print detailed memory status for all devices and system RAM"""
    print(f"\n=== Memory Status {label} ===")
    
    # GPU memory for all devices
    if torch.cuda.is_available():
        print("GPU Memory Status:")
        for i in range(get_num_cuda_devices()):
            total = get_gpu_total_memory(i) / 1024 / 1024 / 1024
            available = get_gpu_available_memory(i) / 1024 / 1024 / 1024
            used = total - available
            usage_percent = (used / total) * 100
            temp = get_gpu_temperature(i)
            print(f"  GPU {i}: {used:.2f}GB used / {total:.2f}GB total ({usage_percent:.1f}%), {available:.2f}GB available, {temp}°C")
    else:
        print("GPU: Not available")
    
    # System RAM - Enhanced calculation
    ram = psutil.virtual_memory()
    ram_total = ram.total / 1024 / 1024 / 1024
    ram_used = ram.used / 1024 / 1024 / 1024
    ram_available = ram.available / 1024 / 1024 / 1024
    
    # Calculate percentage correctly
    ram_usage_percent = (ram_used / ram_total) * 100
    print(f"RAM: {ram_used:.2f}GB used / {ram_total:.2f}GB total ({ram_usage_percent:.1f}%), {ram_available:.2f}GB available")
    
    # Show memory breakdown
    print(f"RAM Details: cached={ram.cached/1024/1024/1024:.2f}GB, buffers={ram.buffers/1024/1024/1024:.2f}GB")
    
    # Debug RAM values
    print(f"DEBUG RAM: total={ram.total}, used={ram.used}, available={ram.available}, percent={ram.percent}")
    
    # Swap memory
    swap = psutil.swap_memory()
    if swap.total > 0:
        swap_total = swap.total / 1024 / 1024 / 1024
        swap_used = swap.used / 1024 / 1024 / 1024
        swap_free = swap.free / 1024 / 1024 / 1024
        swap_usage_percent = (swap_used / swap_total) * 100 if swap_total > 0 else 0
        print(f"Swap: {swap_used:.2f}GB used / {swap_total:.2f}GB total ({swap_usage_percent:.1f}%), {swap_free:.2f}GB free")
    else:
        print("Swap: Not configured")
    
    # Show top memory consuming processes
    try:
        print("Top memory processes:")
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                processes.append((proc.info['pid'], proc.info['name'], proc.info['memory_info'].rss))
            except:
                pass
        processes.sort(key=lambda x: x[2], reverse=True)
        for i, (pid, name, memory) in enumerate(processes[:5]):
            memory_gb = memory / 1024 / 1024 / 1024
            print(f"  {i+1}. {name} (PID {pid}): {memory_gb:.2f}GB")
    except:
        pass
    
    print("=" * 50)

def deep_memory_cleanup():
    """Comprehensive memory cleanup for both GPU and CPU"""
    print("\n🧹 Performing deep memory cleanup...")
    
    # Show memory before cleanup
    print_detailed_memory_status("BEFORE cleanup")
    
    # Clear Python garbage collector
    collected = gc.collect()
    print(f"  ✓ Python GC collected {collected} objects")
    
    # Clear CUDA cache for all devices
    if torch.cuda.is_available():
        print(f"  🔧 Clearing CUDA cache for {torch.cuda.device_count()} devices...")
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                before_cleanup = get_gpu_available_memory(i) / 1024 / 1024 / 1024
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after_cleanup = get_gpu_available_memory(i) / 1024 / 1024 / 1024
                freed = after_cleanup - before_cleanup
                print(f"    GPU {i}: Freed {freed:.2f}GB ({before_cleanup:.2f}GB → {after_cleanup:.2f}GB available)")
    
    # Force garbage collection again
    collected2 = gc.collect()
    if collected2 > 0:
        print(f"  ✓ Second GC pass collected {collected2} additional objects")
    
    # Show final memory status
    print_detailed_memory_status("AFTER cleanup")
    print("🧹 Memory cleanup completed!\n")

def check_inference_memory(device: str = "cuda:0"):
    """Check if there's enough memory for inference"""
    def get_cuda_index(device: str | torch.device = "cuda:0") -> int:
        if device == None:
            return 0
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
    
    device_id = get_cuda_index(device)
    gpu_available = get_gpu_available_memory(device_id) / 1024 / 1024 / 1024
    
    print(f"\n🔍 Pre-inference memory check:")
    print(f"  GPU available: {gpu_available:.2f}GB")
    
    # Need at least 6GB free for inference (more conservative for RTX 5090)
    if gpu_available < 6:
        print(f"⚠️  WARNING: Low GPU memory for inference: {gpu_available:.2f}GB available")
        print("🧹 Performing emergency cleanup...")
        aggressive_memory_cleanup()
        
        # Check again after cleanup
        gpu_available_after = get_gpu_available_memory(device_id) / 1024 / 1024 / 1024
        if gpu_available_after < 4:
            raise RuntimeError(f"Insufficient GPU memory for inference: {gpu_available_after:.2f}GB available. Need at least 4GB.")
        else:
            print(f"✓ Memory freed: {gpu_available_after:.2f}GB now available")
    else:
        print(f"✅ Sufficient GPU memory for inference: {gpu_available:.2f}GB available")

def apply_flux_optimized_offloading(pipe: DiffusionPipeline, device: str = "cuda:0"):
    """Apply FLUX-optimized offloading - designed specifically for 32GB RTX 5090"""
    print("🔄 Applying FLUX-optimized offloading for RTX 5090...")
    
    # Strategy: Keep only the transformer on GPU, offload everything else
    # This leaves maximum memory for the transformer which is the memory-hungry part
    
    print("  📦 Moving core components strategically...")
    
    # Move transformer to GPU first (most important for performance)
    pipe.transformer = pipe.transformer.to(device)
    print("    ✓ transformer loaded to GPU")
    
    # Offload text encoders (used only at beginning)
    apply_group_offloading(
        pipe.text_encoder, 
        offload_device=torch.device("cpu"),
        onload_device=torch.device(device),
        offload_type="leaf_level",
        use_stream=True,
    )
    print("    ✓ text_encoder offloaded to CPU")
    
    apply_group_offloading(
        pipe.text_encoder_2, 
        offload_device=torch.device("cpu"),
        onload_device=torch.device(device),
        offload_type="leaf_level",
        use_stream=True,
    )
    print("    ✓ text_encoder_2 offloaded to CPU")
    
    # Offload VAE (used only at end for decoding)
    apply_group_offloading(
        pipe.vae, 
        offload_device=torch.device("cpu"),
        onload_device=torch.device(device),
        offload_type="leaf_level",
        use_stream=True,
    )
    print("    ✓ vae offloaded to CPU")
    
    print("  ✅ FLUX-optimized offloading completed")
    print("     → Transformer on GPU (main compute)")
    print("     → Text encoders on CPU (encoding phase)")
    print("     → VAE on CPU (decoding phase)")

"""Adjust width and height to be a multiple of 8 rounding up
:param width:
:param height:
:return width, height:
"""
def adjust_width_height(width, height):
    if width % 8 != 0:
        width = width + 8 - (width % 8)
    if height % 8 != 0:
        height = height + 8 - (height % 8)
    return width, height

"""enlarge image to center and fill with a provideed color
:param pil_img:
:param adjusted_width:
:param adjusted_height:
:param color:
"""
def enlarge_to_center(pil_img, adjusted_width, adjusted_height, color=(0, 0, 0)):
    img_width, img_height = pil_img.size
    if img_width == adjusted_width and img_height == adjusted_height:
        return pil_img
    if img_width > adjusted_width or img_height > adjusted_height:
        return pil_img.resize((adjusted_width, adjusted_height), Image.Resampling.LANCZOS)
    new_image = Image.new(pil_img.mode, (adjusted_width, adjusted_height), color=color)
    new_image.paste(pil_img, ((adjusted_width - img_width) // 2,
                               (adjusted_height - img_height) // 2))
    return new_image

"""crop image to center
:param pil_img:
:param crop_width:
:param crop_height:
:return pil_img:
"""
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    if img_width == crop_width and img_height == crop_height:
        return pil_img
    if img_width < crop_width or img_height < crop_height:
        return pil_img.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

"""Fix lora name
:param module_name:
:return:
"""
def fix_lora_name(module_name: str):
    return module_name.replace(".", "_")

class ModelType:
    SD1 = "SD1"
    SDXL = "SDXL"
    FLUX = "FLUX"
    QWEN = "QWEN"

class PromptInfo:
    prompt: str
    negative_prompt: str
    scheduler: str
    steps: int
    cfg: float
    width: int
    height: int
    loras: list = []
    model: str
    model_type: ModelType
    seed: int
    client_id: str
    prompt_id: str
    obj = None
    def __init__(self, prompt: str, negative_prompt: str, scheduler: str, steps: int, cfg: float, width: int, height: int, loras: list, client_id: str, model: str, seed: int, model_type: ModelType, obj = None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.scheduler = scheduler
        self.steps = steps
        self.cfg = cfg
        self.width = width
        self.height = height
        self.loras = loras
        self.model = model
        self.client_id = client_id
        self.prompt_id = str(uuid4())
        self.seed = seed
        self.model_type = model_type
        self.obj = obj
        
    def to_simple_json_string(self):
        prompt_dict =  {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "scheduler": self.scheduler,
            "steps": self.steps,
            "cfg": self.cfg,
            "width": self.width,
            "height": self.height,
            "loras": self.loras,
            "model_type": self.model_type,
            "model": self.model,
            "client_id": self.client_id,
            "prompt_id": self.prompt_id,
            "seed": self.seed
        }
        return json.dumps(prompt_dict, indent=2)

def insensitive_glob(pattern):
    return list(glob.iglob(pattern))

def get_models_from_metadata():
    with open("allMetadata.json", 'r', encoding="utf-8") as f:
        metadata = json.load(f)
        # get only SDXL models
        models = [x for x in metadata if "SDXL" in x["baseModel"]]
        return models

# loras: https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference

def get_long_prompt_embeds(pipeline: DiffusionPipeline, prompt: str, device: str):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length
    compel = Compel([pipeline.tokenizer, pipeline.tokenizer_2],
                    [pipeline.text_encoder, pipeline.text_encoder_2],
                    device=device,
                    truncate_long_prompts=False,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True])
    prompt_embeds, pool = compel(prompt)
    return prompt_embeds, pool

def pad_conditioning_tensors_to_same_length(positive_embeds, negative_embeds, pipeline: DiffusionPipeline, device: str):
    """ Pad conditioning tensors to same length
    :param prostive_embeds:
    :param negative_embeds:
    :return:
    """
    max_length = max(positive_embeds.shape[-2], negative_embeds.shape[-2])
    diff = max(abs(max_length - positive_embeds.shape[-2]), abs(max_length - negative_embeds.shape[-2]))
    if max_length - positive_embeds.shape[-2] > 0:
        positive_embeds = torch.nn.functional.pad(positive_embeds,
                                                  pad=(0, 0, 0, diff),
                                                  mode='constant',
                                                   value=0 ) # add diff to the end of 2nd dimension
    if max_length - negative_embeds.shape[-2] > 0:
        negative_embeds = torch.nn.functional.pad(negative_embeds, 
                                                    pad=(0, 0, 0, diff), # add diff to the end of 2nd dimension
                                                    mode='constant',
                                                    value=0)
    
    return positive_embeds, negative_embeds

class PipeItem:
    def __init__(self, model_name: str, model_type = ModelType.SDXL, device: str = "cuda:0"):
        self.loras = []
        self.model_name = model_name
        self.device = device
        self.model_type = model_type
        print(f"Model Type: {self.model_type}")
        self.last_used = time.time()
        self.is_turbo = model_name.lower().find("turbo") > -1
        self.save_path = os.getenv('OUTPUT_PATH')
        if self.save_path is None or self.save_path == "":
            self.save_path = "./"
        print(f"Save path: {self.save_path}")
        if device.find("cuda:") > -1:
            self.cuda_index = int(device.split(":")[1])
        else:
            self.cuda_index = 0

    vae_path: str = ""
    vae: AutoencoderKL = None
    save_path: str
    pipe: StableDiffusionXLPipeline = None
    loras: list = []
    model_name: str = ""
    device: str = ""
    is_XL: bool = False
    last_used: float = 0
    images: list = []
    self_last_used: float = time.time()
    applied_group_offloading: bool = False

    def pipe(self):
        return self.pipe
    
    def add_lora(self, lora: str, weight: float = 1):
        if lora == "None":
            return False
        generator = Generator()
        full_name_lora = get_lora_by_name(lora)
        if(full_name_lora == None):
            print(f"Lora {lora} not found")
            return False
        
        adapter_name = lora.removesuffix(".safetensors")
        # verify if adapter is already added
        for adapter in self.loras:
            if(adapter["name"] == adapter_name):
                adapter["weight"] = weight
                print(f"Updated lora {adapter_name} with weight {weight}")
                return True
        self.loras.append({
            "name": adapter_name,
            "weight": weight,
            "lora": full_name_lora
        })
        print(f"Added lora {adapter_name} with weight {weight}")
        return True

    def remove_lora(self, lora: str):
        adapter_name = lora.removesuffix(".safetensors")
        for adapter in self.loras:
            if(adapter["name"] == adapter_name):
                self.loras.remove(adapter)
                print(f"Removed lora {adapter_name}")
                return True
        print(f"Lora {adapter_name} not found")
        return False

    def apply_loras(self):
        loaded_loras_in_pipe = self.pipe.get_active_adapters()
        print(f"Already loaded loras: {loaded_loras_in_pipe}")
        weights = []
        adapters = []
        
        for lora in self.loras:
            if(fix_lora_name(lora["name"]) not in loaded_loras_in_pipe):
                lora_path = lora["lora"]
                self.pipe.load_lora_weights(lora_path, weight_name=f"{os.path.basename(lora_path)}", adapter_name=fix_lora_name(lora["name"]))
            weights.append(lora["weight"])
            adapters.append(fix_lora_name(lora["name"]))
            loaded_loras_in_pipe = [x for x in loaded_loras_in_pipe if x != fix_lora_name(lora["name"])]
        print(f"Unmatched loras: {loaded_loras_in_pipe}")
        if len(loaded_loras_in_pipe) > 0:
            self.pipe.delete_adapters(loaded_loras_in_pipe)
            print(f"removed loras: {loaded_loras_in_pipe}")
        if(len(adapters) > 0):
            self.pipe.set_adapters(adapters, weights)
            self.pipe.set_lora_device(adapters, torch.device(self.device))
        print(f"Active loras after: {self.pipe.get_active_adapters()}")

        if len(adapters) == 0:
            self.pipe.disable_lora()
            self.loras = []
            print("No loras active")
            
    def run_prompt(self, prompt_info: PromptInfo):
        self.last_used = time.time()
        prompt = prompt_info.prompt
        negative_prompt = prompt_info.negative_prompt
        scheduler = prompt_info.scheduler
        steps = prompt_info.steps
        cfg = prompt_info.cfg
        width = prompt_info.width
        height = prompt_info.height
        loras = prompt_info.loras
        seed = prompt_info.seed
        self.loras = []
        for lora in loras:
            self.add_lora(lora["name"], lora["weight"])
        self.apply_loras()
        images = []
        
        if self.vae_path != "":
            if self.vae == None:
                print(f"Loading VAE {self.vae_path}")
                self.vae = AutoencoderKL.from_single_file(self.vae_path)

        generator = torch.Generator(self.device).manual_seed(seed)
        scheds_by_model = SchedulersByModel(self.pipe)
        scheduler_obj = scheds_by_model.get_scheduler(prompt_info.scheduler, is_turbo=self.is_turbo)
        print(f"Scheduler: {scheduler_obj.__class__.__name__}")
        new_width, new_height = adjust_width_height(width, height)
        if new_height != height or new_width != width:
            print(f"Adjusted width and height to {new_width}x{new_height}")
        
        # Pre-inference memory check and cleanup
        print_detailed_memory_status("BEFORE inference")
        check_inference_memory(self.device)
        
        print(f"Embeddings - Model Type: {self.model_type}")
        
        try:
            if(self.model_type == ModelType.FLUX):
                prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(self.pipe, prompt, device=self.device)
                print(f"GPU temperature before: {get_gpu_temperature(self.cuda_index)}")

                piperun = self.pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    height=new_height,
                    width=new_width,
                    guidance_scale=cfg,
                    num_inference_steps=steps,
                    max_sequence_length=512,
                    generator=generator,
                )
                print(f"GPU temperature after: {get_gpu_temperature(self.cuda_index)}")
                
            else:
                positive_embeds, negative_embeds, positive_pooled, negative_pooled = get_weighted_text_embeddings_sdxl(self.pipe, prompt, negative_prompt)          
                piperun = self.pipe(prompt_embeds=positive_embeds,
                                        negative_embeds=negative_embeds,
                                        pooled_prompt_embeds=positive_pooled,
                                        negative_pooled_embeds=negative_pooled,
                                        scheduler=scheduler_obj,
                                        num_inference_steps=steps,
                                        generator=generator,
                                        guidance_scale=cfg,
                                        width=new_width,
                                        height=new_height,
                                        )
            images = piperun.images
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM Error during inference: {e}")
            print("🚨 Attempting emergency recovery...")
            
            # Emergency cleanup
            aggressive_memory_cleanup()
            
            # Try to apply more aggressive offloading for FLUX
            if self.model_type == ModelType.FLUX:
                print("🔄 Applying emergency FLUX offloading...")
                try:
                    apply_group_offloading(
                        self.pipe.transformer,
                        offload_type="leaf_level", 
                        offload_device=torch.device("cpu"),
                        onload_device=torch.device(self.device),
                        use_stream=True,
                    )
                    print("    ✓ Emergency transformer offload applied")
                except Exception as offload_error:
                    print(f"    ⚠️  Could not offload transformer: {offload_error}")
            
            raise RuntimeError("CUDA out of memory during inference. Try closing other applications or reducing image size.")
        
        print(f"Generated {len(images)} images")
        try:
            if(prompt_info.obj != None):
                prompt_info.obj.show_image(images, width, height)    
        except:
            print("Error showing image")
        
        metadata = PngInfo()
        metadata.add_text("prompt", prompt_info.to_simple_json_string())

        print(f"Memory used: {torch.cuda.memory_allocated(self.device)}")
        print(f"🧹 Comprehensive cleanup after inference...")
        
        # Use aggressive cleanup instead of basic garbage collection
        aggressive_memory_cleanup()
        
        print(f"Memory used after cleanup: {torch.cuda.memory_allocated(self.device)}")
        for image in images:
            image = crop_center(image, width, height)
            file_name = f"{prompt_info.client_id}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.png"
            full_path = os.path.join(self.save_path, file_name)
            image.save(full_path, pnginfo=metadata)
            print(f"Saved image {full_path}")
            return full_path

    def delete(self):
        del self.pipe

models_root_path = os.getenv('MODELS_ROOT_PATH') or '.'
models_root_models = os.path.join(models_root_path, "models")
models_root_extra_models_yaml = os.path.join(models_root_path, "extra_model_paths.yaml")

class Generator(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Generator, cls).__new__(cls)

        return cls._instance
    running_queue: list = []
    waiting_queue: list = []
    pipe_items: dict = {}
    low_memory_dict: dict = {}

    model_folders: list = None
    lora_folders: list = None
    embed_folders: list = None
    vae_folders: list = None
    upscaler_folders: list = None
    controlnet_folders: list = None
    flux_inpainting_pipe: FluxInpaintPipeline = None
    
    def get_cuda_index(self, device: str | torch.device = "cuda:0") -> int:
        if device == None:
            return self.cuda_index if self.get_cuda_index is not None and hasattr(self, 'cuda_index') else 0
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
        return self.cuda_index
    
    def get_max_pipe_items(self, device = "cuda:0") -> int:
        # max_pipe_items is the base on total memory of the GPU. 32GB per pipe
        cuda_index = self.get_cuda_index(device)
        gpu_total_memory = get_gpu_total_memory(cuda_index) / 1024 / 1024 / 1024
        if gpu_total_memory > 32:
            self.max_pipe_items = int(gpu_total_memory / 32)
        else:
            self.max_pipe_items = 1
        print(f"Max pipe items set to {self.max_pipe_items} for device {device} with {gpu_total_memory}GB of memory")
        return self.max_pipe_items

    def apply_low_memory(self, pipe: DiffusionPipeline, is_flux: bool = False, device: str = "cuda:0", full_model_name: str = None):
        cuda_index = self.get_cuda_index(device)
        gpu_total_memory = get_gpu_total_memory(cuda_index) / 1024 / 1024 / 1024
        gpu_available = get_gpu_available_memory(cuda_index) / 1024 / 1024 / 1024
        usable_ram = get_usable_ram_gb()
        
        applied_group_offloading = False
        print(f"GPU total memory: {gpu_total_memory:.2f}GB, available: {gpu_available:.2f}GB")
        print(f"RAM usable: {usable_ram:.2f}GB")
        
        if full_model_name is not None:
            applied_group_offloading = self.low_memory_dict.get(full_model_name, False)
        print(f"Applying low memory settings for model {full_model_name} is {applied_group_offloading}")

        if is_flux and not applied_group_offloading:
            print("🔧 Applying enhanced FLUX memory optimization...")
            self.low_memory_dict[full_model_name] = True
            
            # Use the same logic as model_utils.py
            if gpu_total_memory >= 30 and usable_ram > 12:
                # RTX 5090 optimized strategy
                print("🚀 Strategy: RTX 5090 OPTIMIZED for FLUX")
                apply_flux_optimized_offloading(pipe, device)
            elif gpu_available > 15 and usable_ram > 8:
                # Standard offloading
                print("⚖️  Strategy: STANDARD FLUX offloading")
                pipe.transformer = pipe.transformer.to(device)
                apply_group_offloading(
                    pipe.text_encoder, 
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                apply_group_offloading(
                    pipe.text_encoder_2, 
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                apply_group_offloading(
                    pipe.vae, 
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    offload_type="leaf_level",
                    use_stream=True,
                )
            elif usable_ram > 6:
                # Full offloading
                print("💾 Strategy: FULL FLUX offloading")
                apply_group_offloading(
                    pipe.text_encoder, 
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                apply_group_offloading(
                    pipe.text_encoder_2, 
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                apply_group_offloading(
                    pipe.transformer,
                    offload_type="leaf_level",
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    use_stream=True,
                )
                apply_group_offloading(
                    pipe.vae, 
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device(device),
                    offload_type="leaf_level",
                    use_stream=True,
                )
            else:
                print("❌ Insufficient memory for FLUX model")
                raise RuntimeError(f"Insufficient memory: GPU={gpu_available:.1f}GB, RAM={usable_ram:.1f}GB usable.")

    def dispose_pipe(self, pipe: DiffusionPipeline, model_name: str = None):
        self.low_memory_dict[model_name] = False
        if(pipe is not None):
            print(f"🗑️  Disposing pipe on device {pipe.device}")
            
            # Enhanced disposal with aggressive cleanup
            pipe.maybe_free_model_hooks()
            pipe.remove_all_hooks()
            pipe.to("cpu")
            
            # Aggressive cleanup after disposal
            aggressive_memory_cleanup()

            index = self.get_cuda_index(pipe.device)
            del pipe
            print(f"✓ Disposed pipe on cuda index {index}")
            if model_name is not None:
                print(f"✓ Removing model {model_name} from pipe items")
                self.low_memory_dict.pop(model_name, None)
                self.pipe_items.pop(model_name, None)
        else:
            print("Pipe is None, nothing to dispose")

    def dispose_all_pipes(self, dispose_inpainting: bool = False):
        print("🗑️  Disposing all pipes...")
        keys = list(self.pipe_items.keys())
        for key in keys:
            pipe_item = self.pipe_items[key]
            if(pipe_item.pipe is not None):
                self.dispose_pipe(pipe_item.pipe, key)
        self.pipe_items = {}
        self.low_memory_dict = {}
        print("✓ Disposed all regular pipes")
        if dispose_inpainting and self.flux_inpainting_pipe is not None:
            print("🗑️  Disposing flux inpainting pipe")
            self.dispose_pipe(self.flux_inpainting_pipe, "flux_inpainting")
            self.flux_inpainting_pipe = None

    def is_low_memory(self, device: str = "cuda:0") -> bool:
        cuda_index = self.get_cuda_index(device)
        gpu_total_memory = get_gpu_total_memory(cuda_index) / 1024 / 1024 / 1024
        gpu_available_memory = get_gpu_available_memory(cuda_index) / 1024 / 1024 / 1024
        usable_ram = get_usable_ram_gb()
        
        if gpu_total_memory < 40 or gpu_available_memory < 8 or usable_ram < 8:
            print(f"Low memory detected on device {device}: GPU={gpu_total_memory:.1f}GB total, {gpu_available_memory:.1f}GB available, RAM={usable_ram:.1f}GB usable")
            return True
        return False

    def load_flux_properly(self, full_model_name: str, device: str = "cuda:0") -> DiffusionPipeline:
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
        from optimum.quanto import freeze, quantize, qfloat8
        
        print("📥 Loading FLUX model components...")
        print_detailed_memory_status("BEFORE FLUX loading")
        
        bl_repo = "black-forest-labs/FLUX.1-dev"
        txt_encoder_and_tokenizer_repo = "openai/clip-vit-large-patch14"
        dtype = torch.bfloat16
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bl_repo, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(txt_encoder_and_tokenizer_repo, torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(bl_repo, subfolder="vae", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(txt_encoder_and_tokenizer_repo, torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(bl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
        tokenizer_2 = T5Tokenizer.from_pretrained(bl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
        transformer = FluxTransformer2DModel.from_pretrained(bl_repo, subfolder="transformer", torch_dtype=dtype)
        
        cuda_index = self.get_cuda_index(device)
        gpu_total_memory = get_gpu_total_memory(cuda_index) / 1024 / 1024 / 1024
        if gpu_total_memory < 25:
            print("🔧 Applying quantization for low memory GPU")
            transformer = quantize(transformer, weights=qfloat8)
            transformer = freeze(transformer)
            quantize(text_encoder_2, weights=qfloat8)

        pipe = FluxPipeline.from_single_file(full_model_name,
                                                text_encoder=text_encoder,
                                                tokenizer=tokenizer,
                                                text_encoder_2=text_encoder_2,
                                                tokenizer_2=tokenizer_2,
                                                transformer=transformer,
                                                vae=vae,
                                                scheduler=scheduler,
                                                torch_dtype=dtype,
                                                from_single_file=True,
                                                use_safetensors=True,
                                                safety_checks=None
                                                )
        
        print_detailed_memory_status("AFTER FLUX loading, BEFORE optimization")
        
        self.low_memory_dict[full_model_name] = False
        self.apply_low_memory(pipe, is_flux=True, device=device, full_model_name=full_model_name)
        print(f"✅ Loaded FLUX model {full_model_name} on device {device}")

        return pipe

    def get_flux_inpainting(self, device="cuda:0"):
        if Generator.flux_inpainting_pipe == None:
            from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
            from optimum.quanto import freeze, quantize, qfloat8
            
            print("📥 Loading FLUX inpainting pipeline...")
            print_detailed_memory_status("BEFORE FLUX inpainting loading")
            
            bl_repo = "black-forest-labs/FLUX.1-dev"
            txt_encoder_and_tokenizer_repo = "openai/clip-vit-large-patch14"
            dtype = torch.bfloat16
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bl_repo, subfolder="scheduler")
            text_encoder = CLIPTextModel.from_pretrained(txt_encoder_and_tokenizer_repo, torch_dtype=dtype)
            vae = AutoencoderKL.from_pretrained(bl_repo, subfolder="vae", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained(txt_encoder_and_tokenizer_repo, torch_dtype=dtype)
            text_encoder_2 = T5EncoderModel.from_pretrained(bl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
            tokenizer_2 = T5Tokenizer.from_pretrained(bl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
            transformer = FluxTransformer2DModel.from_pretrained(bl_repo, subfolder="transformer", torch_dtype=dtype)
            Generator.flux_inpainting_pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype= dtype,
                                                                                 text_encoder=text_encoder,
                                                                                 text_encoder_2=text_encoder_2,
                                                                                 transformer=transformer,
                                                                                 vae=vae,
                                                                                 scheduler=scheduler)

            self.apply_low_memory(Generator.flux_inpainting_pipe, is_flux=True, device=device, full_model_name="flux_inpainting")
            Generator.flux_inpainting_pipe.to(device)
            print("✅ FLUX inpainting pipeline loaded")
        return Generator.flux_inpainting_pipe

    def get_flux_inpainting_pipe_from_pipe(self, model_name: str, device: str = "cuda:0"):
        print(f"Getting flux inpainting pipe from model name {model_name}")
        if self.flux_inpainting_pipe != None:
            return self.flux_inpainting_pipe
        pipe = self.pipe_items.get(model_name)
        if(pipe is not None):
            self.flux_inpainting_pipe = FluxInpaintPipeline.from_pipe(pipe)
            return self.flux_inpainting_pipe
        return None

    def get_pipe_from_model_name(self, model_name: str, model_type: ModelType = None, device: str = "cuda:0"):
        print(f"🔍 Getting pipe from model name {model_name}")
        full_model_name = self.get_model_file_from_model_name(model_name)
        self.model_type = model_type
        if(full_model_name == None):
            return None
        pipe = self.pipe_items.get(model_name)
        if(pipe is not None):
            pipe.last_used = time.time()
            return pipe
        
        if(self.pipe_items.get(model_name) == None):
            print_detailed_memory_status("BEFORE loading new model")
            
            if model_type == ModelType.SDXL:
                print(f"📥 Loading SDXL model: {model_name}")
                _pipe = StableDiffusionXLPipeline.from_single_file(full_model_name,
                                                              torch_dtype=torch.float16,
                                                              from_single_file=True,
                                                              use_safetensors=True,
                                                              safety_checks=None,
                                                              )
                if model_name.lower().find("turbo") > -1:
                    _pipe.upcast_vae()
            else:
                print(f"📥 Loading FLUX model: {model_name}")
                _pipe = self.load_flux_properly(full_model_name, device)
                
            _pipe.to(device)
            print(f"✅ Loaded model {model_name} on device {device}")
            
            print_detailed_memory_status("AFTER loading new model")
            
            self.pipe_items[model_name] = PipeItem(model_name, model_type, device=device)
            self.pipe_items[model_name].pipe = _pipe
            
            if(len(self.pipe_items) > self.get_max_pipe_items()):
                # find the oldest pipe and delete it
                last_used = time.time()
                for key in self.pipe_items.keys():
                    if(self.pipe_items[key].last_used < last_used):
                        last_used = self.pipe_items[key].last_used
                        old_model_name = key
                print(f"🗑️  Memory limit reached, disposing oldest model: {old_model_name}")
                self.dispose_pipe(self.pipe_items[old_model_name].pipe, old_model_name)

                print(f"✓ Deleted model {old_model_name} from memory")
            return self.pipe_items[model_name]

    def get_model_file_from_model_name(self, model_name: str):
        return get_model_by_name(model_name)
    
    def get_lora_file_from_lora_name(self, lora_name: str):
        return get_lora_by_name(lora_name)
    
    def get_embed_file_from_embed_name(self, embed_name: str):
        embeds = self.get_embeds(from_metadata=True)
        for embed in embeds:
            if(embed_name in embed):
                return embed
        return None

    def get_models(self, from_metadata: bool = True):
        if(from_metadata):
            models =  get_models_from_metadata()
            checks =  [x["fullPath"] for x in models if "Checkpoint" in x["modelType"]]
            return checks
        
        _ = self.get_folders()
        available_models = []
        for folder_list in Generator.model_folders:
            if(type(folder_list) == str):
                folder_list = [folder_list]
            for folder in folder_list:
                files = insensitive_glob(f"{folder}{os.path.sep}*xl*.safetensors")
                available_models.extend(files)
        return available_models
    
    def get_embeds(self, from_metadata: bool = True):
        if(from_metadata):
            models =  get_models_from_metadata()
            embeds =  [x["fullPath"] for x in models if "TextualInversion" in x["modelType"]]
            return embeds
        
        model_folders = self.get_folders()
        available_embeds = []
        for folder_list in Generator.embed_folders:
            if(type(folder_list) == str):
                folder_list = [folder_list]
            for folder in folder_list:
                files = insensitive_glob(f"{folder}{os.path.sep}*.pt")
                available_embeds.extend(files)
        return available_embeds
    
    def get_vaes(self, from_metadata: bool = False):
        if(from_metadata):
            models =  get_models_from_metadata()
            vaes =  [x["fullPath"] for x in models if "VAE" in x["modelType"]]
            return vaes
        model_folders = self.get_folders()
        available_vaes = []
        for folder_list in Generator.vae_folders:
            if(type(folder_list) == str):
                folder_list = [folder_list]
            for folder in folder_list:
                files = glob.glob(f"{folder}{os.path.sep}*.safetensors")
                available_vaes.extend(files)
        return available_vaes

    def get_upscalers(self):
        model_folders = self.get_folders()
        available_upscalers = []
        for folder_list in Generator.upscaler_folders:
            if(type(folder_list) == str):
                folder_list = [folder_list]
            for folder in folder_list:
                files = insensitive_glob(f"{folder}{os.path.sep}*.safetensors")
                available_upscalers.extend(files)
        return available_upscalers

    def get_controlnets(self):
        model_folders = self.get_folders()
        available_controlnets = []
        for folder_list in Generator.controlnet_folders:
            if(type(folder_list) == str):
                folder_list = [folder_list]
            for folder in folder_list:
                files = insensitive_glob(f"{folder}{os.path.sep}*.safetensors")
                available_controlnets.extend(files)
        return available_controlnets

    def get_loras(self, from_metadata: bool = True):
        if(from_metadata):
            models =  get_models_from_metadata()
            loras =  [x["fullPath"] for x in models if "LO" in x["modelType"].upper()]
            return loras
        
        model_folders = self.get_folders()
        available_loras = []
        for folder_list in Generator.lora_folders:
            if(type(folder_list) == str):
                folder_list = [folder_list]
            for folder in folder_list:
                files = insensitive_glob(f"{folder}{os.path.sep}*xl*.safetensors")
                available_loras.extend(files)
        return available_loras

    def get_lora_names_only(self, from_metadata: bool = True):
        full_lora_names = self.get_loras(from_metadata)
        lora_names = []
        for lora in full_lora_names:
            lora_names.append(os.path.basename(lora))
        return sorted(set(lora_names))

    def get_model_names_only(self, from_metadata: bool = True):
        full_model_names = self.get_models(from_metadata)
        model_names = []
        for model in full_model_names:
            model_names.append(os.path.basename(model))
        return sorted(set(model_names))

    def get_controlnet_names_only(self):
        full_controlnet_names = self.get_controlnets()
        controlnet_names = []
        for controlnet in full_controlnet_names:
            controlnet_names.append(os.path.basename(controlnet))
        return sorted(set(controlnet_names))

    def get_embed_names_only(self, from_metadata: bool = True):
        full_embed_names = self.get_embeds(from_metadata)
        embed_names = []
        for embed in full_embed_names:
            embed_names.append(os.path.basename(embed))
        return sorted(set(embed_names))
    
    def get_vae_names_only(self):
        full_vae_names = self.get_vaes()
        vae_names = []
        for vae in full_vae_names:
            vae_names.append(os.path.basename(vae))
        return sorted(set(vae_names))
    
    def get_upscaler_names_only(self):
        full_upscaler_names = self.get_upscalers()
        upscaler_names = []
        for upscaler in full_upscaler_names:
            upscaler_names.append(os.path.basename(upscaler))
        return sorted(set(upscaler_names))
    
    def get_folders(self):
        if(Generator.model_folders == None):
            Generator.model_folders = []
            Generator.lora_folders = []
            Generator.embed_folders = []
            Generator.vae_folders = []
            Generator.upscaler_folders = []
            Generator.controlnet_folders = []
            Generator.model_folders.append(os.path.join(models_root_models, "checkpoints"))
            Generator.lora_folders.append(os.path.join(models_root_models, "loras"))
            Generator.embed_folders.append(os.path.join(models_root_models, "embeddings"))
            Generator.vae_folders.append(os.path.join(models_root_models, "vae"))
            Generator.upscaler_folders.append(os.path.join(models_root_models, "upscale_models"))
            Generator.controlnet_folders.append(os.path.join(models_root_models, "controlnet"))
            def combine_folders(base_folder: str, sub_folder_list: list):
                combined_folders = []
                for sub_folder in sub_folder_list:
                    combined_folders.append(os.path.join(base_folder, sub_folder))
                return combined_folders
            try:
                with open(models_root_extra_models_yaml, 'r') as stream:
                    try:
                        data = yaml.safe_load(stream)
                        for key in data.keys():
                            base_path = data[key]["base_path"]
                            folders= data[key]
                            
                            Generator.model_folders.extend(combine_folders(base_path, filter(None, folders["checkpoints"].split("\n"))))
                            Generator.lora_folders.extend(combine_folders(base_path, filter(None, folders["loras"].split("\n"))))
                            Generator.embed_folders.extend(combine_folders(base_path, filter(None, folders["embeddings"].split("\n"))))
                            Generator.vae_folders.extend(combine_folders(base_path, filter(None, folders["vae"].split("\n"))))
                            Generator.upscaler_folders.extend(combine_folders(base_path, filter(None, folders["upscale_models"].split("\n"))))
                            Generator.controlnet_folders.extend(combine_folders(base_path, filter(None, folders["controlnet"].split("\n"))))
                    except yaml.YAMLError as exc:
                        print(exc)
            except:
                print("No extra models found")
        return Generator.model_folders, Generator.lora_folders, Generator.embed_folders, Generator.vae_folders, Generator.upscaler_folders, Generator.controlnet_folders

    def add_queue(self, prompt_info: PromptInfo):
        self.waiting_queue.append(prompt_info)
        if(len(self.waiting_queue) == 1):
            return self.run_queue()

    def run_queue(self):
        while(len(self.waiting_queue) > 0):
            prompt_info = self.waiting_queue.pop(0)
            pipe = self.get_pipe_from_model_name(prompt_info.model, model_type=prompt_info.model_type)
            if(pipe == None):
                print("Model not found")
                continue
            return pipe.run_prompt(prompt_info)