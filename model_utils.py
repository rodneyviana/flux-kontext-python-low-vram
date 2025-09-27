import os
import PIL
import PIL.Image
import numpy as np
import torch
import gc
import psutil
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from diffusers.hooks import apply_group_offloading
from cuda_utils import get_num_cuda_devices, get_gpu_total_memory, get_gpu_available_memory, get_gpu_temperature, garbage_collect
from dotenv import load_dotenv

load_dotenv()

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

def get_usable_ram_gb():
    """Calculate usable RAM in GB, accounting for Linux memory management"""
    ram = psutil.virtual_memory()
    
    # Method 1: Use available directly (conservative)
    available_direct = ram.available / 1024 / 1024 / 1024
    
    # Method 2: Calculate based on total - used (more optimistic)
    available_calculated = (ram.total - ram.used) / 1024 / 1024 / 1024
    
    # Method 3: Use total - used but leave some safety margin (20% of total)
    safety_margin = ram.total * 0.2  # 20% safety margin
    available_with_margin = (ram.total - ram.used - safety_margin) / 1024 / 1024 / 1024
    
    # Use the most optimistic but safe approach
    # If calculated is much higher than available, use calculated with margin
    if available_calculated > available_direct * 2:
        usable = max(available_with_margin, 0)
        print(f"  🔍 RAM calculation: Using calculated with margin ({usable:.1f}GB) instead of available ({available_direct:.1f}GB)")
    else:
        usable = available_direct
        print(f"  🔍 RAM calculation: Using direct available ({usable:.1f}GB)")
    
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
    
    # System RAM - Fix the calculation
    ram = psutil.virtual_memory()
    ram_total = ram.total / 1024 / 1024 / 1024
    ram_used = ram.used / 1024 / 1024 / 1024
    ram_available = ram.available / 1024 / 1024 / 1024
    usable_ram = get_usable_ram_gb()
    
    # Calculate percentage correctly
    ram_usage_percent = (ram_used / ram_total) * 100
    print(f"RAM: {ram_used:.2f}GB used / {ram_total:.2f}GB total ({ram_usage_percent:.1f}%), {ram_available:.2f}GB available, {usable_ram:.2f}GB usable")
    
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
    
    print("=" * 50)

def is_low_memory(device: str = "cuda:0") -> bool:
    cuda_index = get_cuda_index(device)
    gpu_total_memory = get_gpu_total_memory(cuda_index) / 1024 / 1024 / 1024
    gpu_available_memory = get_gpu_available_memory(cuda_index) / 1024 / 1024 / 1024
    
    # Check system RAM availability - Use usable RAM calculation
    usable_ram = get_usable_ram_gb()
    ram = psutil.virtual_memory()
    ram_total = ram.total / 1024 / 1024 / 1024
    ram_used = ram.used / 1024 / 1024 / 1024
    
    print(f"\nMemory Assessment for device {device}:")
    print(f"  GPU Total: {gpu_total_memory:.2f}GB, GPU Available: {gpu_available_memory:.2f}GB")
    print(f"  RAM Total: {ram_total:.2f}GB, RAM Used: {ram_used:.2f}GB, RAM Usable: {usable_ram:.2f}GB")
    
    # More conservative thresholds - need at least 8GB free GPU memory for inference
    # and sufficient RAM for potential offloading
    if gpu_total_memory < 40 or gpu_available_memory < 8 or usable_ram < 8:
        print(f"  → Low memory mode triggered: GPU={gpu_total_memory:.1f}GB total, {gpu_available_memory:.1f}GB available, RAM={usable_ram:.1f}GB usable")
        return True
    else:
        print(f"  → High memory mode: Sufficient resources available")
    return False

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

def get_memory_status():
    """Get comprehensive memory status"""
    status = {}
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(get_num_cuda_devices()):
            total = get_gpu_total_memory(i) / 1024 / 1024 / 1024
            available = get_gpu_available_memory(i) / 1024 / 1024 / 1024
            used = total - available
            status[f'gpu_{i}'] = {
                'total': total,
                'used': used,
                'available': available,
                'usage_percent': (used / total) * 100,
                'temperature': get_gpu_temperature(i)
            }
    
    # System RAM - Use usable RAM calculation
    ram = psutil.virtual_memory()
    ram_total = ram.total / 1024 / 1024 / 1024
    ram_used = ram.used / 1024 / 1024 / 1024
    usable_ram = get_usable_ram_gb()
    
    status['ram'] = {
        'total': ram_total,
        'used': ram_used,
        'available': ram.available / 1024 / 1024 / 1024,
        'usable': usable_ram,
        'usage_percent': (ram_used / ram_total) * 100
    }
    
    # Swap memory
    swap = psutil.swap_memory()
    swap_total = swap.total / 1024 / 1024 / 1024
    swap_used = swap.used / 1024 / 1024 / 1024
    swap_free = swap.free / 1024 / 1024 / 1024
    status['swap'] = {
        'total': swap_total,
        'used': swap_used,
        'free': swap_free,
        'usage_percent': (swap_used / swap_total) * 100 if swap_total > 0 else 0
    }
    
    return status

def apply_flux_optimized_offloading(pipe: FluxKontextPipeline, device: str = "cuda:0"):
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

def apply_low_memory(pipe: FluxKontextPipeline, device: str = "cuda:0"):
    device_id = get_cuda_index(device)
    print(f"\n🔧 Applying memory optimization for CUDA device {device_id}")
    
    # Show memory before optimization
    print_detailed_memory_status("BEFORE memory optimization")
    
    gpu_available = get_gpu_available_memory(device_id) / 1024 / 1024 / 1024
    gpu_total = get_gpu_total_memory(device_id) / 1024 / 1024 / 1024
    
    # Use usable RAM calculation instead of available
    usable_ram = get_usable_ram_gb()
    ram = psutil.virtual_memory()
    ram_total = ram.total / 1024 / 1024 / 1024
    ram_used = ram.used / 1024 / 1024 / 1024
    
    print(f"\n🎯 Memory Analysis:")
    print(f"  GPU: {gpu_available:.2f}GB available / {gpu_total:.2f}GB total")
    print(f"  RAM: {usable_ram:.2f}GB usable / {ram_total:.2f}GB total (used: {ram_used:.2f}GB)")
    
    # Use usable RAM instead of available RAM for decisions
    if gpu_total >= 30 and usable_ram > 8:  # Much more realistic threshold
        # RTX 5090 or similar high-end GPU - use optimized strategy
        print("🚀 Strategy: RTX 5090 OPTIMIZED - selective component placement")
        print(f"  Detected high-end GPU with {gpu_total:.0f}GB VRAM and {usable_ram:.1f}GB usable RAM")
        
        # DON'T use pipe.to(device) - this loads everything to GPU
        # Instead, strategically place components
        apply_flux_optimized_offloading(pipe, device)
        
    elif gpu_available > 15 and usable_ram > 6:
        # Mid-range GPU - standard offloading
        print("⚖️  Strategy: STANDARD MEMORY - selective offloading")
        print(f"  GPU available: {gpu_available:.2f}GB, RAM usable: {usable_ram:.2f}GB")
        
        # Move main pipeline to GPU but offload heavy components
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
        
    elif usable_ram > 4:
        # Low GPU memory - full offloading
        print("💾 Strategy: LOW MEMORY - full offloading to CPU")
        print(f"  GPU available: {gpu_available:.2f}GB, RAM usable: {usable_ram:.2f}GB")
        
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
        print("  ✓ Full offloading completed")
        
    else:
        # Critical memory situation
        print("❌ Strategy: CRITICAL MEMORY - insufficient resources")
        print(f"  GPU available: {gpu_available:.2f}GB, RAM usable: {usable_ram:.2f}GB")
        raise RuntimeError(f"Insufficient memory: GPU={gpu_available:.1f}GB, RAM={usable_ram:.1f}GB usable. Close other applications.")
    
    # Show memory after optimization
    print_detailed_memory_status("AFTER memory optimization")

def check_inference_memory(device: str = "cuda:0"):
    """Check if there's enough memory for inference"""
    device_id = get_cuda_index(device)
    gpu_available = get_gpu_available_memory(device_id) / 1024 / 1024 / 1024
    
    print(f"\n🔍 Pre-inference memory check:")
    print(f"  GPU available: {gpu_available:.2f}GB")
    
    # Need at least 6GB free for inference (more conservative for RTX 5090)
    if gpu_available < 6:
        print(f"⚠️  WARNING: Low GPU memory for inference: {gpu_available:.2f}GB available")
        print("🧹 Performing emergency cleanup...")
        deep_memory_cleanup()
        
        # Check again after cleanup
        gpu_available_after = get_gpu_available_memory(device_id) / 1024 / 1024 / 1024
        if gpu_available_after < 4:
            raise RuntimeError(f"Insufficient GPU memory for inference: {gpu_available_after:.2f}GB available. Need at least 4GB.")
        else:
            print(f"✓ Memory freed: {gpu_available_after:.2f}GB now available")
    else:
        print(f"✅ Sufficient GPU memory for inference: {gpu_available:.2f}GB available")

_pipe = None
def load_model(device: str = "cuda:0") -> FluxKontextPipeline:
    """Load the FLUX model pipeline with optimized memory management""" 
    global _pipe

    model_name = os.getenv("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-Kontext-dev")
    if _pipe is not None:
        print("♻️  Model already loaded, returning existing pipeline")
        print_detailed_memory_status("CURRENT")
        return _pipe
    
    cuda_index = get_cuda_index(device)
    print(f"\n🤖 Loading FLUX Kontext model on CUDA device {cuda_index}")
    
    # Show initial memory state
    print_detailed_memory_status("BEFORE model loading")
    
    print("📥 Downloading/Loading model from Hugging Face...")
    _pipe = FluxKontextPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print("✓ Model loaded successfully")
    
    # Show memory after model loading but before optimization
    print_detailed_memory_status("AFTER model loading, BEFORE optimization")
    
    apply_low_memory(_pipe, device=device)
    
    print("✅ Model loading and optimization completed!")
    return _pipe

def run_prompt(pipe: FluxKontextPipeline, prompt: str, input_image: PIL.Image.Image | np.ndarray, guidance_scale: float = 2.5, num_inference_steps: int = 28):
    if pipe is None:
        raise ValueError("Pipeline is not loaded. Please load the model first.")
    
    print(f"\n🎨 Running prompt: '{prompt}'")
    print(f"⚙️  Parameters: guidance_scale={guidance_scale}, steps={num_inference_steps}")
    
    # Show memory before inference
    print_detailed_memory_status("BEFORE inference")
    
    # Check if we have enough memory for inference
    try:
        check_inference_memory()
        
        print("🔮 Starting image generation...")
        image = pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        print("✓ Image generation completed")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA OOM Error: {e}")
        print("🚨 Attempting emergency recovery...")
        
        # Emergency cleanup
        deep_memory_cleanup()
        
        # Try to apply more aggressive offloading
        print("🔄 Applying emergency offloading...")
        device = str(pipe.device) if hasattr(pipe, 'device') else "cuda:0"
        
        # Offload everything we can
        try:
            apply_group_offloading(
                pipe.transformer,
                offload_type="leaf_level", 
                offload_device=torch.device("cpu"),
                onload_device=torch.device(device),
                use_stream=True,
            )
            print("    ✓ Emergency transformer offload applied")
        except Exception as offload_error:
            print(f"    ⚠️  Could not offload transformer: {offload_error}")
        
        raise RuntimeError("CUDA out of memory during inference. Try closing other applications or using a smaller model.")
    
    # Show memory after inference but before cleanup
    print_detailed_memory_status("AFTER inference, BEFORE cleanup")
    
    # Use comprehensive cleanup
    deep_memory_cleanup()
    
    print("🎉 Prompt execution completed successfully!\n")
    return image