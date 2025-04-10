# -*- coding: utf-8 -*-
# HiDream Sampler Node for ComfyUI
# Version: 2024-07-29d (Removed auto-gptq dependency check)
#
# Required Dependencies:
# - transformers, diffusers, torch, numpy, Pillow
# - For NF4 (GPTQ) models: optimum, accelerate (`pip install optimum accelerate`)
#   * Note: Optimum might require additional backends like exllama (`pip install optimum[exllama]`) depending on your setup.
# - For non-NF4/FP8 models (4-bit): bitsandbytes (`pip install bitsandbytes`)
# - Ensure hi_diffusers library is locally available or hdi1 package is installed.
import torch
import numpy as np
from PIL import Image
import comfy.model_management # Ensure this is imported
import comfy.utils
import gc
import os # For checking paths if needed
import huggingface_hub
from safetensors.torch import load_file

# --- Optional Dependency Handling ---
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False
    print("Warning: accelerate not installed. device_map='auto' for GPTQ models may not work optimally.")

# REMOVED: auto-gptq check is removed. We rely on Optimum now.
# try:
#     import auto_gptq
#     autogptq_available = True
# except ImportError:
#     autogptq_available = False
#     # Note: Optimum might still load GPTQ without auto-gptq if using ExLlama kernels,
#     # but it's often required. Add a warning if NF4 models are selected later.

try:
    import optimum
    optimum_available = True
    print("Optimum library found. GPTQ model loading enabled (requires suitable backend).")
except ImportError:
    optimum_available = False
    print("Warning: optimum not installed. GPTQ models (NF4 variants) will be disabled.")

try:
    # Import specific classes to avoid potential namespace conflicts later
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_available = True
except ImportError:
    bnb_available = False
    # Keep placeholders None to avoid errors later if bnb not available
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None
    print("Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available.")

# --- Core Imports ---
from transformers import LlamaForCausalLM, AutoTokenizer # Use AutoTokenizer

# --- HiDream Specific Imports ---
# Attempt local import first, then fallback (which might fail)
try:
    # Assuming hi_diffusers is cloned into this custom_node's directory
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    hidream_classes_loaded = True
except ImportError as e:
    print("--------------------------------------------------------------------")
    print(f"ComfyUI-HiDream-Sampler: Could not import local hi_diffusers ({e}).")
    print("Please ensure hi_diffusers library is inside ComfyUI-HiDream-Sampler,")
    print("or hdi1 package is installed in the ComfyUI environment.")
    print("Node may fail to load models.")
    print("--------------------------------------------------------------------")
    # Define placeholders so the script doesn't crash immediately
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None
    FlowUniPCMultistepScheduler = None
    FlashFlowMatchEulerDiscreteScheduler = None
    hidream_classes_loaded = False

# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" # For original/FP8
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" # For NF4
# Add uncensored model paths (using the same model as NF4 since it's less censored)
UNCENSORED_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
UNCENSORED_NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# --- Model Configurations ---
# Added flags for dependency checks
# requires_gptq_deps now means "requires Optimum for loading GPTQ format"
MODEL_CONFIGS = {
    # --- NF4 Models (Require Optimum) ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler", # Use string names for dynamic import
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True # Requires Optimum
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True # Requires Optimum
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True # Requires Optimum
    },
    # --- Original/BNB Models (Require BitsAndBytes) ---
     "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    }
}

# --- Filter models based on available dependencies ---
original_model_count = len(MODEL_CONFIGS)
if not bnb_available:
    print("Filtering out models requiring BitsAndBytes.")
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_bnb", False)}
# MODIFIED: Only check for optimum for GPTQ models
if not optimum_available:
    print("Filtering out models requiring Optimum (NF4/GPTQ variants).")
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_gptq_deps", False)}
if not hidream_classes_loaded:
    print("Filtering out all models because hi_diffusers classes failed to load.")
    MODEL_CONFIGS = {} # Clear all if core classes missing

filtered_model_count = len(MODEL_CONFIGS)
if filtered_model_count == 0 and hidream_classes_loaded: # Only show critical error if classes loaded but no models usable
    print("*"*70)
    print("CRITICAL ERROR: No HiDream models available due to missing dependencies (Optimum/BitsAndBytes).")
    print("Please install the required libraries:")
    print("  - For NF4 models: pip install optimum accelerate")
    print("    (Optimum might need specific backends like exllama: pip install optimum[exllama])")
    print("  - For standard models: pip install bitsandbytes")
    print("*"*70)
elif filtered_model_count < original_model_count:
    print("*"*70)
    print("Warning: Some HiDream models are disabled due to missing optional dependencies (Optimum/BitsAndBytes).")
    print("Check console warnings above for details.")
    print("*"*70)

# Define BitsAndBytes configs (if available)
bnb_llm_config = None
bnb_transformer_4bit_config = None
if bnb_available and TransformersBitsAndBytesConfig and DiffusersBitsAndBytesConfig:
    try:
        bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
        bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
        print("BitsAndBytes 4-bit configurations prepared.")
    except Exception as e:
        print(f"Warning: Failed to create BitsAndBytes configs: {e}")
        bnb_available = False # Disable BNB if config creation fails

model_dtype = torch.bfloat16

# Get available scheduler classes
available_schedulers = {}
if hidream_classes_loaded:
    available_schedulers = {
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler
    }

# --- Helper: Get Scheduler Instance ---
def get_scheduler_instance(scheduler_name, shift_value):
    if not available_schedulers:
        raise RuntimeError("No schedulers available (HiDream classes might have failed to load).")

    scheduler_class = available_schedulers.get(scheduler_name)
    if scheduler_class is None:
        raise ValueError(f"Scheduler class '{scheduler_name}' not found in available schedulers: {list(available_schedulers.keys())}")

    # Ensure parameters are appropriate for the class
    # Example: Check if shift is relevant for this scheduler
    # (For now, assume it's okay, but real code might need more checks)
    try:
        instance = scheduler_class(num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False)
        print(f"Scheduler instance created: {scheduler_name} with shift={shift_value}")
        return instance
    except Exception as e:
        print(f"Error creating scheduler instance '{scheduler_name}': {e}")
        raise # Re-raise the exception

# --- Loading Function (Handles NF4, FP8, and default BNB) ---
def load_models(model_type, use_uncensored_llm):
    if not hidream_classes_loaded:
        raise ImportError("Cannot load models: HiDream classes failed to import.")
    if model_type not in MODEL_CONFIGS:
        # Check if it was filtered out
        if model_type in ["full-nf4", "dev-nf4", "fast-nf4"] and not optimum_available:
             raise ValueError(f"Model type '{model_type}' requires Optimum, which is not installed or failed to load.")
        if model_type in ["full", "dev", "fast"] and not bnb_available:
             raise ValueError(f"Model type '{model_type}' requires BitsAndBytes, which is not installed or failed to load.")
        raise ValueError(f"Unknown or incompatible model_type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)
    requires_gptq_deps = config.get("requires_gptq_deps", False) # This now means "requires Optimum"

    # Dependency checks specific to this model type
    if requires_bnb and not bnb_available:
        raise ImportError(f"Model '{model_type}' requires BitsAndBytes, but it's not available or failed to initialize.")
    # MODIFIED: Only check for optimum here
    if requires_gptq_deps and not optimum_available:
        raise ImportError(f"Model '{model_type}' requires the Optimum library for GPTQ loading, but it's not available.")

    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"NF4 (GPTQ): {is_nf4}, Requires BNB: {requires_bnb}, Requires Optimum: {requires_gptq_deps}")

    start_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(Start VRAM: {start_mem:.2f} MB)")

    # --- 1. Load LLM (Conditional) ---
    text_encoder_load_kwargs = {
        # "output_hidden_states": True, # Often not needed for inference, reduces memory slightly
        "low_cpu_mem_usage": True,
        "torch_dtype": model_dtype,
    }
    tokenizer = None
    text_encoder = None

    if is_nf4: # NF4 / GPTQ Loading Path
        if not optimum_available: raise ImportError("Optimum is required for NF4/GPTQ models.")
        if use_uncensored_llm:
            llama_model_name = UNCENSORED_NF4_LLAMA_MODEL_NAME
        else:
            llama_model_name = NF4_LLAMA_MODEL_NAME
        print(f"\n[1a] Preparing LLM (GPTQ via Optimum): {llama_model_name}")
        if accelerate_available:
            text_encoder_load_kwargs["device_map"] = "auto"
            print("     Using device_map='auto' (requires accelerate).")
        else:
            print("     accelerate not found. Model will be loaded onto the default device (likely CUDA if available).")
            # No device_map, will manually move later if needed

        # No explicit quantization_config needed here; Optimum handles it based on model config
        print(f"     Loading Tokenizer: {llama_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False) # GPTQ often prefers slow tokenizers
        print("     Tokenizer loaded.")

        print(f"[1c] Loading Text Encoder (GPTQ): {llama_model_name}... (May download files/require backend)")
        # This is where Optimum hooks in implicitly if installed
        text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
        # Note: If Optimum isn't set up correctly (e.g., missing exllama), this step might fail.

    else: # Standard / BNB Loading Path
        if not bnb_available: raise ImportError("BitsAndBytes is required for standard 4-bit models.")
        if use_uncensored_llm:
            # Currently uses the same GPTQ model even for non-NF4 uncensored.
            # This might need adjustment if a non-GPTQ uncensored model is preferred.
            # For now, we will load the specified UNCENSORED_LLAMA_MODEL_NAME (which is GPTQ)
            # but apply BNB config to it IF it weren't GPTQ. This path needs care.
            # Let's assume for now UNCENSORED_LLAMA_MODEL_NAME should match ORIGINAL_LLAMA_MODEL_NAME
            # if we strictly want BNB loading here. Reverting to original for clarity:
            # llama_model_name = UNCENSORED_LLAMA_MODEL_NAME # This is GPTQ, might conflict
            llama_model_name = ORIGINAL_LLAMA_MODEL_NAME # Sticking to non-GPTQ base for BNB path
            print(f"\n[1a] Preparing LLM (4-bit BNB) - Uncensored Fallback: {llama_model_name}")
        else:
            llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing LLM (4-bit BNB): {llama_model_name}")

        if bnb_llm_config:
            text_encoder_load_kwargs["quantization_config"] = bnb_llm_config
            print("     Using 4-bit BNB quantization config for LLM.")
        else:
            raise ImportError("BNB config required for standard LLM loading but is unavailable.")

        # Add flash attention 2 if available (good for non-quantized or BNB attention)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
             text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2"
             print("     Using Flash Attention 2 implementation.")
        else:
             text_encoder_load_kwargs["attn_implementation"] = "eager"
             print("     Using eager attention implementation.")

        print(f"[1b] Loading Tokenizer: {llama_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False) # Slow tokenizer often safer
        print("     Tokenizer loaded.")

        print(f"[1c] Loading Text Encoder (BNB): {llama_model_name}... (May download files)")
        text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)


    # Manual device placement if device_map wasn't used (or failed)
    if "device_map" not in text_encoder_load_kwargs:
        target_device = comfy.model_management.get_torch_device()
        print(f"     Moving text encoder manually to device: {target_device}...")
        try:
            text_encoder.to(target_device)
        except Exception as e:
            print(f"     Error moving text encoder to {target_device}: {e}. Trying 'cuda'.")
            try:
                text_encoder.to("cuda") # Fallback attempt
            except Exception as e2:
                print(f"     Error moving text encoder to 'cuda': {e2}. Model might stay on CPU.")


    step1_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Text encoder loaded! (VRAM: {step1_mem:.2f} MB)")

    # --- 2. Load Transformer (Conditional) ---
    print(f"\n[2] Preparing Transformer from: {model_path}")
    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True
    }
    transformer = None

    if is_nf4:
        # NF4 transformer doesn't use explicit quantization config here
        print("     Type: NF4 (using specified model_dtype)")
        # No quantization_config added
    else:  # Default BNB case for transformer
        if not bnb_available: raise ImportError("BitsAndBytes is required for standard transformer 4-bit quantization.")
        print("     Type: Standard (Applying 4-bit BNB quantization)")
        if bnb_transformer_4bit_config:
            transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
            print("     Using 4-bit BNB quantization config for Transformer.")
        else:
            raise ImportError("BNB config required for transformer but unavailable.")

    print("     Loading Transformer... (May download files)")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(model_path, **transformer_load_kwargs)

    target_device = comfy.model_management.get_torch_device()
    print(f"     Moving Transformer to device: {target_device}...")
    try:
        transformer.to(target_device)
    except Exception as e:
        print(f"     Error moving transformer to {target_device}: {e}. Trying 'cuda'.")
        try:
             transformer.to("cuda")
        except Exception as e2:
             print(f"     Error moving transformer to 'cuda': {e2}. Model might stay on CPU.")

    step2_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Transformer loaded! (VRAM: {step2_mem:.2f} MB)")

    # --- 3. Load Scheduler ---
    print(f"\n[3] Preparing Scheduler: {scheduler_name}")
    scheduler = get_scheduler_instance(scheduler_name, shift)
    print(f"     Using Scheduler: {scheduler_name} with shift={shift}")

    # --- 4. Load Pipeline ---
    print(f"\n[4] Loading Pipeline structure from: {model_path}")
    print("     Passing pre-loaded components (tokenizer, text_encoder)...")
    pipe = HiDreamImagePipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        # Pass the loaded components directly
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        transformer=None, # Will assign the loaded transformer later
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        # Variants might have different expected encoders, be careful
        # Assuming tokenizer_4/text_encoder_4 maps to the Llama model here
    )
    print("     Pipeline structure loaded.")

    # --- 5. Final Setup ---
    print("\n[5] Finalizing Pipeline...")
    print("     Assigning pre-loaded transformer...")
    pipe.transformer = transformer # Assign the transformer we loaded

    # The pipeline object itself doesn't usually need moving if components are correct,
    # but some internal states might benefit. Let's try moving the whole pipe object.
    target_device = comfy.model_management.get_torch_device()
    print(f"     Moving pipeline object to {target_device} (final check)...")
    try:
        pipe.to(target_device)
    except Exception as e:
        print(f"     Warning: Could not move the main pipeline object to {target_device}: {e}.")

    # Enable CPU offloading *if* using accelerate's device_map and it's supported
    # Generally applicable to large models, especially with device_map='auto'
    if accelerate_available and "device_map" in text_encoder_load_kwargs:
        print("     Attempting to enable CPU offload (requires accelerate)...")
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                # Offload guidance can be model-specific, check pipeline docs if issues arise
                pipe.enable_sequential_cpu_offload(gpu_id=0) # Assume GPU 0 if multiple
                print("     ✅ Sequential CPU offload enabled.")
            except Exception as e:
                print(f"     ⚠️ Failed to enable sequential CPU offload: {e}")
        else:
            print("     ⚠️ enable_sequential_cpu_offload() not found on pipeline.")
        # Also consider model offload if memory is still tight
        # if hasattr(pipe, "enable_model_cpu_offload"):
        #    pipe.enable_model_cpu_offload()


    final_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    vram_increase = final_mem - start_mem
    print(f"✅ Pipeline ready! (VRAM: {final_mem:.2f} MB / Increase: {vram_increase:.2f} MB)")

    return pipe, config

# --- Resolution Parsing & Tensor Conversion ---
RESOLUTION_OPTIONS = [ # (Keep list the same)
    "1024 × 1024 (Square)","768 × 1360 (Portrait)","1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)","1168 × 880 (Landscape)","1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]
def parse_resolution(resolution_str):
    """Parse resolution string into height and width dimensions."""
    try:
        # Extract the resolution part before the parenthesis
        res_part = resolution_str.split(" (")[0].strip()
        # Replace 'x' with '×' for consistency if needed
        parts = res_part.replace('x', '×').split("×")

        if len(parts) != 2:
            raise ValueError(f"Expected format 'width × height', got '{res_part}'")

        width_str = parts[0].strip()
        height_str = parts[1].strip()

        width = int(width_str)
        height = int(height_str)
        # print(f"Successfully parsed resolution: {width}x{height}") # Less verbose
        return height, width
    except Exception as e:
        print(f"Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024 # Default fallback

def pil2tensor(image: Image.Image):
    """Convert PIL image to tensor [1, H, W, 3] in float32 format"""
    if image is None:
        print("pil2tensor: Image is None, cannot convert.")
        return None
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            # print(f"Converting image from {image.mode} to RGB") # Less verbose
            image = image.convert('RGB')

        # Convert to numpy array
        np_image = np.array(image).astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        # Shape needs to be [Batch, Height, Width, Channel] for ComfyUI
        tensor_image = torch.from_numpy(np_image).unsqueeze(0)

        # print(f"pil2tensor: Converted image to tensor with shape {tensor_image.shape}") # Less verbose
        return tensor_image
    except Exception as e:
        print(f"Error in pil2tensor: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- ComfyUI Node Definition ---
class HiDreamSampler:
    _model_cache = {}
    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            # Provide a more informative error if no models could be listed
            error_message = "No HiDream models available. Check console for missing dependencies (Optimum/BitsAndBytes) or errors loading hi_diffusers."
            return {"required": {"error": ("STRING", {"default": error_message, "multiline": True})}}

        # Try to find a sensible default, preferring nf4 if available
        default_model = "fast-nf4" if "fast-nf4" in available_model_types else \
                        "dev-nf4" if "dev-nf4" in available_model_types else \
                        "full-nf4" if "full-nf4" in available_model_types else \
                        "fast" if "fast" in available_model_types else \
                        "dev" if "dev" in available_model_types else \
                        "full" if "full" in available_model_types else \
                        available_model_types[0] # Fallback to first available

        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful fantasy landscape"}),
                "fixed_resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 200}), # Increased max steps
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
                "override_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}), # Increased max size
                "override_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    # --- Cache Management ---
    @classmethod
    def cleanup_cache(cls, exclude_key=None):
        """Removes models from cache, optionally excluding one key."""
        keys_to_del = list(cls._model_cache.keys())
        if exclude_key and exclude_key in keys_to_del:
            keys_to_del.remove(exclude_key)

        if not keys_to_del:
            return # Nothing to clear except the excluded key

        print(f"HiDream Sampler: Cleaning up model cache (excluding '{exclude_key}')...")
        for key in keys_to_del:
            print(f"  Removing '{key}' from cache...")
            try:
                pipe_to_del, config_to_del = cls._model_cache.pop(key, (None, None))
                # Attempt to explicitly delete components to help GC
                if pipe_to_del:
                    if hasattr(pipe_to_del, 'transformer'): pipe_to_del.transformer = None
                    if hasattr(pipe_to_del, 'text_encoder_4'): pipe_to_del.text_encoder_4 = None
                    if hasattr(pipe_to_del, 'tokenizer_4'): pipe_to_del.tokenizer_4 = None
                    # Add other encoders if necessary
                    if hasattr(pipe_to_del, 'scheduler'): pipe_to_del.scheduler = None
                    del pipe_to_del
                del config_to_del # Delete config dict too
            except Exception as e:
                print(f"  Error during cleanup of {key}: {e}")
        # Force garbage collection and CUDA cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # torch.cuda.synchronize() # Might be too slow/aggressive here
        print("Model cache cleanup complete.")


    def generate(self, model_type, prompt, fixed_resolution, seed, override_steps, override_cfg, override_width, override_height, **kwargs):
        # print("DEBUG: HiDreamSampler.generate() called") # Less verbose debug
        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Error: No models available or error state selected.")
            # Return a blank image matching a common default or the override if provided
            error_h = override_height if override_height > 0 else 1024
            error_w = override_width if override_width > 0 else 1024
            return (torch.zeros((1, error_h, error_w, 3), dtype=torch.float32),)

        pipe = None
        config = None
        # Use uncensored flag from kwargs if present (for compatibility with Advanced node maybe)
        # Defaulting to False for the basic sampler.
        use_uncensored_llm = kwargs.get('use_uncensored_llm', False)
        cache_key = f"HiDreamSampler_{model_type}_{'uncensored' if use_uncensored_llm else 'standard'}"

        # --- Model Loading / Caching ---
        if cache_key in self._model_cache:
            # print(f"Checking cache for {cache_key}...") # Less verbose
            pipe, config = self._model_cache[cache_key]
            # Basic validation: Ensure pipe and required components exist
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None or not hasattr(pipe, 'text_encoder_4') or pipe.text_encoder_4 is None:
                print(f"Invalid cache entry for {cache_key}, removing and reloading...")
                del self._model_cache[cache_key]
                pipe, config = None, None # Force reload
            else:
                print(f"Using cached model: {cache_key}")

        if pipe is None:
            # Clean up *other* models before loading a new one
            self.cleanup_cache(exclude_key=None) # Clean all before loading new

            print(f"Loading model for {cache_key}...")
            try:
                # Load the model
                pipe, config = load_models(model_type, use_uncensored_llm)
                # Store in cache
                self._model_cache[cache_key] = (pipe, config)
                print(f"Model {cache_key} loaded and cached successfully!")
            except Exception as e:
                print(f"!!! ERROR loading model '{model_type}': {e}")
                import traceback
                traceback.print_exc()
                # Return blank image on loading failure
                error_h = override_height if override_height > 0 else 1024
                error_w = override_width if override_width > 0 else 1024
                return (torch.zeros((1, error_h, error_w, 3), dtype=torch.float32),)

        # Final check after loading or retrieval from cache
        if pipe is None or config is None:
            print(f"CRITICAL ERROR: Model pipe or config is None after loading/cache check for {cache_key}.")
            error_h = override_height if override_height > 0 else 1024
            error_w = override_width if override_width > 0 else 1024
            return (torch.zeros((1, error_h, error_w, 3), dtype=torch.float32),)

        # --- Generation Setup ---
        # Determine resolution
        if override_width > 0 and override_height > 0:
            height, width = override_height, override_width
            print(f"Using override resolution: {width}x{height}")
        else:
            height, width = parse_resolution(fixed_resolution)
            print(f"Using fixed resolution: {width}x{height} ({fixed_resolution})")

        # Determine steps and CFG scale
        num_inference_steps = override_steps if override_steps >= 1 else config["num_inference_steps"] # Ensure steps >= 1
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # Progress bar
        pbar = comfy.utils.ProgressBar(num_inference_steps)

        # Generator device placement
        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using inference device: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)

        print(f"\n--- Starting Generation ---")
        print(f"  Model: {model_type}, Res: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        # --- Run Inference ---
        output_images = None
        try:
            # Ensure model components are on the correct device just before inference
            # This is important if CPU offloading or manual moves happened
            target_device = comfy.model_management.get_torch_device()
            if hasattr(pipe, 'transformer') and pipe.transformer.device != target_device:
                 print(f"Moving transformer to {target_device} before inference...")
                 pipe.transformer.to(target_device)
            if hasattr(pipe, 'text_encoder_4') and pipe.text_encoder_4.device.type != target_device.type: # device_map uses index, check type
                 print(f"Moving text_encoder_4 to {target_device} before inference...")
                 pipe.text_encoder_4.to(target_device)
            # Add checks for other encoders if the pipeline uses them

            print("Executing pipeline inference...")
            # Use inference_mode for efficiency
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    # Pass max sequence length based on model type?
                    # Defaulting here, but advanced node handles this better
                    max_sequence_length=128, # Default from original code
                    # Add callback for progress bar
                    callback_on_step_end=lambda *args: pbar.update(1) or {}
                )
                output_images = output.images

            print("Pipeline inference finished.")

        except Exception as e:
            print(f"!!! ERROR during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            # Return blank image on execution failure
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)
        finally:
            # Ensure pbar completes fully even if callback didn't reach the end
            pbar.update_absolute(num_inference_steps, num_inference_steps)


        print("--- Generation Complete ---")

        # --- Process Output ---
        if output_images is None or len(output_images) == 0 or output_images[0] is None:
            print("ERROR: No valid image returned from the pipeline. Returning blank image.")
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

        try:
            # print(f"Processing output image. Type: {type(output_images[0])}") # Less verbose
            # Convert the first image (PIL) to tensor
            output_tensor = pil2tensor(output_images[0])

            if output_tensor is None:
                print("ERROR: pil2tensor failed to convert the output image. Returning blank image.")
                return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

            # Ensure correct dtype (ComfyUI expects float32)
            if output_tensor.dtype != torch.float32:
                print(f"Converting output tensor from {output_tensor.dtype} to float32.")
                output_tensor = output_tensor.to(torch.float32)

            # Verify tensor shape [1, H, W, 3]
            if len(output_tensor.shape) != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[3] != 3:
                print(f"ERROR: Invalid tensor shape {output_tensor.shape} after conversion. Expected [1, H, W, 3]. Returning blank image.")
                return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

            print(f"Successfully processed output image to tensor shape: {output_tensor.shape}")

            # Optional: Soft cleanup after generation
            # comfy.model_management.soft_empty_cache() # Can sometimes help release VRAM faster

            return (output_tensor,)

        except Exception as e:
            print(f"!!! Error processing output image: {e}")
            import traceback
            traceback.print_exc()
            # Return blank image on processing failure
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)


# --- ComfyUI Node 2 Definition (Advanced) ---
class HiDreamSamplerAdvanced:
    _model_cache = HiDreamSampler._model_cache  # Share model cache
    cleanup_cache = HiDreamSampler.cleanup_cache # Share cleanup method

    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            error_message = "No HiDream models available. Check console for missing dependencies (Optimum/BitsAndBytes) or errors loading hi_diffusers."
            return {"required": {"error": ("STRING", {"default": error_message, "multiline": True})}}

        default_model = "fast-nf4" if "fast-nf4" in available_model_types else \
                        "dev-nf4" if "dev-nf4" in available_model_types else \
                        "full-nf4" if "full-nf4" in available_model_types else \
                        "fast" if "fast" in available_model_types else \
                        "dev" if "dev" in available_model_types else \
                        "full" if "full" in available_model_types else \
                        available_model_types[0] # Fallback to first available

        # Schedulers: Use names that match the classes used internally
        scheduler_options = [
            "Default for model", # Will use config["scheduler_class"]
            "FlowUniPCMultistepScheduler",
            "FlashFlowMatchEulerDiscreteScheduler",
            # Add variants if the schedulers support them via flags
            "FlashFlowMatchEulerDiscreteScheduler (Karras)",
            "FlashFlowMatchEulerDiscreteScheduler (Exponential)"
        ]

        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "primary_prompt": ("STRING", {"multiline": True, "default": "cinematic photo of a majestic cat astronaut exploring a nebula"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "ugly, deformed, blurry, low quality, text, watermark, signature"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}), # Wider range
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (scheduler_options, {"default": "Default for model"}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 200}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
                "use_uncensored_llm": ("BOOLEAN", {"default": False}) # Exposed uncensored toggle
            },
            "optional": { # Keep these optional as before
                "clip_l_prompt": ("STRING", {"multiline": True, "default": ""}),
                "openclip_prompt": ("STRING", {"multiline": True, "default": ""}),
                "t5_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llama_prompt": ("STRING", {"multiline": True, "default": ""}), # This is prompt_4
                "max_length_clip_l": ("INT", {"default": 77, "min": 64, "max": 308}), # Increased max length
                "max_length_openclip": ("INT", {"default": 77, "min": 64, "max": 308}),
                "max_length_t5": ("INT", {"default": 128, "min": 64, "max": 512}),
                "max_length_llama": ("INT", {"default": 128, "min": 64, "max": 2048}) # Llama often handles longer
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    def generate(self, model_type, primary_prompt, negative_prompt, width, height, seed, scheduler,
                 override_steps, override_cfg, use_uncensored_llm=False,
                 clip_l_prompt="", openclip_prompt="", t5_prompt="", llama_prompt="",
                 max_length_clip_l=77, max_length_openclip=77, max_length_t5=128, max_length_llama=128, **kwargs):
        # print("DEBUG: HiDreamSamplerAdvanced.generate() called") # Less verbose
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            # print(f"HiDream Adv: Initial VRAM: {initial_mem:.2f} MB") # Less verbose

        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Adv Error: No models available or error state selected.")
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

        pipe = None; config = None
        # Cache key includes uncensored state
        cache_key = f"HiDreamSamplerAdvanced_{model_type}_{'uncensored' if use_uncensored_llm else 'standard'}"

        # --- Model Loading / Caching (using shared cache and cleanup) ---
        if cache_key in self._model_cache:
            # print(f"Checking Adv cache for {cache_key}...") # Less verbose
            pipe, config = self._model_cache[cache_key]
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None or not hasattr(pipe, 'text_encoder_4') or pipe.text_encoder_4 is None:
                print(f"Invalid Adv cache entry for {cache_key}, removing and reloading...")
                del self._model_cache[cache_key]
                pipe, config = None, None
            else:
                print(f"Using cached model: {cache_key}")

        if pipe is None:
            # Clean up *other* models before loading a new one
            self.cleanup_cache(exclude_key=None) # Clean all before loading

            print(f"Loading model for {cache_key}...")
            try:
                pipe, config = load_models(model_type, use_uncensored_llm)
                self._model_cache[cache_key] = (pipe, config)
                print(f"Model {cache_key} loaded and cached successfully!")
            except Exception as e:
                print(f"!!! ERROR loading model '{model_type}' for Advanced Node: {e}")
                import traceback
                traceback.print_exc()
                return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

        if pipe is None or config is None:
            print(f"CRITICAL ERROR: Model pipe or config is None after loading/cache check for {cache_key}.")
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

        # --- Scheduler Selection ---
        original_scheduler_class_name = config["scheduler_class"]
        original_shift = config["shift"]
        target_scheduler_name = scheduler # The user's choice
        scheduler_kwargs = {"num_train_timesteps": 1000, "shift": original_shift, "use_dynamic_shifting": False}
        new_scheduler_instance = None

        if target_scheduler_name == "Default for model":
            # Ensure the *correct* default scheduler instance is used
            print(f"Using model's default scheduler: {original_scheduler_class_name}")
            target_scheduler_class = available_schedulers.get(original_scheduler_class_name)
            if target_scheduler_class:
                 new_scheduler_instance = target_scheduler_class(**scheduler_kwargs)
            else:
                 print(f"Warning: Default scheduler {original_scheduler_class_name} not found in available schedulers!")
                 # Fallback? Or error? Let's try keeping the one already in pipe.
                 new_scheduler_instance = pipe.scheduler # Hope it's the right one
        else:
            # Handle specific selections, including variants
            selected_class_name = target_scheduler_name.split(" (")[0] # Get base name e.g., "FlashFlowMatchEulerDiscreteScheduler"
            target_scheduler_class = available_schedulers.get(selected_class_name)

            if target_scheduler_class:
                print(f"Using selected scheduler: {target_scheduler_name}")
                # Add flags for variants
                if "(Karras)" in target_scheduler_name:
                    scheduler_kwargs["use_karras_sigmas"] = True
                    print("  - Applying Karras sigmas")
                elif "(Exponential)" in target_scheduler_name:
                    scheduler_kwargs["use_exponential_sigmas"] = True
                    print("  - Applying Exponential sigmas")

                # Only include relevant kwargs for the specific class
                # (This requires knowing scheduler signatures or more robust handling)
                # For now, assume base kwargs are okay, but filter if needed.
                try:
                    new_scheduler_instance = target_scheduler_class(**scheduler_kwargs)
                except TypeError as te:
                     print(f"Warning: TypeError initializing {selected_class_name}: {te}. Trying without variant flags.")
                     # Try removing variant flags if they caused the error
                     scheduler_kwargs.pop("use_karras_sigmas", None)
                     scheduler_kwargs.pop("use_exponential_sigmas", None)
                     try:
                         new_scheduler_instance = target_scheduler_class(**scheduler_kwargs)
                     except Exception as e_fallback:
                          print(f"Error creating scheduler {selected_class_name} even after fallback: {e_fallback}")
                          new_scheduler_instance = pipe.scheduler # Fallback to existing
            else:
                print(f"Warning: Selected scheduler class {selected_class_name} not found! Using existing scheduler.")
                new_scheduler_instance = pipe.scheduler # Keep the one already in the pipe

        # Assign the chosen or fallback scheduler
        if new_scheduler_instance and pipe.scheduler != new_scheduler_instance:
             print(f"Assigning scheduler instance: {type(new_scheduler_instance).__name__}")
             pipe.scheduler = new_scheduler_instance
        elif not new_scheduler_instance:
             print("Error: Could not determine scheduler instance. Execution may fail.")


        # --- Generation Setup ---
        num_inference_steps = override_steps if override_steps >= 1 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        pbar = comfy.utils.ProgressBar(num_inference_steps)

        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using inference device: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)

        print(f"\n--- Starting Adv Generation ---")
        print(f"  Model: {model_type}{' (uncensored)' if use_uncensored_llm else ''}, Res: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        print(f"  Scheduler: {type(pipe.scheduler).__name__} (Shift: {pipe.scheduler.config.shift}, Karras: {getattr(pipe.scheduler.config, 'use_karras_sigmas', 'N/A')}, Exp: {getattr(pipe.scheduler.config, 'use_exponential_sigmas', 'N/A')})")

        # --- Prepare Prompts ---
        # Use specific prompts if provided, otherwise fallback to primary prompt
        prompt_clip_l_final = clip_l_prompt.strip() if clip_l_prompt.strip() else primary_prompt
        prompt_openclip_final = openclip_prompt.strip() if openclip_prompt.strip() else primary_prompt
        prompt_t5_final = t5_prompt.strip() if t5_prompt.strip() else primary_prompt
        prompt_llama_final = llama_prompt.strip() if llama_prompt.strip() else primary_prompt
        negative_prompt_final = negative_prompt.strip() if negative_prompt else None

        print(f"  Prompts (Max Lens):")
        print(f"    CLIP-L ({max_length_clip_l}): {prompt_clip_l_final[:80]}{'...' if len(prompt_clip_l_final) > 80 else ''}")
        print(f"    OpenCLIP ({max_length_openclip}): {prompt_openclip_final[:80]}{'...' if len(prompt_openclip_final) > 80 else ''}")
        print(f"    T5 ({max_length_t5}): {prompt_t5_final[:80]}{'...' if len(prompt_t5_final) > 80 else ''}")
        print(f"    Llama ({max_length_llama}): {prompt_llama_final[:80]}{'...' if len(prompt_llama_final) > 80 else ''}")
        if negative_prompt_final:
             print(f"    Negative: {negative_prompt_final[:80]}{'...' if len(negative_prompt_final) > 80 else ''}")
        else:
             print("    Negative: (Not provided)")


        # --- Run Inference ---
        output_images = None
        try:
             # Ensure model components are on the correct device just before inference
            target_device = comfy.model_management.get_torch_device()
            # Simplified checks - assume pipe components are moved correctly during load or by offloader
            # pipe.to(target_device) # Re-moving the whole pipe might be needed if offloading isn't perfect

            print("Executing pipeline inference (Advanced)...")
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt_clip_l_final,           # Corresponds to prompt/tokenizer
                    prompt_2=prompt_openclip_final,       # Corresponds to prompt_2/tokenizer_2 (if exists)
                    prompt_3=prompt_t5_final,             # Corresponds to prompt_3/tokenizer_3 (if exists)
                    prompt_4=prompt_llama_final,          # Corresponds to prompt_4/tokenizer_4 (Llama in this case)
                    negative_prompt=negative_prompt_final,
                    negative_prompt_2=negative_prompt_final, # Often need negative for multiple encoders
                    negative_prompt_3=negative_prompt_final,
                    negative_prompt_4=negative_prompt_final,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    # Pass the max sequence lengths
                    max_sequence_length_clip_l=max_length_clip_l,
                    max_sequence_length_openclip=max_length_openclip,
                    max_sequence_length_t5=max_length_t5,
                    max_sequence_length_llama=max_length_llama,
                    # Add callback for progress bar
                    callback_on_step_end=lambda *args: pbar.update(1) or {}
                )
                output_images = output.images
            print("Pipeline inference finished.")

        except Exception as e:
            print(f"!!! ERROR during pipeline execution (Advanced): {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)
        finally:
            pbar.update_absolute(num_inference_steps, num_inference_steps) # Ensure pbar completes

        print("--- Generation Complete ---")

        # --- Process Output ---
        if output_images is None or len(output_images) == 0 or output_images[0] is None:
            print("ERROR: No valid image returned from the pipeline. Returning blank image.")
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

        try:
            output_tensor = pil2tensor(output_images[0]) # Use shared helper

            if output_tensor is None:
                print("ERROR: pil2tensor failed to convert the output image. Returning blank image.")
                return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

            if output_tensor.dtype != torch.float32:
                print(f"Converting output tensor from {output_tensor.dtype} to float32.")
                output_tensor = output_tensor.to(torch.float32)

            if len(output_tensor.shape) != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[3] != 3:
                print(f"ERROR: Invalid tensor shape {output_tensor.shape} after conversion. Expected [1, H, W, 3]. Returning blank image.")
                return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

            print(f"Successfully processed output image to tensor shape: {output_tensor.shape}")

            # Log final memory usage for advanced node
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"HiDream Adv: Final VRAM: {final_mem:.2f} MB (Change: {final_mem-initial_mem:.2f} MB)")

            # comfy.model_management.soft_empty_cache()

            return (output_tensor,)
        except Exception as e:
            print(f"!!! Error processing output image (Advanced): {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler,
    "HiDreamSamplerAdvanced": HiDreamSamplerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler 🦒", # Added emoji for fun
    "HiDreamSamplerAdvanced": "HiDream Sampler Advanced 🦒"
}

# Final status message
print("-" * 60)
print("HiDream Sampler Node Initialized")
available_models_list = list(MODEL_CONFIGS.keys())
if not available_models_list:
     print("WARNING: No HiDream models are available based on installed dependencies!")
     print("Please check installation of: Optimum, Accelerate, BitsAndBytes")
else:
     print(f"Available Models: {available_models_list}")
print(f"Requires Optimum for GPTQ (-nf4 models): {'Yes' if optimum_available else 'No (Optimum not found!)'}")
print(f"Requires BitsAndBytes for BNB (standard models): {'Yes' if bnb_available else 'No (BitsAndBytes not found!)'}")
print("-" * 60)
