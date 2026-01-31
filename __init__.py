import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
import folder_paths


class LoadHuggingFaceModel:
    """Node to load any HuggingFace model."""

    CATEGORY = "transformers"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("hf_model", "hf_tokenizer")

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {
                        "default": "microsoft/Phi-3.5-mini-instruct",
                        "tooltip": "HuggingFace model ID or path"
                    }
                ),
                "model_type": (
                    ["auto", "causal_lm", "base"],
                    {
                        "default": "causal_lm",
                        "tooltip": "Type of model to load"
                    }
                ),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "Device to load model on"
                    }
                ),
                "torch_dtype": (
                    ["auto", "float32", "float16", "bfloat16"],
                    {
                        "default": "auto",
                        "tooltip": "Data type for model weights"
                    }
                ),
            },
            "optional": {
                "trust_remote_code": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow executing remote code from model repo"
                    }
                ),
                "use_local_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use ComfyUI models folder for caching"
                    }
                ),
            }
        }

    def execute(self, model_id, model_type, device, torch_dtype, trust_remote_code=False, use_local_cache=True):
        # Handle torch dtype
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        dtype = dtype_map[torch_dtype]

        # Setup cache directory
        cache_dir = None
        if use_local_cache:
            try:
                hf_folder = folder_paths.get_folder_paths("huggingface")[0]
                cache_dir = os.path.join(hf_folder, model_id.replace("/", "_"))
            except:
                pass

        # Load model based on type
        model_kwargs = {
            "device_map": device,
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code
        }
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        elif model_type == "base":
            model = AutoModel.from_pretrained(model_id, **model_kwargs)
        else:  # auto
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            except:
                model = AutoModel.from_pretrained(model_id, **model_kwargs)

        # Load tokenizer
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        return (model, tokenizer)


class RunHuggingFaceModel:
    """Node to run inference with any HuggingFace model."""

    CATEGORY = "transformers"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_model": ("hf_model",),
                "hf_tokenizer": ("hf_tokenizer",),
                "task": (
                    ["text-generation", "text2text-generation", "conversational"],
                    {
                        "default": "text-generation",
                        "tooltip": "HuggingFace pipeline task"
                    }
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "What is the meaning of life?",
                        "multiline": True,
                        "tooltip": "Input prompt or instruction"
                    }
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 500,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Maximum tokens to generate"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sampling temperature"
                    }
                ),
                "do_sample": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable sampling (vs greedy decoding)"
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "tooltip": "Random seed for reproducibility"
                    }
                ),
            },
            "optional": {
                "system_message": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional system message (for chat models)"
                    }
                ),
                "return_full_text": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Return full text including prompt"
                    }
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling probability"
                    }
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 200,
                        "tooltip": "Top-k sampling parameter"
                    }
                ),
            }
        }

    def execute(self, hf_model, hf_tokenizer, task, prompt, max_new_tokens, temperature, 
                do_sample, seed, system_message="", return_full_text=False, top_p=1.0, top_k=50):
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build pipeline
        pipe = pipeline(task, model=hf_model, tokenizer=hf_tokenizer)

        # Prepare generation arguments
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "return_full_text": return_full_text,
            "top_p": top_p,
            "top_k": top_k,
        }

        # Handle different input formats based on task
        if task in ["text-generation", "conversational"]:
            # Check if model uses chat template
            if hasattr(hf_tokenizer, 'chat_template') and hf_tokenizer.chat_template:
                # Format as messages for chat models
                messages = []
                if system_message and system_message.strip():
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                output = pipe(messages, **generation_args)
            else:
                # Direct text input for non-chat models
                if system_message and system_message.strip():
                    full_prompt = f"{system_message}\n\n{prompt}"
                else:
                    full_prompt = prompt
                output = pipe(full_prompt, **generation_args)
        else:
            # For other tasks, use direct text input
            output = pipe(prompt, **generation_args)

        # Extract text from output
        if isinstance(output, list) and len(output) > 0:
            result = output[0]
            if isinstance(result, dict):
                # For text-generation tasks
                if "generated_text" in result:
                    response = result["generated_text"]
                    # If it's a list of messages, extract the assistant's response
                    if isinstance(response, list):
                        for msg in reversed(response):
                            if msg.get("role") == "assistant":
                                response = msg.get("content", "")
                                break
                        else:
                            response = str(response)
                else:
                    response = str(result)
            else:
                response = str(result)
        else:
            response = str(output)

        return (response,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "LoadHuggingFaceModel": LoadHuggingFaceModel,
    "RunHuggingFaceModel": RunHuggingFaceModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHuggingFaceModel": "Load HuggingFace Model",
    "RunHuggingFaceModel": "Run HuggingFace Model",
}


