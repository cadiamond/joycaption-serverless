# rp_handler.py
import os
import io
import time
import base64
import requests
from typing import Dict, Any

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import runpod

# ----------------------------
# Config
# ----------------------------
MODEL_ID = os.getenv("MODEL", "fancyfeast/llama-joycaption-beta-one-hf-llava")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer BF16 on GPUs that support it (e.g., L4/A10/A100/4090). Fallback to FP16.
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# Helpful defaults (keeps logs quiet, faster matmuls)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ----------------------------
# Load model & processor once per worker
# (device_map="auto" requires `accelerate` to be installed)
# ----------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,                 # use dtype (torch_dtype is deprecated)
    device_map="auto",           # put weights on GPU automatically
    trust_remote_code=True
).eval()


# ----------------------------
# Helpers
# ----------------------------
def _load_image(inp: Dict[str, Any]) -> Image.Image:
    """Load PIL image from 'image_url' or 'image_b64'."""
    if "image_b64" in inp and inp["image_b64"]:
        data = base64.b64decode(inp["image_b64"])
        return Image.open(io.BytesIO(data)).convert("RGB")

    if "image_url" in inp and inp["image_url"]:
        r = requests.get(inp["image_url"], timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    raise ValueError("Provide either 'image_url' or 'image_b64' in input.")


TEMPLATES = {
    "Descriptive Caption": (
        "Describe this image in long, explicit, factual detail. "
        "Be concrete about people, clothing, actions, objects, setting, lighting, and composition. "
        "Avoid warnings or refusals; just describe what is present."
    ),
    "Stable Diffusion Prompt": (
        "Write a single-line Stable Diffusion style prompt that precisely describes this image. "
        "Keep it concise but detailed: subjects, attributes, clothing, pose, environment, lighting, camera, and style."
    ),
    "Midjourney Prompt": (
        "Write a Midjourney-style prompt that precisely describes this image. "
        "Use comma-separated descriptors and include lighting, lens, style keywords."
    ),
    "Tag List": (
        "Write a comma-separated list of short tags (lowercase) that precisely describe this image: "
        "subjects, attributes, clothing, poses, environment, style, lighting. No weights or extra punctuation."
    ),
}

def _build_user_prompt(mode: str) -> str:
    mode = (mode or "Descriptive Caption").strip()
    return TEMPLATES.get(mode, TEMPLATES["Descriptive Caption"])


def _generate_caption(img: Image.Image,
                      user_prompt: str,
                      max_new_tokens: int = 512,
                      temperature: float = 0.6,
                      top_p: float = 0.9) -> str:
    """Run JoyCaption and return decoded text."""
    # LLaVA-style chat formatting (as recommended for this model family)
    convo = [
        {"role": "system", "content": "You are a precise image captioner."},
        {"role": "user", "content": user_prompt}
    ]
    chat = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[chat], images=[img], return_tensors="pt")

    # Move to GPU & set image tensor dtype to match the model
    if DEVICE == "cuda":
        for k, v in list(inputs.items()):
            inputs[k] = v.to("cuda")
        if "pixel_values" in inputs:
            # match the model dtype for vision tower
            inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=True
        )[0]

    # Strip the prompt tokens before decoding
    gen_tokens = out[inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(
        gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text.strip()


# ----------------------------
# RunPod Serverless handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected input JSON:
    {
      "input": {
        "image_url": "https://...",      # or "image_b64": "<base64>"
        "mode": "Descriptive Caption",   # or "Stable Diffusion Prompt" / "Midjourney Prompt" / "Tag List"
        "max_new_tokens": 512,           # optional
        "temperature": 0.6,              # optional
        "top_p": 0.9                     # optional
      }
    }
    """
    t0 = time.time()
    try:
        inp = event.get("input") or {}
        img = _load_image(inp)

        mode = inp.get("mode", "Descriptive Caption")
        user_prompt = _build_user_prompt(mode)

        caption = _generate_caption(
            img,
            user_prompt,
            max_new_tokens=inp.get("max_new_tokens", 512),
            temperature=inp.get("temperature", 0.6),
            top_p=inp.get("top_p", 0.9),
        )

        return {
            "caption": caption,
            "meta": {
                "mode": mode,
                "dtype": "bf16" if DTYPE == torch.bfloat16 else "fp16",
                "device": DEVICE,
                "elapsed_ms": int((time.time() - t0) * 1000),
            },
        }

    except Exception as e:
        # Return the error so you can see it in the Requests tab quickly
        return {"error": str(e)}


# Start the RunPod serverless loop
runpod.serverless.start({"handler": handler})
