import os, base64, io, requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import runpod

# Hugging Face model id (JoyCaption Beta One)
MODEL = os.getenv("MODEL", "fancyfeast/llama-joycaption-beta-one-hf-llava")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once per worker
processor = AutoProcessor.from_pretrained(MODEL)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map=0 if DEVICE == "cuda" else "auto"
).eval()

def _load_image(input_dict):
    if "image_b64" in input_dict:
        b = base64.b64decode(input_dict["image_b64"])
        return Image.open(io.BytesIO(b)).convert("RGB")
    url = input_dict["image_url"]
    r = requests.get(url, timeout=30); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def _caption(img, prompt, max_new_tokens=512, temperature=0.6, top_p=0.9):
    convo = [
        {"role": "system", "content": "You are a precise image captioner."},
        {"role": "user", "content": prompt}
    ]
    convo_str = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[convo_str], images=[img], return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens),
            do_sample=True, temperature=float(temperature), top_p=float(top_p),
            use_cache=True
        )[0]
    out = out[inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.strip()

def handler(event):
    """
    Input example:
    {
      "input": {
        "image_url": "https://...",      # or "image_b64": "<base64>"
        "mode": "Descriptive Caption",   # or "Stable Diffusion Prompt", "Midjourney Prompt", "Tag List"
        "max_new_tokens": 512, "temperature": 0.6, "top_p": 0.9
      }
    }
    """
    inp = event["input"]
    img = _load_image(inp)
    mode = (inp.get("mode") or "Descriptive Caption").strip()
    user_prompt = f"Write a long, explicit, factual {mode.lower()} for this image."
    text = _caption(
        img, user_prompt,
        max_new_tokens=inp.get("max_new_tokens", 512),
        temperature=inp.get("temperature", 0.6),
        top_p=inp.get("top_p", 0.9)
    )
    return {"caption": text}

runpod.serverless.start({"handler": handler})
