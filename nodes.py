"""
ComfyUI-MOSS-Layout
Qwen3-VL / Qwen3.5-VL  ->  MOSS-Image layout prompt (JSON, abs_centralized, yxyx)

Backends
  1. CLIP connected  : uses ComfyUI's own text-generation path (clip.generate).
                       Load Qwen3-VL / Qwen3.5-VL with a text-encoder loader node.
  2. No CLIP         : loads a Hugging Face / local Qwen VL model via transformers.
"""
import gc
import inspect
import json
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import comfy.model_management as mm
except Exception:  # allows importing outside ComfyUI for testing
    mm = None

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img[0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3])


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _round16(v: int) -> int:
    return max(16, int(round(v / 16.0)) * 16)


def _extract_json(text: str):
    """Pull the first parseable JSON object/array out of a model response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


_TEXT_WORDS = re.compile(
    r"\b(text|texts|title|titles|word|words|wording|letter|letters|lettering|writing|written|"
    r"typography|font|slogan|headline|subtitle|inscription|handwritten|says|reads)\b|文字|文本|标题",
    re.I,
)
_QUOTED = re.compile(
    r'(?:\s*\b(?:reading|reads|saying|says|labeled|labelled|titled|that say|that reads|'
    r'with the (?:text|words|word|letters)|the (?:text|words|word))\s*)?'
    r'\s*["“”«»„][^"“”«»„]*["“”«»„]', re.I)
_DANGLING = re.compile(
    r"\b(?:reading|reads|saying|says|labeled|labelled|titled|that say|that reads|"
    r"with the (?:text|words|word|letters))\s*(?=[,.;:)]|$)", re.I)


def _wants_text(*parts):
    """Text is only in scope when the user's own targets/changes ask for it."""
    joined = "\n".join(p for p in parts if p)
    return bool(_TEXT_WORDS.search(joined) or re.search(r'["“”«»„]', joined))


def _strip_text_mentions(s):
    s = _QUOTED.sub("", s)
    s = _DANGLING.sub("", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r",\s*(?=[.;])", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _targets_block(targets, smart_mode, max_items):
    extra = "\n".join(f"- {t}" for t in targets)
    if not smart_mode:
        return "Find ALL instances of these targets in the image:\n" + (extra or "- (none specified)")
    block = f"""SMART MODE: decide for yourself what to detect. Study the whole image and list every visible element that helps recreate its layout as faithfully as possible, up to {max_items} items:
- Each person or animal: one item for the whole subject, then separate items for their visible (or clearly implied but hidden) parts: head/face, hair, torso, each arm, each hand, each leg, each foot, and notable clothing and accessories (top, jacket, dress, trousers, shoes, hat, glasses, jewelry, bag).
- Every distinct object, piece of furniture, vehicle, food item, plant, prop and handheld item.
- The main background and scene regions (sky, ground or floor, walls, water, buildings, mountains, foliage, windows, doors).
- Together the items should account for most of the canvas so nothing important is left unlisted. Skip tiny or insignificant details.
- If you must cut items to fit the limit, keep the larger and more important ones."""
    if extra:
        block += "\nAlso make sure these specific targets are included:\n" + extra
    return block


def _build_qwen_prompt(targets, W, H, iw, ih, coord_mode, changes, meta,
                       smart_mode=False, max_items=40, text_ok=False):
    if coord_mode == "normalized_0_1000":
        coord_rule = (
            "bbox_2d must be [x1, y1, x2, y2] in normalized 0-1000 coordinates relative to the "
            "full image (origin at top-left, x to the right, y downward)."
        )
    else:
        coord_rule = (
            f"bbox_2d must be [x1, y1, x2, y2] in absolute pixels of the input image "
            f"(which is {iw}x{ih}), origin at top-left."
        )

    # Text handling: the schema, rules and change instructions only mention text when the
    # user asked for it, so the model is never primed with the idea.
    if text_ok:
        text_rule = (
            'Use label "text" ONLY for targets or requested changes that are explicitly about text. '
            "Ignore every other piece of writing in the image."
        )
        label_line = '"label": "object", or "text" for a text element,'
        text_field = '      "text": "exact visible text, only for label text, copied character for character",\n'
        text_lang_rule = '- Keep "text" in its original language, unmodified.\n'
        text_change_line = '- If a change edits text, put the NEW text in "text".\n'
    else:
        text_rule = (
            "Writing on signs, labels, logos, clothing, packaging, screens, posters and watermarks is not "
            "part of the layout: create no items for it, do not read it, and never quote or mention "
            "any written words in any caption. Describe such surfaces only by shape, color and material."
        )
        label_line = '"label": "object",'
        text_field = ""
        text_lang_rule = ""
        text_change_line = ""

    meta_tags = ", ".join(m.strip() for m in meta.split(",") if m.strip())
    if meta_tags:
        style_rule = (
            f'- Style is fixed by the meta tags: "{meta_tags}". Never name, describe or infer an art style or '
            "medium (photo, photorealistic, anime, painting, 3D render, illustration, etc.) in ANY caption, "
            "even if the reference image has one. Describe only content, composition, colors and lighting."
        )
    else:
        style_rule = (
            "- Never name, describe or infer an art style or medium (photo, anime, painting, 3D render, etc.) "
            "in ANY caption. Describe only content, composition, colors and lighting."
        )

    if changes.strip():
        change_block = f"""
REQUESTED CHANGES (the final image must reflect these):
{changes.strip()}

How to apply the changes:
- Boxes still follow where things are in the reference image (unless a change says to move, resize or add something).
- Every caption (and the overall caption) must describe the element AS IT WILL LOOK AFTER the change. Write the new color, outfit, object or person in full visual detail, and do NOT mention the old version or say "changed to".
- If a change replaces a person or object, keep its box and fully describe the replacement (appearance, clothing, pose, colors).
- If a change adds something not in the image, add a new item with a sensible bbox for where it belongs.
- If a change removes something, leave that item out entirely.
{text_change_line}- If a change is global (lighting, season, time of day), reflect it in the overall caption and in every affected item caption.
- Elements not touched by the changes are described as they appear.
"""
    else:
        change_block = ""

    smart_rule = ("- Order items: whole subjects first, then their parts and clothing, then objects, then background regions.\n"
                  if smart_mode else "- Order items from most to least prominent.\n")

    return f"""You are a precise object detector and layout planner.
The final composition is a {W}x{H} canvas (width {W}, height {H}); the input image is the layout reference.
{change_block}
{_targets_block(targets, smart_mode, max_items)}

{text_rule}

Return exactly one JSON object and nothing else:
{{
  "caption": "one or two sentences describing the overall scene, composition, colors and lighting (no art style or medium)",
  "items": [
    {{
      {label_line}
      "caption": "detailed description of this element: 2-4 sentences covering appearance, colors, materials, clothing, pose or orientation, and its spatial relationships (what it is in front of, behind, holding, inside or touching), including which parts are hidden or cut off",
{text_field}      "bbox_2d": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- {coord_rule}
- Boxes cover the element's FULL estimated extent, not just the visible part. If part of it is hidden behind or inside something else (a hand or foot behind a person, an object partly inside a container, a person cropped by another object), extend the box to where the whole element would be, staying inside the image bounds. Overlapping or nested boxes are fine. Fully visible elements get a tight box.
- Captions must be specific and detailed. Always state spatial relationships and occlusion explicitly, for example "her left hand is hidden behind the man's shoulder" or "only the top half of the vase is visible above the counter". For people include clothing, hair, expression, pose, facing direction and where hands and feet are. For objects include material, color, shape, size and state.
- Do not invent elements that are not visible.
- One item per instance (two people = two items).
- Write captions in English.
{text_lang_rule}{style_rule}
{smart_rule}- Output JSON only, no Markdown, no explanation."""


def _to_moss_bbox(bb, W, H, iw, ih, coord_mode, pad_pct=0.0):
    """Qwen [x1,y1,x2,y2] -> MOSS [ymin,xmin,ymax,xmax] absolute, centred on canvas."""
    try:
        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
    except Exception:
        return None
    if coord_mode == "normalized_0_1000":
        x1, x2 = x1 / 1000.0 * W, x2 / 1000.0 * W
        y1, y2 = y1 / 1000.0 * H, y2 / 1000.0 * H
    else:
        x1, x2 = x1 / iw * W, x2 / iw * W
        y1, y2 = y1 / ih * H, y2 / ih * H
    x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
    y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
    if pad_pct > 0:
        px, py = (x2 - x1) * pad_pct / 100.0, (y2 - y1) * pad_pct / 100.0
        x1, x2 = max(x1 - px, 0), min(x2 + px, W)
        y1, y2 = max(y1 - py, 0), min(y2 + py, H)
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return [int(round(y1 - H / 2)), int(round(x1 - W / 2)),
            int(round(y2 - H / 2)), int(round(x2 - W / 2))]


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

def _generate_with_clip(clip, pil, prompt, max_new_tokens, temperature, seed):
    if not hasattr(clip, "generate") or not hasattr(clip, "tokenize"):
        raise RuntimeError(
            "This CLIP object has no .generate(). Update ComfyUI (text generation support is "
            "recent) and load Qwen3-VL/Qwen3.5-VL through a text-encoder loader, or disconnect "
            "the clip input and use the hf_model path instead."
        )
    img_t = _pil_to_tensor(pil)

    tok_kwargs = {"image": img_t}
    sig = inspect.signature(clip.tokenize).parameters
    if "skip_template" in sig:
        tok_kwargs["skip_template"] = False
    if "min_length" in sig:
        tok_kwargs["min_length"] = 1
    tokens = clip.tokenize(prompt, **tok_kwargs)

    gen_kwargs = dict(
        do_sample=temperature > 0, max_length=max_new_tokens, temperature=max(temperature, 1e-4),
        top_k=64, top_p=0.95, min_p=0.0, repetition_penalty=1.0, seed=seed,
    )
    gsig = inspect.signature(clip.generate).parameters
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in gsig}
    ids = clip.generate(tokens, **gen_kwargs)
    return clip.decode(ids, skip_special_tokens=True)


_HF_CACHE = {}


def _load_hf(model_id):
    if model_id in _HF_CACHE:
        return _HF_CACHE[model_id]
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    _HF_CACHE[model_id] = (model, processor)
    return model, processor


def _generate_with_hf(model_id, pil, prompt, max_new_tokens, temperature, seed, keep_loaded):
    model, processor = _load_hf(model_id)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-4), top_p=0.95,
        )
    text = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    if not keep_loaded:
        _HF_CACHE.pop(model_id, None)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if mm is not None:
            mm.soft_empty_cache()
    return text


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class MOSSLayoutFromQwenVL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                                  "tooltip": "Target MOSS canvas width. 0 = use the image's width (rounded to 16)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                                   "tooltip": "Target MOSS canvas height. 0 = use the image's height (rounded to 16)."}),
                "targets": ("STRING", {"multiline": True, "default": "person",
                                       "tooltip": "One thing to detect per line. Ignored/optional extras when smart_mode is on. For text, say so explicitly, e.g. 'title text EXPLORE'. Text is only handled when you ask for it."}),
                "meta": ("STRING", {"default": "photo",
                                    "tooltip": "Comma-separated style/meta tags for summary.meta, e.g. photo, poster. Qwen is told NOT to describe style in captions, so style comes only from here."}),
                "coord_mode": (["normalized_0_1000", "absolute_pixels"],
                               {"default": "normalized_0_1000",
                                "tooltip": "Qwen3-VL/3.5 use normalized 0-1000. Use absolute_pixels for Qwen2.5-VL style models."}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Qwen3-VL / Qwen3.5-VL loaded as a text encoder. If unconnected, hf_model is used."}),
                "hf_model": ("STRING", {"default": "Qwen/Qwen3-VL-8B-Instruct",
                                        "tooltip": "HF repo id or local folder. Used only when clip is not connected."}),
                "keep_loaded": ("BOOLEAN", {"default": False,
                                            "tooltip": "HF path only: keep Qwen in memory after running."}),
                "change_request": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "Describe edits to make, one per line. E.g. 'make the dress red', 'replace the man with an elderly woman in a grey coat', 'add a dog on the left', 'change the sign text to OPEN'. Leave empty to just describe the image."}),
                "box_padding": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5,
                                          "tooltip": "Grow every box by this % of its size on each side (clamped to the canvas). Extra safety on top of Qwen estimating hidden/occluded parts."}),
                "smart_mode": ("BOOLEAN", {"default": False,
                                           "tooltip": "Qwen decides what to detect by itself: people and their body parts, clothing, objects and background regions, to match the reference layout as closely as possible. Anything in targets is still included."}),
                "max_items": ("INT", {"default": 40, "min": 5, "max": 120, "step": 1,
                                      "tooltip": "Smart mode only: maximum number of regions Qwen may return."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "preview", "raw_qwen_output", "width", "height")
    FUNCTION = "run"
    CATEGORY = "MOSS-Image"

    def run(self, image, width, height, targets, meta, coord_mode,
            max_new_tokens, temperature, seed, clip=None, hf_model="", keep_loaded=False,
            change_request="", box_padding=0.0, smart_mode=False, max_items=40):
        pil = _tensor_to_pil(image)
        iw, ih = pil.size
        W = _round16(width if width > 0 else iw)
        H = _round16(height if height > 0 else ih)

        target_list = [t.strip() for t in targets.splitlines() if t.strip()]
        text_ok = _wants_text(targets, change_request)
        prompt = _build_qwen_prompt(target_list, W, H, iw, ih, coord_mode,
                                    change_request, meta, smart_mode, max_items, text_ok)

        if clip is not None:
            raw = _generate_with_clip(clip, pil, prompt, max_new_tokens, temperature, seed)
        else:
            if not hf_model.strip():
                raise ValueError("Connect a CLIP (Qwen VL) or provide hf_model.")
            raw = _generate_with_hf(hf_model.strip(), pil, prompt, max_new_tokens,
                                    temperature, seed, keep_loaded)

        data = _extract_json(raw)
        if isinstance(data, list):
            data = {"caption": "", "items": data}
        if not isinstance(data, dict):
            raise RuntimeError("Could not parse JSON from Qwen output. See raw_qwen_output:\n" + raw[:1500])

        items_out = []
        for it in data.get("items", []) or []:
            if not isinstance(it, dict):
                continue
            bb = it.get("bbox_2d") or it.get("bbox")
            moss_bb = _to_moss_bbox(bb, W, H, iw, ih, coord_mode, box_padding) if bb else None
            if moss_bb is None:
                continue
            label = "text" if str(it.get("label", "object")).lower() == "text" else "object"
            if label == "text" and not text_ok:
                continue  # text was never requested: drop it even if the model added it
            cap = str(it.get("caption", "")).strip()
            if not text_ok:
                cap = _strip_text_mentions(cap)
            entry = {"label": label, "caption": cap}
            if label == "text":
                entry["text"] = str(it.get("text", ""))
            entry["bbox"] = moss_bb
            items_out.append(entry)

        layout = {
            "summary": {
                "meta": [m.strip() for m in meta.split(",") if m.strip()],
                "canvas": "single",
                "caption": (_strip_text_mentions(str(data.get("caption", "")))
                            if not text_ok else str(data.get("caption", "")).strip()),
            },
            "items": items_out,
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
        out_prompt = json.dumps(layout, ensure_ascii=False, indent=2)

        # Preview: draw the converted boxes back onto the canvas to sanity-check them.
        prev = pil.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(prev)
        for it in items_out:
            ymin, xmin, ymax, xmax = it["bbox"]
            box = [xmin + W / 2, ymin + H / 2, xmax + W / 2, ymax + H / 2]
            color = (255, 80, 80) if it["label"] == "text" else (80, 200, 255)
            d.rectangle(box, outline=color, width=max(2, W // 400))
            d.text((box[0] + 4, box[1] + 2), it.get("text") or it["caption"][:24], fill=color)

        return (out_prompt, _pil_to_tensor(prev), raw, W, H)


NODE_CLASS_MAPPINGS = {"MOSSLayoutFromQwenVL": MOSSLayoutFromQwenVL}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSLayoutFromQwenVL": "MOSS Layout from Qwen-VL"}
