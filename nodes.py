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
from PIL import Image, ImageDraw, ImageFont

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


# --------------------------------------------------------------------------
# Meta (style) lock
#   * the meta tags are the single source of style
#   * the overall caption must state them (added if Qwen forgot)
#   * words from OTHER styles are stripped from every caption
# --------------------------------------------------------------------------

# "core": unambiguous style words, stripped from every caption.
_STYLE_CORE = {
    "photo": ["photorealistic", "photo-realistic", "hyperrealistic", "hyper-realistic", "photographic",
              "dslr", "bokeh", "film grain", "35mm"],
    "anime": ["anime", "manga", "cel-shaded", "cel shaded", "cel shading", "chibi"],
    "cartoon": ["cartoon", "cartoonish"],
    "illustration": ["line art", "lineart", "vector art", "flat design"],
    "painting": ["watercolor", "watercolour", "impasto", "gouache", "brushstrokes", "brush strokes"],
    "3d": ["3d render", "3d rendered", "cgi", "octane render", "unreal engine"],
    "pixel": ["pixel art", "pixelated", "8-bit", "16-bit"],
}
# "extra": medium nouns that can also be real objects ("a photo frame"), so only the overall caption is checked.
_STYLE_EXTRA = {
    "photo": ["photograph", "photography", "photo", "realistic"],
    "cartoon": ["comic book"],
    "illustration": ["illustration", "illustrated", "sketch"],
    "painting": ["painting", "oil painting"],
    "3d": ["3d", "render", "rendered"],
    "pixel": ["pixel"],
}


def _meta_tags(meta):
    seen, tags = set(), []
    for t in re.split(r"[,;]", meta or ""):
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            tags.append(t)
    return tags


def _style_patterns(tags):
    """(item_pattern, summary_pattern) matching words of every style NOT named in the meta."""
    meta_lc = " ".join(tags).lower()

    def named(word):
        return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", meta_lc) is not None

    allowed = set()
    for cat in set(_STYLE_CORE) | set(_STYLE_EXTRA):
        if named(cat) or any(named(w) for w in _STYLE_CORE.get(cat, []) + _STYLE_EXTRA.get(cat, [])):
            allowed.add(cat)

    def build(*tables):
        words = sorted({w for tbl in tables for cat, ws in tbl.items() if cat not in allowed
                        for w in ws if not named(w)}, key=len, reverse=True)
        if not words:
            return None
        alt = "|".join(re.escape(w) for w in words)
        return alt

    item_alt, summ_alt = build(_STYLE_CORE), build(_STYLE_CORE, _STYLE_EXTRA)
    return item_alt, summ_alt


def _strip_style_words(text, alt):
    """Remove words of non-meta styles. Returns (clean_text, [removed words])."""
    if not alt or not text:
        return text, []
    word = re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.I)
    found = [m.group(0).lower() for m in word.finditer(text)]
    if not found:
        return text, []
    # "a close up photograph of X" -> "X"
    text = re.sub(rf"(?<![\w-])(?:an?|the)\s+(?:[\w-]+\s+){{0,3}}?(?:{alt})\s+of\s+", "", text, flags=re.I)
    # a style word used as a phrase ("with film grain", "in the style of a cartoon"): remove it with its connector
    text = re.sub(
        rf"(?:,\s*)?(?:\s+(?:in\s+(?:the\s+|an?\s+)?style\s+of|with|and|using|featuring|plus|in)\s+(?:an?\s+|the\s+)?)?"
        rf"(?<![\w-])(?:{alt})(?:[\s-]+(?:style|styled|look|aesthetic))?(?![\w-])"
        rf"(?=\s*(?:[,.;:]|$)|\s+(?:and|with|in|behind|on|of|against|under|over|near)\b)",
        "", text, flags=re.I)
    # a style word used as an adjective ("photorealistic portrait", "anime-style eyes"): just drop it
    text = re.sub(rf"(?<![\w-])(?:{alt})(?:[\s-]+(?:style|styled|look|aesthetic))?(?![\w-])", "", text, flags=re.I)
    text = re.sub(r"(?<![\w-])(?:in\s+(?:the\s+|an?\s+)?style\s+of|in\s+an?|an?|the)\s*(?=[,.;:]|$)", "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;])\s*(?=[,.;])", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text, found


def _ensure_summary_style(summary, tags):
    """Guarantee the overall caption states the locked style. Returns (caption, was_added)."""
    if not tags:
        return summary, False
    low = summary.lower()
    if any(t.lower() in low for t in tags):
        return summary, False
    return (f"Style: {', '.join(tags)}. {summary}").strip(), True


# ---- Color lock (monochrome metas) and per-item style clause ----
_MONO_RE = re.compile(
    r"black[\s-]*(?:and|&|n)[\s-]*white|\bb\s*&\s*w\b|\bbw\b|monochrom|gr[ae]yscale|"
    r"\bno\s+colou?rs?\b|colou?rless|without\s+colou?r|desaturat", re.I)
_TONES = {
    "light": ["yellow", "pink", "cyan", "lavender", "lime", "beige"],
    "dark": ["purple", "violet", "indigo", "navy", "maroon", "burgundy", "crimson", "brown"],
    "mid": ["red", "orange", "green", "blue", "teal", "turquoise", "magenta", "scarlet", "olive"],
}
_TONE_OF = {w: t for t, ws in _TONES.items() for w in ws}
_COLOR_RE = re.compile(
    r"(?<!\w)(?:(dark|deep|light|pale|pastel|bright|vivid|neon)\s+)?("
    + "|".join(sorted(_TONE_OF, key=len, reverse=True)) + r")(?!\w)", re.I)
_HYPE_RE = re.compile(
    r"(?<!\w)(?:(?:brightly|vividly|vibrantly|richly|deeply)\s+colou?red|colou?rful|multi-?colou?red|"
    r"vibrant|vivid|saturated)(?!\w)", re.I)


def _is_mono(tags):
    return bool(_MONO_RE.search(" ".join(tags)))


def _tone_sub(m):
    mod, col = (m.group(1) or "").lower(), m.group(2).lower()
    if mod in ("dark", "deep"):
        return "dark gray"
    if mod:
        return "light gray"
    return {"light": "light gray", "dark": "dark gray", "mid": "mid-gray"}[_TONE_OF[col]]


def _to_mono(text):
    """Swap color words for gray tones so a no-color style is never contradicted. Returns (text, count)."""
    if not text:
        return text, 0
    text, n1 = _COLOR_RE.subn(_tone_sub, text)
    text, n2 = _HYPE_RE.subn("high-contrast", text)
    text = re.sub(r"(gray)[\s-]colou?red", r"\1", text, flags=re.I)
    return text, n1 + n2


def _mentions_style(text, tags):
    low = text.lower()
    if any(t.lower() in low for t in tags):
        return True
    toks = {w for t in tags for w in re.findall(r"[a-z]+", t.lower()) if len(w) >= 6}
    return any(re.search(rf"\b{w}", low) for w in toks)


def _ensure_item_style(cap, tags):
    """Guarantee an item caption carries the locked style. Returns (caption, was_added)."""
    if not tags or _mentions_style(cap, tags):
        return cap, False
    cap = cap.rstrip()
    if cap and cap[-1] not in ".!?":
        cap += "."
    return f"{cap} Rendered as {', '.join(tags)}.".strip(), True


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
                       smart_mode=False, max_items=40, text_ok=False, style_in_items=True):
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

    tag_list = _meta_tags(meta)
    meta_tags = ", ".join(tag_list)
    mono = _is_mono(tag_list)
    color_word, color_one = ("tones", "tone") if mono else ("colors", "color")
    if meta_tags:
        item_style_rule = (
            "  Every item caption must also include a short clause saying how that element is rendered in the locked "
            'style, using the meta words (for example "drawn with fine pencil linework and shading" for a pencil '
            "drawing). Never use words that belong to a different style.\n"
            if style_in_items else
            "  Item captions must not name any style or medium.\n"
        )
        mono_rule = (
            "  The locked style has NO color. Never use a color word (no green, red, orange, purple, blue, teal, brown, "
            "etc.) in any caption, even if the reference image is colorful. Describe tone and value instead: black, "
            "white, light gray, dark gray, shadowed, highlighted, high-contrast.\n"
            if mono else ""
        )
        style_rule = (
            f'- STYLE LOCK: the final image will be generated ONLY in this style: "{meta_tags}". The reference image is '
            "a layout and content reference, nothing more: its own style, medium, rendering and colors do not carry over.\n"
            "  Describe every element as it would look in the locked style, not as it looks in the reference.\n"
            "  The overall caption must open with a phrase that uses these exact style words, then describe the scene.\n"
            f"{item_style_rule}{mono_rule}"
        ).rstrip("\n")
        summary_hint = (f"one or two sentences describing the overall scene, composition, {color_word} and lighting, "
                        "opening with the locked style words")
    else:
        style_rule = (
            "- Never name, describe or infer an art style or medium (photo, anime, painting, 3D render, etc.) "
            "in ANY caption. Describe only content, composition, colors and lighting."
        )
        summary_hint = "one or two sentences describing the overall scene, composition, colors and lighting (no art style or medium)"

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
  "caption": "{summary_hint}",
  "items": [
    {{
      {label_line}
      "caption": "detailed description of this element: 2-4 sentences covering appearance, {color_word}, materials, clothing, pose or orientation, and its spatial relationships (what it is in front of, behind, holding, inside or touching), including which parts are hidden or cut off",
{text_field}      "bbox_2d": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- {coord_rule}
- Boxes cover the element's FULL estimated extent, not just the visible part. If part of it is hidden behind or inside something else (a hand or foot behind a person, an object partly inside a container, a person cropped by another object), extend the box to where the whole element would be, staying inside the image bounds. Overlapping or nested boxes are fine. Fully visible elements get a tight box.
- Captions must be specific and detailed. Always state spatial relationships and occlusion explicitly, for example "her left hand is hidden behind the man's shoulder" or "only the top half of the vase is visible above the counter". For people include clothing, hair, expression, pose, facing direction and where hands and feet are. For objects include material, {color_one}, shape, size and state.
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

# Bright, well-separated colors. Each box gets a black under-stroke, so they read on any image.
_PALETTE = [
    (255, 64, 64), (255, 196, 0), (0, 230, 118), (0, 200, 255), (255, 0, 200),
    (255, 128, 0), (160, 255, 0), (140, 120, 255), (0, 255, 220), (255, 120, 170),
]


def _get_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_preview(base, items, W, H):
    d = ImageDraw.Draw(base)
    short = min(W, H)
    lw = max(3, short // 250)
    font = _get_font(max(16, short // 40))
    for i, it in enumerate(items):
        ymin, xmin, ymax, xmax = it["bbox"]
        x1, y1, x2, y2 = xmin + W / 2, ymin + H / 2, xmax + W / 2, ymax + H / 2
        color = _PALETTE[i % len(_PALETTE)]
        d.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=lw + 3)   # dark under-stroke
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        raw_label = (it.get("text") or it.get("caption") or "").strip()
        if len(raw_label) > 30:
            raw_label = (raw_label[:30].rsplit(" ", 1)[0] or raw_label[:30]) + "…"
        label = f"{i + 1} {raw_label}"
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, th // 5)
        cw, ch = tw + 2 * pad, th + 2 * pad
        cx = min(max(x1, 0), max(W - cw, 0))
        cy = y1 - ch if y1 - ch >= 0 else y1          # above the box, or inside if no room
        d.rectangle([cx, cy, cx + cw, cy + ch], fill=color, outline=(0, 0, 0), width=2)
        d.text((cx + pad - tb[0], cy + pad - tb[1]), label, fill=(0, 0, 0), font=font)


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
                                    "tooltip": "Style lock. Comma-separated tags, e.g. photo, poster. The overall caption is built around them (added if Qwen forgets) and words from other styles are stripped from all captions."}),
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
                "style_in_items": ("BOOLEAN", {"default": True,
                                               "tooltip": "Every item caption must carry the meta style (Qwen is asked to, and 'Rendered as <meta>.' is added if it doesn't). Turn off to keep style only in the overall caption."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "preview", "raw_qwen_output", "width", "height")
    FUNCTION = "run"
    CATEGORY = "MOSS-Image"

    def run(self, image, width, height, targets, meta, coord_mode,
            max_new_tokens, temperature, seed, clip=None, hf_model="", keep_loaded=False,
            change_request="", box_padding=0.0, smart_mode=False, max_items=40, style_in_items=True):
        pil = _tensor_to_pil(image)
        iw, ih = pil.size
        W = _round16(width if width > 0 else iw)
        H = _round16(height if height > 0 else ih)

        target_list = [t.strip() for t in targets.splitlines() if t.strip()]
        text_ok = _wants_text(targets, change_request)
        prompt = _build_qwen_prompt(target_list, W, H, iw, ih, coord_mode,
                                    change_request, meta, smart_mode, max_items, text_ok, style_in_items)

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

        tags = _meta_tags(meta)
        item_alt, summ_alt = _style_patterns(tags)
        removed = []
        mono = _is_mono(tags)
        recolored = 0
        items_styled = 0

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
            cap, rem = _strip_style_words(cap, item_alt)
            removed += rem
            if mono:
                cap, n = _to_mono(cap)
                recolored += n
            if style_in_items and tags:
                cap, was_added = _ensure_item_style(cap, tags)
                items_styled += was_added
            entry = {"label": label, "caption": cap}
            if label == "text":
                entry["text"] = str(it.get("text", ""))
            entry["bbox"] = moss_bb
            items_out.append(entry)

        summary = (_strip_text_mentions(str(data.get("caption", "")))
                   if not text_ok else str(data.get("caption", "")).strip())
        summary, rem = _strip_style_words(summary, summ_alt)
        removed += rem
        if mono:
            summary, n = _to_mono(summary)
            recolored += n
        summary, added = _ensure_summary_style(summary, tags)

        layout = {
            "summary": {
                "meta": tags,
                "canvas": "single",
                "caption": summary,
            },
            "items": items_out,
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
        out_prompt = json.dumps(layout, ensure_ascii=False, indent=2)

        # Preview: draw the converted boxes back onto the canvas to sanity-check them.
        prev = pil.resize((W, H), Image.LANCZOS)
        _draw_preview(prev, items_out, W, H)

        meta_note = f"\n\n[meta] locked: {', '.join(tags) or '(none)'}"
        if removed:
            meta_note += " | removed conflicting style words: " + ", ".join(sorted(set(removed)))
        if recolored:
            meta_note += f" | no-color style: {recolored} color word(s) converted to gray tones"
        if items_styled:
            meta_note += f" | style clause added to {items_styled} item caption(s)"
        if added:
            meta_note += " | overall caption did not state the style, so it was added"
        return (out_prompt, _pil_to_tensor(prev), raw + meta_note, W, H)


NODE_CLASS_MAPPINGS = {"MOSSLayoutFromQwenVL": MOSSLayoutFromQwenVL}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSLayoutFromQwenVL": "MOSS Layout from Qwen-VL"}
