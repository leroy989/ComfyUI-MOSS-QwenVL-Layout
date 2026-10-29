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
import math
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
    """Pull the best JSON object/array out of a model response.
    - ignores thinking blocks (closed, or closed by a lone </think>)
    - prefers the object holding the most items
    - if the answer was cut off mid-JSON, recovers every complete item (result has "_salvaged": True)
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.split("<think>")[0]                      # an unclosed thinking block holds no answer
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    parsed, best, best_n, pos = [], None, -1, 0
    for m in re.finditer(r"[\{\[]", text):
        if m.start() < pos:          # nested inside an object we already parsed
            continue
        try:
            obj, end = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        pos = m.start() + end
        parsed.append(obj)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            n = len(obj["items"])
        elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            n = len(obj)
        else:
            n = -1
        if n > best_n:
            best, best_n = obj, n
    if best_n >= 0:
        return best
    # cut-off answer: the outer object never closed, so recover the complete item objects inside it
    items = [o for o in parsed if isinstance(o, dict)
             and ("bbox_2d" in o or "bbox" in o or ("id" in o and "caption" in o))]
    if items:
        cap = ""
        mc = re.search(r'"caption"\s*:\s*("(?:[^"\\]|\\.)*")', text)
        if mc and 0 <= mc.start() < (text.find('"items"') if '"items"' in text else len(text)):
            try:
                cap = json.loads(mc.group(1))
            except Exception:
                cap = ""
        return {"caption": cap, "items": items, "_salvaged": True}
    return parsed[0] if parsed else None


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
    word = re.compile(rf"(?<![\w-])(?:{alt})(?:-(?:style|styled|inspired|like|esque)|(?![\w-]))", re.I)
    found = [m.group(0).lower() for m in word.finditer(text)]
    if not found:
        return text, []
    # "a close up photograph of X" -> "X"
    text = re.sub(rf"(?<![\w-])(?:an?|the)\s+(?:[\w-]+\s+){{0,3}}?(?:{alt})\s+of\s+", "", text, flags=re.I)
    # a style word used as a phrase ("with film grain", "in the style of a cartoon"): remove it with its connector
    text = re.sub(
        rf"(?:,\s*)?(?:\s+(?:in\s+(?:the\s+|an?\s+)?style\s+of|with|and|using|featuring|plus|in)\s+(?:an?\s+|the\s+)?)?"
        rf"(?<![\w-])(?:{alt})(?:[\s-]+(?:style|styled|look|aesthetic|inspired|like|esque))?(?![\w-])"
        rf"(?=\s*(?:[,.;:]|$)|\s+(?:and|with|in|behind|on|of|against|under|over|near)\b)",
        "", text, flags=re.I)
    # a style word used as an adjective ("photorealistic portrait", "anime-style eyes"): drop it, leaving a marker
    # so the article before it can be fixed ("An anime-style knight" -> "A knight")
    text = re.sub(rf"(?<![\w-])(?:{alt})(?:[\s-]+(?:style|styled|look|aesthetic|inspired|like|esque))?(?![\w-])",
                  "\x00", text, flags=re.I)

    def _article(m):
        art, nxt = m.group(1), m.group(2)
        fixed = ("an" if nxt.lower() in "aeiou" else "a")
        return (fixed.capitalize() if art[0].isupper() else fixed) + " " + nxt

    text = re.sub(r"(?<![\w-])(an?|An?)\s+\x00\s*(\w)", _article, text)
    text = text.replace("\x00", "")
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
                       smart_mode=False, max_items=40, text_ok=False, style_in_items=True, depth="off"):
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

    if depth != "off":
        layer_field = '      "layer": 1,\n'
        layer_rule = ("- layer: an integer depth order for every item. 1 = farthest from the camera (sky, backdrop, "
                      "ground, walls); higher numbers are closer to the camera. Elements that overlap must have "
                      "different layers so it is clear which is in front. In each caption say what the element is "
                      "in front of or behind.\n")
    else:
        layer_field, layer_rule = "", ""

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
{text_field}{layer_field}      "bbox_2d": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- {coord_rule}
- Boxes cover the element's FULL estimated extent, not just the visible part. If part of it is hidden behind or inside something else (a hand or foot behind a person, an object partly inside a container, a person cropped by another object), extend the box to where the whole element would be, staying inside the image bounds. Overlapping or nested boxes are fine. Fully visible elements get a tight box.
- Captions must be specific and detailed. Always state spatial relationships and occlusion explicitly, for example "her left hand is hidden behind the man's shoulder" or "only the top half of the vase is visible above the counter". For people include clothing, hair, expression, pose, facing direction and where hands and feet are. For objects include material, {color_one}, shape, size and state.
{layer_rule}- Do not invent elements that are not visible.
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

def _generate_with_clip(clip, pil, prompt, max_new_tokens, temperature, seed, thinking=False):
    if not hasattr(clip, "generate") or not hasattr(clip, "tokenize"):
        raise RuntimeError(
            "This CLIP object has no .generate(). Update ComfyUI (text generation support is "
            "recent) and load Qwen3-VL/Qwen3.5-VL through a text-encoder loader, or disconnect "
            "the clip input and use the hf_model path instead."
        )
    tok_kwargs = {"image": _pil_to_tensor(pil)} if pil is not None else {}
    sig = inspect.signature(clip.tokenize).parameters
    if "skip_template" in sig:
        tok_kwargs["skip_template"] = False
    if "min_length" in sig:
        tok_kwargs["min_length"] = 1
    if "thinking" in sig:
        tok_kwargs["thinking"] = thinking
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


def _unload_hf(model_id):
    _HF_CACHE.pop(model_id, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if mm is not None:
        mm.soft_empty_cache()


def _generate_with_hf(model_id, pil, prompt, max_new_tokens, temperature, seed, keep_loaded, thinking=False):
    model, processor = _load_hf(model_id)
    content = ([{"type": "image", "image": pil}] if pil is not None else []) + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    chat_kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    try:
        inputs = processor.apply_chat_template(messages, enable_thinking=thinking, **chat_kwargs)
    except TypeError:                      # template/processor without the thinking switch
        inputs = processor.apply_chat_template(messages, **chat_kwargs)
    inputs = inputs.to(model.device)
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


# --------------------------------------------------------------------------
# Canvas sizing (scale to total pixels, aspect ratios)
# --------------------------------------------------------------------------

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
ASPECT_RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21"]
_PIL_METHOD = {"nearest-exact": Image.NEAREST, "bilinear": Image.BILINEAR, "area": Image.BOX,
               "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}

MODE_IMAGE = "From image"
MODE_PROMPT = "From prompt (experimental)"
DEPTH_MODES = ["back to front", "front to back", "off"]


def _scale_image(image, megapixels, steps, method):
    """Same maths as ComfyUI's 'Scale Image to Total Pixels'. Returns (tensor [1,H,W,3], W, H)."""
    img = image[:1, :, :, :3]
    h0, w0 = int(img.shape[1]), int(img.shape[2])
    steps = max(1, int(steps))
    if megapixels and megapixels > 0:
        scale_by = math.sqrt(int(megapixels * 1024 * 1024) / (w0 * h0))
        W = max(steps, int(round(w0 * scale_by / steps)) * steps)
        H = max(steps, int(round(h0 * scale_by / steps)) * steps)
    else:                                     # megapixels 0 = keep the original size (snapped to steps)
        W = max(steps, int(round(w0 / steps)) * steps)
        H = max(steps, int(round(h0 / steps)) * steps)
    if (W, H) == (w0, h0):
        return img, W, H
    try:
        import comfy.utils
        s = comfy.utils.common_upscale(img.movedim(-1, 1), W, H, method, "disabled")
        return s.movedim(1, -1), W, H
    except Exception:
        pil = _tensor_to_pil(img).resize((W, H), _PIL_METHOD.get(method, Image.LANCZOS))
        return _pil_to_tensor(pil), W, H


def _size_from_aspect(aspect, megapixels, steps):
    try:
        aw, ah = [float(v) for v in aspect.split(":")]
    except Exception:
        aw, ah = 1.0, 1.0
    steps = max(1, int(steps))
    total = int((megapixels if megapixels and megapixels > 0 else 1.0) * 1024 * 1024)
    scale = math.sqrt(total / (aw * ah))
    return (max(steps, int(round(aw * scale / steps)) * steps),
            max(steps, int(round(ah * scale / steps)) * steps))


# --------------------------------------------------------------------------
# Prompt mode: build a complete box layout from text only (no image)
# --------------------------------------------------------------------------

def _build_layout_prompt_text(user_prompt, W, H, targets, changes, meta, max_items, text_ok,
                              style_in_items=True, depth="off"):
    tag_list = _meta_tags(meta)
    meta_tags = ", ".join(tag_list)
    mono = _is_mono(tag_list)
    color_word, color_one = ("tones", "tone") if mono else ("colors", "color")

    if text_ok:
        text_rule = ('- Text: when the prompt asks for text (a title, sign, caption, label, words on an object), create '
                     'one element with label "text" per piece of text, copy the string exactly as the prompt gives it, '
                     "place it where such text normally sits, and size the box so the string fits (wide enough for "
                     "its length). Never invent text the prompt did not ask for.\n")
        label_line = '"label": "object", or "text" for a text element,'
        text_field = '      "text": "exact text, only for label text",\n'
    else:
        text_rule = ("- The prompt asks for no text: create no text elements and never put written words, signs with "
                     "writing, logos or captions in any caption.\n")
        label_line = '"label": "object",'
        text_field = ""

    if meta_tags:
        item_clause = (
            "  Every item caption must also include a short clause saying how that element is rendered in the locked "
            "style, using the meta words. Never use words that belong to a different style.\n"
            if style_in_items else "  Item captions must not name any style or medium.\n")
        mono_rule = (
            "  The locked style has NO color. Never use a color word in any caption. Describe tone and value instead: "
            "black, white, light gray, dark gray, shadowed, highlighted, high-contrast.\n" if mono else "")
        style_rule = (
            f'- STYLE LOCK: the final image will be generated ONLY in this style: "{meta_tags}". If the user\'s prompt '
            "asks for a different style or medium, ignore that part and use the locked style.\n"
            "  The overall caption must open with a phrase that uses these exact style words, then describe the scene.\n"
            f"{item_clause}{mono_rule}")
        summary_hint = (f"one or two sentences describing the whole scene, composition, {color_word} and lighting, "
                        "opening with the locked style words")
    else:
        style_rule = ("- The overall caption states the style or medium the user's prompt asks for (if it names one), "
                      "then describes the scene. Item captions never name a style or medium.\n")
        summary_hint = ("one or two sentences describing the whole scene, composition, colors and lighting, including "
                        "any style or medium the prompt asks for")

    if depth != "off":
        layer_field = '      "layer": 1,\n'
        layer_rule = ("- layer: an integer depth order for every item. 1 = farthest from the camera (sky, backdrop, "
                      "ground, walls); higher numbers are closer to the camera. Elements that overlap must have "
                      "different layers so it is clear which is in front. In each caption say what the element is in "
                      "front of or behind.\n")
    else:
        layer_field, layer_rule = "", ""

    extra = (f"\nEXTRA INSTRUCTIONS FROM THE USER (apply them to the plan):\n{changes.strip()}\n"
             if changes.strip() else "")
    must = ("\nMake sure these elements are included:\n" + "\n".join(f"- {t}" for t in targets) + "\n"
            if targets else "")

    ex_items = [
        ("A wide sky filling the upper half, glowing near the horizon.", [0, 0, 1000, 540], 1),
        ("A grassy hill filling the lower half, behind and below both figures.", [0, 500, 1000, 1000], 2),
        ("A horse standing at the right, side view facing left, in front of the hill.", [520, 330, 940, 890], 3),
        ("A knight in plate armor standing at the left facing right, helmet under his arm, in front of the hill.",
         [170, 250, 450, 890], 3),
    ]
    example = {"caption": "A knight beside his horse on a hill at sunrise.", "items": [
        {"label": "object", "caption": c, **({"layer": l} if depth != "off" else {}), "bbox_2d": b}
        for c, b, l in ex_items]}
    ex_json = json.dumps(example, ensure_ascii=False)
    shape = f"{H / W:.2f}"
    ar = f"{W / H:.2f}:1"

    return f"""You are an expert art director and layout planner for a text-to-image model that is guided by bounding boxes.
Turn the user's prompt into a complete, well-composed layout for a {W}x{H} canvas (width {W}, height {H}, aspect ratio {ar}).
{extra}
USER'S PROMPT:
-----
{user_prompt.strip()}
-----
{must}
How to plan:
1. List every subject, character, object and background region the prompt mentions, and add the supporting elements the scene naturally implies (ground, sky, walls, furniture, props). Use as many elements as the scene needs, at most {max_items}; a simple prompt needs only a handful. Do not add unrelated things.
2. Compose the canvas well: the main subject is prominent and not cut off, space is balanced, scale relationships are sensible (a person is much smaller than a house, a hand is smaller than its arm), parts sit inside their wholes (a head at the top of its body box, a hat on the head), and background regions cover the area behind everything.
3. Give every element a bounding box [x1, y1, x2, y2] in normalized 0-1000 coordinates: origin at the top-left, x to the right, y downward, 0-1000 on both axes whatever the canvas size. x comes FIRST. Keep x1 < x2, y1 < y2, and everything inside 0-1000.
4. Mind the canvas shape: one unit of x is {W / 1000:.2f} px and one unit of y is {H / 1000:.2f} px. A box with equal x and y extents is not square unless the canvas is square; something that should look square needs an x extent of about {shape} times its y extent.
5. Boxes overlap only where things really overlap or contain each other (a person in front of a background, a hat on a head).

Return exactly one JSON object and nothing else:
{{
  "caption": "{summary_hint}",
  "items": [
    {{
      {label_line}
      "caption": "detailed description of this element: 2-4 sentences covering appearance, {color_word}, materials, clothing, pose or orientation, expression, and its spatial relationships (what it is in front of, behind, holding, inside or touching)",
{text_field}{layer_field}      "bbox_2d": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- Captions are specific and detailed: for people include clothing, hair, expression, pose, facing direction and where hands and feet are; for objects include material, {color_one}, shape, size and state.
{layer_rule}{text_rule}- Write captions in English.
{style_rule}- Output JSON only, no Markdown, no explanation.

Format example only (it shows how boxes relate to a scene on a SQUARE canvas; never reuse its content), for the prompt "a knight standing beside his horse on a hill at sunrise":
{ex_json}"""


def _blank_canvas(W, H):
    """Neutral canvas with a rule-of-thirds grid, used as the preview background in prompt mode."""
    base = Image.new("RGB", (W, H), (38, 40, 48))
    d = ImageDraw.Draw(base)
    for k in (1, 2):
        d.line([(W * k / 3, 0), (W * k / 3, H)], fill=(70, 74, 86), width=max(1, min(W, H) // 400))
        d.line([(0, H * k / 3), (W, H * k / 3)], fill=(70, 74, 86), width=max(1, min(W, H) // 400))
    return base


class MOSSLayoutFromQwenVL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([MODE_IMAGE, MODE_PROMPT], {"default": MODE_IMAGE,
                         "tooltip": "From image: detect and caption a reference image. From prompt (experimental): no image, Qwen designs the whole box layout from your prompt."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Prompt mode only: describe the image you want. Qwen turns it into a full box layout."}),
                "meta": ("STRING", {"default": "photo",
                                    "tooltip": "Style lock. Comma-separated tags, e.g. photo, poster. The overall caption is built around them (added if Qwen forgets) and words from other styles are stripped from all captions. In prompt mode, leave empty to take the style from your prompt."}),
                "targets": ("STRING", {"multiline": True, "default": "person",
                                       "tooltip": "Image mode: one thing to detect per line (extra when smart_mode is on). Prompt mode: elements that must be included. For text, say so explicitly, e.g. 'title text EXPLORE'."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.01,
                                         "tooltip": "Image mode: the image is scaled to this many total megapixels (0 = keep original size). Prompt mode: size of the canvas."}),
                "resolution_steps": ("INT", {"default": 16, "min": 1, "max": 256,
                                             "tooltip": "Width and height are snapped to a multiple of this. MOSS needs multiples of 16."}),
                "upscale_method": (UPSCALE_METHODS, {"default": "lanczos", "tooltip": "Image mode: resampling used to scale the image."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                                  "tooltip": "0 = automatic. Set both width and height to force a canvas size instead (rounded to 16)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                                   "tooltip": "0 = automatic. Set both width and height to force a canvas size instead (rounded to 16)."}),
                "coord_mode": (["normalized_0_1000", "absolute_pixels"],
                               {"default": "normalized_0_1000",
                                "tooltip": "Qwen3-VL/3.5 use normalized 0-1000. Use absolute_pixels for Qwen2.5-VL style models. Ignored in prompt mode."}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Reference image (image mode)."}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1",
                                                 "tooltip": "Prompt mode: canvas shape. Combined with megapixels and resolution_steps (or set width and height)."}),
                "clip": ("CLIP", {"tooltip": "Qwen3-VL / Qwen3.5-VL loaded as a text encoder. If unconnected, hf_model is used."}),
                "hf_model": ("STRING", {"default": "Qwen/Qwen3-VL-8B-Instruct",
                                        "tooltip": "HF repo id or local folder. Used only when clip is not connected."}),
                "keep_loaded": ("BOOLEAN", {"default": False,
                                            "tooltip": "HF path only: keep Qwen in memory after running."}),
                "change_request": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "Image mode: edits to make, one per line (e.g. 'make the dress red', 'add a dog on the left'). Prompt mode: extra instructions for the layout."}),
                "box_padding": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5,
                                          "tooltip": "Grow every box by this % of its size on each side (clamped to the canvas)."}),
                "smart_mode": ("BOOLEAN", {"default": False,
                                           "tooltip": "Image mode: Qwen decides what to detect by itself (people and their body parts, clothing, objects, background regions). Anything in targets is still included."}),
                "max_items": ("INT", {"default": 40, "min": 5, "max": 120, "step": 1,
                                      "tooltip": "Smart mode and prompt mode: maximum number of regions Qwen may return."}),
                "style_in_items": ("BOOLEAN", {"default": True,
                                               "tooltip": "Every item caption must carry the meta style ('Rendered as <meta>.' is added if it doesn't). Turn off to keep style only in the overall caption."}),
                "depth_order": (DEPTH_MODES, {"default": "back to front",
                                              "tooltip": "Qwen assigns each item a depth layer, captions say what is in front of or behind what, and items are sorted by it. 'off' keeps Qwen's own order."}),
                "enable_thinking": ("BOOLEAN", {"default": False,
                                                "tooltip": "Let Qwen reason before answering. Can be more accurate, but the reasoning uses up max_new_tokens and can cut the JSON off. Keep off for smart mode."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("prompt", "preview", "raw_qwen_output", "width", "height", "scaled_image")
    FUNCTION = "run"
    CATEGORY = "MOSS-Image"

    def run(self, mode, prompt, meta, targets, megapixels, resolution_steps, upscale_method, width, height,
            coord_mode, max_new_tokens, temperature, seed, image=None, aspect_ratio="1:1", clip=None,
            hf_model="", keep_loaded=False, change_request="", box_padding=0.0, smart_mode=False,
            max_items=40, style_in_items=True, depth_order="back to front", enable_thinking=False):
        text_mode = mode == MODE_PROMPT
        size_note = ""

        # ---- canvas + (image mode) the scaled reference image ----
        if text_mode:
            if not prompt.strip():
                raise ValueError("Prompt mode needs text in the prompt box.")
            if width > 0 and height > 0:
                W, H = _round16(width), _round16(height)
            else:
                W, H = _size_from_aspect(aspect_ratio, megapixels, resolution_steps)
            pil, iw, ih, scaled = None, W, H, None
            coord_eff = "normalized_0_1000"
            if width > 0 and height > 0:
                size_note = f"canvas {W}x{H} (forced)"
            else:
                size_note = f"canvas {W}x{H} ({aspect_ratio}, {megapixels:g} MP)"
        else:
            if image is None:
                raise ValueError("Connect an image, or switch mode to 'From prompt (experimental)'.")
            scaled, sw, sh = _scale_image(image, megapixels, resolution_steps, upscale_method)
            pil = _tensor_to_pil(scaled)
            iw, ih = pil.size
            W = _round16(width) if width > 0 else sw
            H = _round16(height) if height > 0 else sh
            coord_eff = coord_mode
            size_note = (f"image scaled to {sw}x{sh} ({megapixels:g} MP, steps {resolution_steps}, {upscale_method})"
                         if megapixels > 0 else f"image kept at {sw}x{sh} (snapped to steps of {resolution_steps})")
            if (W, H) != (sw, sh):
                size_note += f"; canvas forced to {W}x{H}"
        if W % 16 or H % 16:
            size_note += " | WARNING: MOSS needs multiples of 16, set resolution_steps to 16"

        # ---- instructions for Qwen ----
        target_list = [t.strip() for t in targets.splitlines() if t.strip()]
        text_ok = _wants_text(prompt if text_mode else "", targets, change_request)
        if text_mode:
            q_prompt = _build_layout_prompt_text(prompt, W, H, target_list, change_request, meta, max_items,
                                                 text_ok, style_in_items, depth_order)
        else:
            q_prompt = _build_qwen_prompt(target_list, W, H, iw, ih, coord_eff, change_request, meta,
                                          smart_mode, max_items, text_ok, style_in_items, depth_order)

        model_id = hf_model.strip()
        if clip is None and not model_id:
            raise ValueError("Connect a CLIP (Qwen VL) or provide hf_model.")

        eff_tokens = max_new_tokens
        if smart_mode or text_mode:          # many items need many tokens: max_new_tokens is a floor here
            eff_tokens = min(16384, max(max_new_tokens, 600 + int(max_items) * 140))
        state = {"cut": False}

        def ask(p, keep):
            if clip is not None:
                out = _generate_with_clip(clip, pil, p, eff_tokens, temperature, seed, enable_thinking)
            else:
                out = _generate_with_hf(model_id, pil, p, eff_tokens, temperature, seed, keep, enable_thinking)
            d = _extract_json(out)
            if isinstance(d, list):
                d = {"caption": "", "items": d}
            if isinstance(d, dict) and d.pop("_salvaged", False):
                state["cut"] = True
            return out, (d if isinstance(d, dict) else None)

        raw, data = ask(q_prompt, keep_loaded)
        if data is None:
            hint = ("\nQwen ran out of tokens while still thinking. Turn enable_thinking off or raise max_new_tokens."
                    if "<think>" in raw and "</think>" not in raw else "")
            raise RuntimeError("Could not parse JSON from Qwen output." + hint + " See raw_qwen_output:\n" + raw[:1500])

        # ---- clean up and convert ----
        tags = _meta_tags(meta)
        style_free = text_mode and not tags      # prompt mode with no meta: the prompt's own style is allowed
        item_alt, summ_alt = (None, None) if style_free else _style_patterns(tags)
        removed = []
        mono = _is_mono(tags)
        recolored = 0
        items_styled = 0

        rows = []
        for it in data.get("items", []) or []:
            if not isinstance(it, dict):
                continue
            bb = it.get("bbox_2d") or it.get("bbox")
            moss_bb = _to_moss_bbox(bb, W, H, iw, ih, coord_eff, box_padding) if bb else None
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
            try:
                layer = int(float(it.get("layer")))
            except (TypeError, ValueError):
                layer = None
            rows.append((layer, len(rows), entry))

        # depth ordering: sort by the layer Qwen assigned (items without one go to the front)
        order_note = ""
        if depth_order != "off" and rows:
            known = [l for l, _, _ in rows if l is not None]
            default = (max(known) + 1) if known else 0
            sign = 1 if depth_order == "back to front" else -1
            rows.sort(key=lambda r: (sign * (r[0] if r[0] is not None else default), r[1]))
            order_note = f" | items ordered {depth_order}" + ("" if known else " (Qwen gave no layers, order kept)")
        items_out = [e for _, _, e in rows]

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

        # ---- preview ----
        if text_mode:
            prev = _blank_canvas(W, H)
            scaled_out = _pil_to_tensor(prev)          # blank canvas of the right size
        else:
            prev = pil.resize((W, H), Image.LANCZOS)
            scaled_out = scaled
        _draw_preview(prev, items_out, W, H)

        meta_note = f"\n\n[size] {size_note}\n[meta] locked: {', '.join(tags) or '(none)'}"
        if style_free:
            meta_note = meta_note.replace("locked: (none)", "none (style taken from your prompt)")
        if removed:
            meta_note += " | removed conflicting style words: " + ", ".join(sorted(set(removed)))
        if state["cut"]:
            meta_note += (f" | WARNING: Qwen's answer was cut off at {eff_tokens} tokens; {len(items_out)} complete "
                          "item(s) were recovered. Raise max_new_tokens, lower max_items, or keep enable_thinking off")
        if not items_out:
            meta_note += (" | WARNING: no boxes were produced. " +
                          ("Try a more detailed prompt or raise max_new_tokens" if text_mode
                           else "Check targets / smart_mode, or raise max_new_tokens"))
        if recolored:
            meta_note += f" | no-color style: {recolored} color word(s) converted to gray tones"
        if items_styled:
            meta_note += f" | style clause added to {items_styled} item caption(s)"
        if added:
            meta_note += " | overall caption did not state the style, so it was added"
        meta_note += order_note
        return (out_prompt, _pil_to_tensor(prev), raw + meta_note, W, H, scaled_out)


NODE_CLASS_MAPPINGS = {"MOSSLayoutFromQwenVL": MOSSLayoutFromQwenVL}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSLayoutFromQwenVL": "MOSS Layout from Qwen-VL"}
