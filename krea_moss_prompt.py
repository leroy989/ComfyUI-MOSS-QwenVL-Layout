"""
Krea 2 / MOSS Image Prompt Builder  -  single-file ComfyUI custom node
=======================================================================

Part of ComfyUI-MOSS-QwenVL-Layout: loaded automatically by this pack's __init__.py.
The node appears as  "Krea 2 / MOSS Image Prompt Builder"  under  text/prompt.

How it works
------------
It uses ComfyUI's native CLIP text-generation API (the same calls the core
"Generate Text" node makes):  clip.tokenize() -> clip.generate() -> clip.decode().
So the CLIP you plug in must be an LLM-type text encoder that can generate text
(e.g. Qwen3.5 / Qwen3-VL / Gemma). For image mode it must also be vision capable.
A plain SD/SDXL CLIP cannot do this.

Modes
-----
* Style box : optional single-line style_override forces every mode into that style
              (in image mode the image's own style is ignored)
* Text      : your short idea  -> fully fleshed-out prompt
* Image     : toggle use_image -> prompt that describes the connected image
* Wildcard  : toggle use_wildcard -> your unordered list of ingredients (and/or
              nothing at all) -> a brand-new invented scene built from them
Each mode writes in the format of the selected target model.

Sources the prompt rules were written from
------------------------------------------
Krea 2  : github.com/krea-ai/krea-2  docs/prompting.md and docs/expansion.txt
MOSS    : github.com/OpenMOSS/MOSS-Image  USERGUIDE.md and
          skills/moss-image-prompt-enhancer/SKILL.md
(The system prompts below are written in my own words from those documents.)
"""

import inspect
import random
import re

# --------------------------------------------------------------------------- #
#  System prompts (one per target model)
# --------------------------------------------------------------------------- #

KREA_SYSTEM = """You write prompts for Krea 2, an aesthetic-first text-to-image model. \
Krea 2 responds best to natural language that reads like a precise description of a finished image.

OUTPUT CONTRACT
- Reply with ONE finished prompt and nothing else. No preface, no labels, no notes, no explanations.
- One single cohesive paragraph. No bullet points, no JSON, no markdown, no headings.
- Do not wrap the whole prompt in quotation marks.

HOW TO WRITE IT
- Settle the medium and look first (photograph, 35mm film still, macro photograph, digital painting, \
flat-color illustration, 3D render, risograph print, ink illustration, cel-shaded anime, collage...). \
Let that choice shape every other detail. If the user names a medium (photo, painting, sketch, \
3D render, illustration...), keep exactly that medium.
- Describe each subject together with its own attributes: appearance, pose, action, expression, \
clothing and materials. Use grounded, concrete phrasing for poses, interactions and spatial layout \
(foreground and background, left and right, high or low angle).
- State the framing and camera: shot distance, angle, perspective, depth of field.
- State the light and color: direction and quality of light, time of day, palette, contrast, \
background color.
- Add surface and texture cues that sell the medium: film grain, brushstrokes, stippling, paper \
texture, motion blur, glossy reflections, volumetric haze.
- Prefer concrete visual facts over praise. Never use filler tags such as "masterpiece", "8k", \
"best quality" or "trending on artstation".
- If the image should contain visible text (a sign, title, label, caption), write the exact words \
inside double quotes and say where and how they appear.

FAITHFULNESS
- Keep every subject, action, color, count and spatial relationship the user gave.
- Do not add new objects, props, characters or animals the user did not clearly imply.
- Do not invent highly specific clothing, colors, materials or brand details the input does not \
support. Richness should come mostly from composition, light, texture and atmosphere.
- If the user's prompt is already detailed, lightly polish and finalize it. Keep their wording and \
direction instead of expanding heavily.
- Depict people with dignity; assume clothing covers intimate anatomy."""


MOSS_SYSTEM = """You write image captions that are used as prompts for MOSS-Image, a text-to-image \
model driven by an LLM text encoder and trained on long, detailed captions. It follows a full scene \
description, so write the prompt as a cohesive natural-language description of the finished image.

OUTPUT CONTRACT
- Reply with the caption only: plain prose. No preface, no labels, no markdown, no bullet lists, \
no JSON, no tag lists, no notes about your process.
- Write in English unless the user asks for another language. Text that must appear inside the \
image stays verbatim in its original language.
- Do not wrap the whole caption in quotation marks.

WHAT A GOOD CAPTION CONTAINS
- Overall impression and composition first, then the key subjects and what they are doing, then the \
setting and supporting elements, with colors, lighting, materials, mood, camera or framing, and the \
medium or artistic style woven in where they belong.
- Spatial relations written in words ("in the lower-left foreground", "behind her", "to the right \
of the lamp"). Never use coordinates or bounding boxes.
- Visible choices only: describe what can be seen, not intentions, process or things outside the frame.
- No quality-tag stacks ("masterpiece, best quality, 8k") and no watermarks, signatures or logos \
unless the user asks for them.
- Keep an explicitly minimalist request minimal. Develop a rich request fully without falling back \
on a generic title / hero / footer template.
- The caption may use several paragraphs when the scene is complex.

RENDERED TEXT
- When words must appear in the image, put the exact wording in double quotes and keep the \
capitalization, punctuation and requested line breaks as given (put each line break on its own line \
inside the quotes). Describe where the text sits, its size relative to other elements, color, and \
style (handwritten, printed, neon, stencil...).
- Never write placeholders such as "headline here". If the user gave no copy but the image needs \
text, write the actual copy.

FAITHFULNESS
- Preserve the user's subjects, counts, relationships, medium, style, exclusions and intended wording.
- Fill ordinary creative gaps yourself, but do not present invented facts (dates, prices, addresses) \
as if the user supplied them.
- Keep photographs as photographs, illustrations as illustrations, paintings as paintings."""


# --------------------------------------------------------------------------- #
#  Per-target configuration
# --------------------------------------------------------------------------- #

TARGETS = {
    "Krea 2": {
        "system": KREA_SYSTEM,
        "single_paragraph": True,
        # Krea 2 Turbo runs at CFG 0 (no negative used); RAW has no documented negative tags.
        "negative": "",
        "image_focus": (
            "Lead with the medium and overall look, then the subject and its attributes, the framing "
            "and camera angle, the lighting and color palette, and the texture cues (grain, brushwork, "
            "blur, paper, gloss) that define the aesthetic."
        ),
        "expand_focus": (
            "Decide the medium and look yourself if the idea does not state one, keeping any medium "
            "the idea does state."
        ),
        "wildcard_focus": (
            "Commit to one clear medium and look, then describe the subject, framing, light, palette "
            "and texture."
        ),
    },
    "MOSS Image": {
        "system": MOSS_SYSTEM,
        "single_paragraph": False,
        # Effective negative tags recommended in the official MOSS-Image user guide.
        "negative": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
        "image_focus": (
            "Cover the composition, key subjects and actions, setting, colors, lighting and artistic "
            "style. If there is visible text, reproduce it exactly (no translating, paraphrasing or "
            "normalizing) and describe its layout: line breaks, reading order, position, alignment, "
            "size hierarchy, font style and color."
        ),
        "expand_focus": (
            "Write it as a full scene caption: composition, subjects and actions, setting, colors, "
            "lighting, medium or style."
        ),
        "wildcard_focus": (
            "Write it as a full scene caption: composition, subjects and actions, setting, colors, "
            "lighting, medium or style."
        ),
    },
}

DETAIL = {
    "Concise": (
        "Keep it compact: a tight description that nails the medium, the subject, the framing and the "
        "light. Leave out minor secondary details."
    ),
    "Standard": (
        "Make it well rounded: the medium, subject details, setting, framing, lighting and palette "
        "each get their place."
    ),
    "Detailed": (
        "Make it long and rich: develop the subject, setting, composition, light, color palette, "
        "textures and atmosphere thoroughly, while staying faithful to the request."
    ),
}

# Small pools of "creative nudges" used only in wildcard mode. Small LLMs are poor at being
# random on their own, so Python draws these from the seed and hands them over.
_MEDIUMS = [
    "35mm film photograph", "high-fashion editorial photograph", "macro photograph",
    "cinematic movie still", "digital painting", "gouache illustration",
    "flat-color vector illustration", "3D render with studio lighting", "risograph print",
    "ink and watercolor illustration", "1990s cel-animation frame", "oil painting",
    "analog paper collage", "charcoal and pencil sketch", "stylized claymation still",
    "long-exposure night photograph", "pointillist painting", "vintage travel poster",
]
_LIGHTING = [
    "soft overcast daylight", "hard direct sunlight with sharp shadows", "golden hour backlight",
    "moody low-key lighting", "neon glow in the dark", "diffused studio softbox light",
    "dramatic shafts of light through dust", "cool blue moonlight", "flat even lighting",
    "warm candle-like glow", "harsh flash photography", "high-key bright airy light",
]
_FRAMING = [
    "extreme close-up", "medium close-up portrait framing", "wide establishing shot",
    "low-angle hero perspective", "high-angle overhead view", "tightly cropped detail shot",
    "dynamic tilted framing", "centered symmetrical composition", "off-center composition with lots of negative space",
    "over-the-shoulder view", "shallow depth of field with a blurred foreground",
]
_PALETTES = [
    "muted earthy tones", "saturated primary colors", "pastel peach and mint",
    "black and white with one accent color", "teal and orange contrast", "monochrome blue",
    "warm sepia", "vivid crimson and deep black", "faded vintage colors", "neon magenta and cyan",
    "soft greens and creams", "high-contrast black and gold",
]

_LABEL_PREFIXES = ("prompt", "caption", "expanded prompt", "final prompt", "image prompt", "output")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def parse_items(text):
    """Split a wildcard list into items. Newline-separated if there are newlines
    (so items may contain commas); otherwise comma-separated. De-duplicated."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"[\r\n]+", text) if "\n" in text else text.split(",")
    seen, items = set(), []
    for p in parts:
        p = p.strip(" \t-\u2022*")
        key = p.lower()
        if p and key not in seen:
            seen.add(key)
            items.append(p)
    return items


def _style_block(style, mode):
    """Mandatory style instruction appended to the task. Empty when no override is set."""
    style = (style or "").strip()
    if not style:
        return ""
    block = (
        "\n\nSTYLE OVERRIDE (mandatory, highest priority): render the whole prompt in this style: "
        f"{style}\n"
        "- State this style clearly near the start, then let it shape the medium, color, lighting, "
        "texture and the vocabulary used throughout.\n"
        "- This overrides any other instruction about choosing or keeping a medium or style.\n"
        "- Do not use words that belong to a different style or medium (for example photographic "
        "terms such as lens, bokeh or film grain for a painting style)."
    )
    if mode == "image":
        block += (
            "\n- IGNORE the attached image's own style, medium, color grading and rendering. Take only "
            "its content from the image (subjects, actions, setting, composition, visible text) and "
            "describe all of it as it would appear in the override style."
        )
    elif mode == "text":
        block += "\n- If the idea names a different medium or style, the override wins."
    else:
        block += "\n- If an ingredient names a different medium or style, the override wins."
    return block


def build_user_message(cfg, detail, prompt, use_image, use_wildcard, items, rng, style=""):
    """Create the task message for the LLM. Returns (message, mode_name)."""
    detail_text = DETAIL[detail]

    if use_image:
        msg = (
            "Look at the attached image and write the prompt that would recreate it, following the "
            "format in your instructions.\n"
            "- Describe only what is actually visible. Do not mention \"the image\" or \"the photo\"; "
            "describe the scene directly.\n"
            "- Do not describe watermarks.\n"
            f"- {cfg['image_focus']}\n"
            f"- {detail_text}"
        )
        if prompt:
            msg += (
                "\n\nThe user also gave this direction. Apply it while describing, so the result "
                f"reflects it:\n{prompt}"
            )
        return msg + _style_block(style, "image"), "image"

    if use_wildcard:
        lines = ["Invent an original image and write the final prompt for it."]
        if prompt:
            lines.append(f"\nCore idea to keep:\n{prompt}")
        if items:
            bullet = "\n".join(f"- {i}" for i in items)
            lines.append(
                "\nIngredients supplied by the user (an unordered pool; use them as the creative core "
                "and weave them into ONE coherent scene; include all of them unless one truly cannot "
                f"fit):\n{bullet}"
            )
        else:
            lines.append("\nThe user gave no ingredients, so invent the subject yourself.")
        medium = rng.choice(_MEDIUMS)  # always drawn so other picks stay identical per seed
        nudge_parts = [] if style else [f"medium/look: {medium}"]
        nudge_parts += [f"light: {rng.choice(_LIGHTING)}", f"framing: {rng.choice(_FRAMING)}",
                        f"palette: {rng.choice(_PALETTES)}"]
        nudges = "; ".join(nudge_parts)
        lines.append(
            f"\nCreative nudges for this run: {nudges}.\n"
            "Use a nudge only where the ingredients or the core idea do not already decide that "
            "aspect. If a nudge conflicts with them, ignore the nudge."
        )
        lines.append(
            "\nInvent everything else yourself (setting, action, composition, small details) so the "
            "result feels specific and distinctive rather than generic. "
            f"{cfg['wildcard_focus']}\n{detail_text}"
        )
        return "\n".join(lines) + _style_block(style, "wildcard"), "wildcard"

    msg = (
        "Write the final prompt for this image idea.\n\n"
        f"Image idea:\n{prompt}\n\n"
        f"{cfg['expand_focus']}\n{detail_text}"
    )
    return msg + _style_block(style, "text"), "text"


def _tokenize(clip, user_text, system_prompt, image, thinking):
    """Call clip.tokenize with the arguments the core Generate Text node uses.
    Image kwarg name differs between ComfyUI versions, so fall back gracefully."""
    base = dict(skip_template=False, min_length=1, thinking=thinking, system_prompt=system_prompt)
    if image is None:
        return clip.tokenize(user_text, **base)

    attempts = (
        dict(images=[image], image=image),  # works on both old and new builds
        dict(images=[image]),
        dict(image=image),
    )
    last_err = None
    for img_kw in attempts:
        try:
            return clip.tokenize(user_text, **base, **img_kw)
        except TypeError as e:  # tokenizer rejected an unknown kwarg
            last_err = e
    raise RuntimeError(
        "This CLIP/text encoder rejected the image input. Use a vision-capable LLM text encoder "
        f"(e.g. Qwen3.5 / Qwen3-VL) and a recent ComfyUI. Original error: {last_err}"
    )


def _generate(clip, tokens, **kw):
    """clip.generate, passing only the kwargs this ComfyUI build supports."""
    try:
        params = inspect.signature(clip.generate).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        pass
    return clip.generate(tokens, **kw)


def _decode(clip, ids):
    try:
        return clip.decode(ids, skip_special_tokens=True)
    except TypeError:
        return clip.decode(ids)


def clean_output(text, single_paragraph):
    """Strip reasoning blocks, markdown, labels and wrapper quotes from LLM output."""
    text = text or ""
    # reasoning blocks (closed, or closed-only when the opening tag was primed by the template)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<\|[^|>]*\|>", "", text)  # leftover special tokens
    text = text.replace("```", "").replace("**", "")
    text = text.strip()

    # leading labels like "Prompt:" / "Caption:"
    label_re = r"^(?:%s)\s*[:\-]\s*" % "|".join(re.escape(l) for l in _LABEL_PREFIXES)
    text = re.sub(label_re, "", text, flags=re.IGNORECASE).strip()

    # wrapper quotes around the whole thing (only if no other quote of that kind inside,
    # so quoted text-to-render is never damaged)
    for q_open, q_close in (('"', '"'), ("\u201c", "\u201d"), ("'", "'")):
        if len(text) > 1 and text[0] == q_open and text[-1] == q_close and q_open not in text[1:-1] \
                and q_close not in text[1:-1]:
            text = text[1:-1].strip()
            break

    if single_paragraph:
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
    else:
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
#  The node
# --------------------------------------------------------------------------- #

class KreaMossPromptBuilder:
    CATEGORY = "text/prompt"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "An LLM-type text encoder that can generate text "
                                             "(Qwen3.5, Qwen3-VL, Gemma...). Needs vision for image mode."}),
                "target_model": (list(TARGETS.keys()), {"default": "Krea 2",
                                 "tooltip": "Which model's prompt format to write."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                           "tooltip": "Your basic idea. In image mode it becomes optional extra direction; "
                                      "in wildcard mode it is a core idea to keep."}),
                "style_override": ("STRING", {"multiline": False, "default": "",
                                   "tooltip": "Optional. If filled, the prompt is forced into this style "
                                              "in every mode. In image mode the image's own style is "
                                              "ignored. e.g. 'watercolor illustration'."}),
                "detail": (list(DETAIL.keys()), {"default": "Standard"}),
                "use_image": ("BOOLEAN", {"default": False, "label_on": "image mode ON",
                              "label_off": "image mode off",
                              "tooltip": "Describe the connected image as a prompt in the target format."}),
                "use_wildcard": ("BOOLEAN", {"default": False, "label_on": "wildcard ON",
                                 "label_off": "wildcard off",
                                 "tooltip": "Invent a scene from the ingredient list below."}),
                "wildcard_items": ("STRING", {"multiline": True, "default": "",
                                   "tooltip": "Ingredients in any order, one per line (or comma separated). "
                                              "Leave empty for a fully random scene."}),
                "wildcard_pick": ("INT", {"default": 0, "min": 0, "max": 64,
                                  "tooltip": "0 = use every ingredient. N = randomly pick N of them each run."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 768, "min": 64, "max": 4096,
                               "tooltip": "Raise this if you turn thinking on."}),
                "thinking": ("BOOLEAN", {"default": False,
                             "tooltip": "Let the LLM reason first (if the model supports it). Slower."}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def build(self, clip, target_model, prompt, detail, use_image, use_wildcard, wildcard_items,
              wildcard_pick, seed, temperature, max_tokens, thinking, style_override="", image=None):

        if not (hasattr(clip, "tokenize") and hasattr(clip, "generate") and hasattr(clip, "decode")):
            raise ValueError("The connected CLIP cannot generate text. Load an LLM-type text encoder "
                             "(e.g. Qwen3.5) with the Load CLIP node.")

        cfg = TARGETS[target_model]
        prompt = (prompt or "").strip()
        rng = random.Random(int(seed))

        img = None
        items = []
        if use_image:
            if image is None:
                raise ValueError("use_image is ON but no image is connected.")
            img = image[:1]  # first frame only
            if use_wildcard:
                print("[KreaMossPromptBuilder] image mode and wildcard are both on; image mode wins.")
        elif use_wildcard:
            items = parse_items(wildcard_items)
            if wildcard_pick and 0 < wildcard_pick < len(items):
                items = rng.sample(items, int(wildcard_pick))
        elif not prompt:
            raise ValueError("Type a prompt, or turn on image mode or wildcard mode.")

        user_msg, _mode = build_user_message(cfg, detail, prompt, use_image, use_wildcard, items, rng,
                                         (style_override or "").strip())

        tokens = _tokenize(clip, user_msg, cfg["system"], img, bool(thinking))
        ids = _generate(
            clip, tokens,
            do_sample=temperature > 0,
            max_length=int(max_tokens),
            temperature=float(temperature),
            top_k=64, top_p=0.95, min_p=0.05,
            repetition_penalty=1.05,
            seed=int(seed),
        )
        raw = _decode(clip, ids)
        result = clean_output(raw, cfg["single_paragraph"])

        if not result:
            raise RuntimeError("The LLM returned an empty prompt. If thinking is on, raise max_tokens "
                               "(reasoning uses the same token budget).")
        return (result, cfg["negative"])


NODE_CLASS_MAPPINGS = {"KreaMossPromptBuilder": KreaMossPromptBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"KreaMossPromptBuilder": "Krea 2 / MOSS Image Prompt Builder"}
