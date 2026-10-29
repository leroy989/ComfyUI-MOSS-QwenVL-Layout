# ComfyUI-MOSS-QwenVL-Layout

A ComfyUI node that turns a reference image into a **MOSS-Image layout prompt**.
Qwen3-VL / Qwen3.5-VL looks at your image, finds the things you ask for (including text, if you ask), and the node writes the layout JSON that MOSS-Image expects: captions, text content and bounding boxes converted to MOSS's `abs_centralized` / `yxyx` coordinate format.

There are two modes. **From image** (default) detects and captions a reference image. **From prompt (experimental)** needs no image: you write a prompt and Qwen designs the whole box layout for it.

You can also add **change requests** ("make the dress red", "replace the man with an elderly woman in a grey coat", "add a dog on the left"). The captions are then written to describe the *edited* result, so MOSS generates the changed version while keeping the original composition.

<img width="3134" height="1148" alt="Screenshot 2026-10-28 210728" src="https://github.com/user-attachments/assets/56e15571-0628-4309-8529-f924c7e0cec6" />
<img width="3176" height="1118" alt="Screenshot 2026-10-28 210340" src="https://github.com/user-attachments/assets/34b05d82-0342-4aaa-8cf7-cdf8f435184c" />

> Unofficial community node. Not affiliated with the MOSS-Image team or Alibaba/Qwen.

## Install

**Manual**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/leroy989/ComfyUI-MOSS-QwenVL-Layout
pip install -r ComfyUI-MOSS-QwenVL-Layout/requirements.txt   # only needed for the hf_model path
```
Restart ComfyUI. The node is at **Add Node → MOSS-Image → MOSS Layout from Qwen-VL**.

## Usage

**From image**
1. Connect an `IMAGE`. It is scaled to `megapixels` total pixels (same maths as ComfyUI's *Scale Image to Total Pixels*, with `upscale_method` and `resolution_steps`), and the scaled size becomes the `width` / `height` outputs. Wire those to your MOSS-Image latent and use `scaled_image` wherever you need the resized picture. Set `megapixels` to 0 to keep the original size.
2. List what to detect in `targets`, one per line. Text is only detected when you ask for it (for example `title text`, `the sign`).
3. Optionally fill in `change_request`, one change per line.
4. Connect `prompt` to your MOSS-Image text prompt and check `preview` before generating.

**From prompt (experimental)**
1. Set `mode` to `From prompt (experimental)` and write your idea in `prompt`.
2. Choose the canvas: `aspect_ratio` + `megapixels` + `resolution_steps`, or set both `width` and `height`.
3. Qwen (no image involved) plans every element and places its box. `targets` lists elements that must be included, `change_request` adds extra instructions, `max_items` caps the count. Clear `meta` if you want the style to come from your prompt.
4. `preview` shows the boxes on a neutral canvas with a rule-of-thirds grid. How good the layout is depends heavily on the model, so treat it as a starting point and check it.

### Choosing a Qwen backend

| Backend | How | Notes |
|---|---|---|
| **CLIP input** | Load Qwen3-VL / Qwen3.5-VL as a text encoder and connect it to `clip` | Uses ComfyUI's text generation. Needs a recent ComfyUI. **Less tested**, so check `raw_qwen_output` on first use. |
| **hf_model** | Leave `clip` empty and set `hf_model` to a Hugging Face repo or local folder | Uses `transformers>=4.57`. Most dependable. Turn `keep_loaded` off to free VRAM before MOSS runs. |

### Inputs

| Input | Description |
|---|---|
| `mode` | `From image` or `From prompt (experimental)` |
| `prompt` | Prompt mode: the image you want |
| `image` *(optional)* | Reference image (image mode) |
| `meta` | Style lock: comma-separated tags (e.g. `photo`, `anime, cel shading`). See *Meta lock* below |
| `targets` | Image mode: things to detect, one per line. Prompt mode: elements that must be included |
| `megapixels`, `resolution_steps`, `upscale_method` | Image mode: scale the image to this many total pixels (0 = keep size), snapped to steps (use 16 for MOSS). Prompt mode: canvas size |
| `width`, `height` | 0 = automatic. Set both to force a canvas size (rounded to 16) |
| `aspect_ratio` *(optional)* | Prompt mode canvas shape |
| `coord_mode` | `normalized_0_1000` (Qwen3-VL / 3.5) or `absolute_pixels` (Qwen2.5-VL style). Ignored in prompt mode |
| `max_new_tokens`, `temperature`, `seed` | Generation settings |
| `clip` *(optional)* | Qwen VL text encoder |
| `hf_model`, `keep_loaded` *(optional)* | Standalone Hugging Face path |
| `change_request` *(optional)* | Image mode: edits to apply, one per line. Prompt mode: extra instructions |
| `smart_mode` *(optional)* | Image mode: Qwen picks the targets itself. Anything in `targets` is still included |
| `max_items` *(optional)* | Smart mode and prompt mode: cap on the number of regions (default 40) |
| `style_in_items` *(optional)* | Require every item caption to carry the meta style (default on) |
| `depth_order` *(optional)* | `back to front` (default), `front to back` or `off`. See *Depth ordering* |
| `enable_thinking` *(optional)* | Let Qwen reason before answering (default off). Reasoning uses up `max_new_tokens` and can cut the JSON off |
| `box_padding` *(optional)* | Grow every box by this % per side (0 = off) |

### Outputs

| Output | Description |
|---|---|
| `prompt` | MOSS layout JSON |
| `preview` | Image with the numbered boxes drawn on it |
| `raw_qwen_output` | Unprocessed model response, for debugging |
| `width`, `height` | Final canvas size (the scaled image size in image mode). Feed these to MOSS |
| `scaled_image` | Image mode: the scaled image. Prompt mode: a blank canvas of the right size |

### Change request tips

- Describe replacements in concrete visual terms: "elderly woman, grey wool coat, short white hair" works better than "a different person".
- Boxes follow the reference image. For objects you add, Qwen guesses the position, so adjust the `bbox` values by hand if needed.
- Global changes ("golden hour lighting", "winter") are applied to the overall caption and affected items.

## Smart mode and long answers

Smart mode can return dozens of regions, which needs a lot of tokens, so in smart mode `max_new_tokens` is treated as a minimum and raised to about `600 + max_items x 140` if needed. If an answer is still cut off, the complete items are recovered instead of being lost and a warning is added to `raw_qwen_output`. Thinking is off by default for the same reason.

## Depth ordering

With `depth_order` on (default `back to front`), Qwen gives every item a depth layer (1 = farthest from the camera, higher = closer) and each caption says what the element is in front of or behind. Items are then sorted by layer, so the background comes first and the foreground last (or the reverse for `front to back`). `off` keeps Qwen's own order. The layer numbers are only used for sorting and are not written into the JSON. If Qwen gives no layers, the original order is kept and noted in `raw_qwen_output`.

## Prompt mode notes

- Boxes are placed by the model from text alone, with no image to check against, so expect to iterate. Longer, more specific prompts give better layouts.
- Qwen is told the canvas size and how it affects shape (a box with equal x and y extents is not square on a non-square canvas) and is shown a small format example.
- With a locked `meta`, other styles in your prompt are overridden by the meta. With `meta` empty, the style in your prompt is kept and no style cleanup is applied.
- Text appears only if your prompt, `targets` or `change_request` ask for it.

## Meta lock

`meta` is the single source of style, and it is enforced in several ways:
1. **Prompt:** Qwen is told the final image will be generated *only* in that style, that the reference image (or your prompt's own style words) is a layout and content reference only (its own style and colors don't carry over), and that the overall caption must open with the style words. The "describe colors" wording is swapped for "tones" when the meta is monochrome.
2. **Every item carries the style** (`style_in_items`, on by default): each item caption must say how that element is rendered in the locked style. If Qwen doesn't, `Rendered as <meta>.` is appended. Turn the option off to keep style only in the overall caption.
3. **No-color metas** (`black and white`, `monochrome`, `grayscale`, `no color`, ...): color words in any caption are converted to gray tones (`green` -> `mid-gray`, `dark blue` -> `dark gray`, `colorful` -> `high-contrast`).
4. **Other styles are stripped:** words from styles not in your meta (for example `photorealistic` or `bokeh` when the meta is `anime`) are removed from every caption. Words named in your meta are never touched. Ambiguous nouns like `photograph` or `illustration` are only checked in the overall caption, so an item like "a photo frame" survives.
5. **Forced:** if the overall caption still doesn't mention any meta tag, `Style: <tags>.` is added to the front. `summary.meta` always equals your tags exactly.

What was changed is reported at the end of `raw_qwen_output` (`[meta] locked: ...`). The cleanup is word-based, so a heavily conflicting caption can read slightly clipped.

## Text handling

Text is **off unless you ask for it**. If your `targets` or `change_request` don't mention text (words like `text`, `title`, `lettering`, `slogan`, or a quoted string), Qwen is never told text exists, any text items it returns are dropped, and quoted words are stripped from captions. To include text, say so, for example `title text EXPLORE` or `change the sign text to "OPEN"`.

## Smart mode

Turn on `smart_mode` to let Qwen decide what to target: each person plus their head, hair, torso, arms, hands, legs, feet and clothing, every distinct object, and the main background regions. Use `max_items` to cap the count, and keep entries in `targets` for anything you want guaranteed.

## Detail and occluded parts

Qwen is asked for detailed captions that state spatial relationships and occlusion explicitly (for example "left hand hidden behind the man's shoulder"). Boxes cover the element's **full estimated extent**, including parts hidden behind or inside other things, so a partly obscured object isn't cut short. If boxes are still tight, raise `box_padding` (2-5% is a good start) or add a note to `change_request`.

## How the coordinates work

MOSS layouts use absolute pixel boxes ordered `[y_min, x_min, y_max, x_max]`, with the canvas **center** as the origin. Qwen3-VL returns `[x1, y1, x2, y2]` on a 0–1000 grid. The node scales to your canvas size, subtracts `W/2` and `H/2`, and reorders the values. Because the 0–1000 grid is size-independent, the layout still maps correctly if you force a canvas with a different aspect ratio from the scaled image (the layout is stretched to fit).

## Credits and licenses

This repo contains no model weights. Please read the license terms of the models you use:
- MOSS-Image (see its model page and user guide for the layout format)
- Qwen3-VL / Qwen3.5-VL (Qwen team)

Code in this repo is MIT licensed.
