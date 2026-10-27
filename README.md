# ComfyUI-MOSS-QwenVL-Layout

A ComfyUI node that turns a reference image into a **MOSS-Image layout prompt**.
Qwen3-VL / Qwen3.5-VL looks at your image, finds the things you ask for (including text, if you ask), and the node writes the layout JSON that MOSS-Image expects: captions, text content and bounding boxes converted to MOSS's `abs_centralized` / `yxyx` coordinate format.

You can also add **change requests** ("make the dress red", "replace the man with an elderly woman in a grey coat", "add a dog on the left"). The captions are then written to describe the *edited* result, so MOSS generates the changed version while keeping the original composition.

<img width="3373" height="1205" alt="Screenshot 2026-10-27 145849" src="https://github.com/user-attachments/assets/9120a6eb-0ff7-4c3c-9d4f-5ef19cdc46e7" />

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

1. Connect an `IMAGE`.
2. Set `width` / `height` to your MOSS output size (0 = use the image's size). Values are rounded to multiples of 16, as MOSS requires.
3. List what to detect in `targets`, one per line. Text is only detected when you ask for it (for example `title text`, `the sign`).
4. Optionally fill in `change_request`, one change per line.
5. Connect the `prompt` output to your MOSS-Image text prompt.
6. Use the `preview` output to check the boxes before generating.

### Choosing a Qwen backend

| Backend | How | Notes |
|---|---|---|
| **CLIP input** | Load Qwen3-VL / Qwen3.5-VL as a text encoder and connect it to `clip` | Uses ComfyUI's text generation. Needs a recent ComfyUI. **Less tested**, so check `raw_qwen_output` on first use. |
| **hf_model** | Leave `clip` empty and set `hf_model` to a Hugging Face repo or local folder | Uses `transformers>=4.57`. Most dependable. Turn `keep_loaded` off to free VRAM before MOSS runs. |

### Inputs

| Input | Description |
|---|---|
| `image` | Reference image |
| `width`, `height` | Target MOSS canvas (0 = use image size), rounded to multiples of 16 |
| `targets` | Things to detect, one per line |
| `meta` | Comma-separated style/meta tags for `summary.meta` (e.g. `poster`, `photo`). This is the **only** place style comes from: Qwen is told not to describe art style or medium in any caption |
| `coord_mode` | `normalized_0_1000` (Qwen3-VL / 3.5) or `absolute_pixels` (Qwen2.5-VL style) |
| `max_new_tokens`, `temperature`, `seed` | Generation settings |
| `clip` *(optional)* | Qwen VL text encoder |
| `hf_model`, `keep_loaded` *(optional)* | Standalone Hugging Face path |
| `change_request` *(optional)* | Edits to apply, one per line |
| `smart_mode` *(optional)* | Qwen picks the targets itself: people and their body parts, clothing, objects, background regions. Anything in `targets` is still included |
| `max_items` *(optional)* | Smart mode only: cap on the number of regions (default 40) |
| `box_padding` *(optional)* | Grow every box by this % per side (0 = off), on top of Qwen's own estimate of hidden parts |

### Outputs

| Output | Description |
|---|---|
| `prompt` | MOSS layout JSON |
| `preview` | Image with the converted boxes drawn on it |
| `raw_qwen_output` | Unprocessed model response, for debugging |
| `width`, `height` | Final rounded canvas size |

### Change request tips

- Describe replacements in concrete visual terms: "elderly woman, grey wool coat, short white hair" works better than "a different person".
- Boxes follow the reference image. For objects you add, Qwen guesses the position, so adjust the `bbox` values by hand if needed.
- Global changes ("golden hour lighting", "winter") are applied to the overall caption and affected items.

## Text handling

Text is **off unless you ask for it**. If your `targets` or `change_request` don't mention text (words like `text`, `title`, `lettering`, `slogan`, or a quoted string), Qwen is never told text exists, any text items it returns are dropped, and quoted words are stripped from captions. To include text, say so, for example `title text EXPLORE` or `change the sign text to "OPEN"`.

## Smart mode

Turn on `smart_mode` to let Qwen decide what to target: each person plus their head, hair, torso, arms, hands, legs, feet and clothing, every distinct object, and the main background regions. Use `max_items` to cap the count, and keep entries in `targets` for anything you want guaranteed.

## Detail and occluded parts

Qwen is asked for detailed captions that state spatial relationships and occlusion explicitly (for example "left hand hidden behind the man's shoulder"). Boxes cover the element's **full estimated extent**, including parts hidden behind or inside other things, so a partly obscured object isn't cut short. If boxes are still tight, raise `box_padding` (2-5% is a good start) or add a note to `change_request`.

## How the coordinates work

MOSS layouts use absolute pixel boxes ordered `[y_min, x_min, y_max, x_max]`, with the canvas **center** as the origin. Qwen3-VL returns `[x1, y1, x2, y2]` on a 0–1000 grid. The node scales to your canvas size, subtracts `W/2` and `H/2`, and reorders the values. Because the 0–1000 grid is size-independent, the layout still maps correctly if your target canvas has a different aspect ratio from the reference image (the layout is stretched to fit).

## Credits and licenses

This repo contains no model weights. Please read the license terms of the models you use:
- MOSS-Image (see its model page and user guide for the layout format)
- Qwen3-VL / Qwen3.5-VL (Qwen team)

Code in this repo is MIT licensed.
