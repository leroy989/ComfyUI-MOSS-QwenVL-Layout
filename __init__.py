from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Krea 2 / MOSS Image Prompt Builder (separate module; a failure here
# never prevents the MOSS Layout node above from loading).
try:
    from .krea_moss_prompt import (
        NODE_CLASS_MAPPINGS as _KREA_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _KREA_NAMES,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_KREA_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_KREA_NAMES}
except Exception as _e:
    print(f"[ComfyUI-MOSS-QwenVL-Layout] Could not load Krea/MOSS Prompt Builder: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
