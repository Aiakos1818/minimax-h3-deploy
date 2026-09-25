"""Load the Z-Image text encoder (Qwen3-4B) in bf16.

ComfyUI's stock CLIPLoader requests fp16 for CUDA text encoders. Qwen3-4B
overflows in fp16, so its conditioning collapses into near-noise and Z-Image
renders an arbitrary stock photo instead of the prompt. This is the same loader
with the one thing that matters for Z-Image fixed: it always asks for bf16, the
dtype Qwen3 is trained in. The H3 clip loader sets bf16 itself, so this node is
only wired into the Z-Image workflows.
"""

import folder_paths
import torch

import comfy.sd


class ZImageCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_name": (folder_paths.get_filename_list("text_encoders"),)},
                "optional": {"device": (["default", "cpu"], {"advanced": True})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"

    def load_clip(self, clip_name, device="default"):
        model_options = {"dtype": torch.bfloat16}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                                  embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=comfy.sd.CLIPType.LUMINA2,
                                  model_options=model_options)
        return (clip,)


NODE_CLASS_MAPPINGS = {"ZImageCLIPLoader": ZImageCLIPLoader}

NODE_DISPLAY_NAME_MAPPINGS = {"ZImageCLIPLoader": "Load CLIP (Z-Image, bf16)"}
