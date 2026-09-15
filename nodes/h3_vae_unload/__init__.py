"""Move the MiniMax H3 video VAE out of VRAM once it is no longer needed.

The VAE is loaded twice per prompt on this setup: for keyframe encoding before
sampling (image/first-last-frame workflows) and for decoding after it. Keeping
it in VRAM during sampling starves the RayLight sampler, so this node passes a
value through while freeing the connected VAE to host memory.
"""

import logging

import comfy.model_management
import torch


def log_vram(tag):
    if not torch.cuda.is_available():
        return
    parts = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        parts.append("cuda:%d free=%.2f/%.2fGB" % (index, free / 2**30, total / 2**30))
    logging.info("VRAM[%s] %s", tag, ", ".join(parts))


class AnyType(str):
    def __ne__(self, __value):
        return False


class UnloadVideoVAE:
    """Pass a value through while moving the connected VAE out of VRAM."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",), "anything": (AnyType("*"),)},
                "optional": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = (AnyType("*"),)
    RETURN_NAMES = ("output",)
    FUNCTION = "unload"
    CATEGORY = "model/patch"

    def unload(self, vae, anything, ray_actors=None):
        log_vram("before-unload")
        patcher = getattr(vae, "patcher", None)
        if patcher is not None:
            comfy.model_management.unload_model_and_clones(patcher, all_devices=True)
            comfy.model_management.soft_empty_cache()
        # Release the Ray workers' cached CUDA blocks. Their pools otherwise keep
        # the sampler's transients reserved across prompts, which is what leaves
        # the next prompt's VAE encode without room next to the resident
        # text encoder and UNet shards.
        workers = (ray_actors or {}).get("workers") or []
        if workers:
            import ray
            ray.get([worker.free_cached_vae.remote() for worker in workers])
        log_vram("after-unload")
        return (anything,)


NODE_CLASS_MAPPINGS = {"UnloadVideoVAE": UnloadVideoVAE}

NODE_DISPLAY_NAME_MAPPINGS = {"UnloadVideoVAE": "Unload Video VAE"}
