# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import time
import types

import torch

import comfy.sd
import folder_paths

def _log_vram(tag):
    if not torch.cuda.is_available():
        return
    parts = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        parts.append("cuda:%d free=%.2f/%.2fGB" % (index, free / 2**30, total / 2**30))
    logging.info("VRAM[%s] %s", tag, ", ".join(parts))


COND_CACHE_ROOT = os.path.join(os.path.expanduser("~/MiniMax-H3-Deploy"), "cond_cache")


def _cond_cache_key(clip_name, tokens, unprojected, add_dict):
    h = hashlib.sha256()
    h.update(clip_name.encode())
    h.update(b"\x00")
    h.update(json.dumps(tokens, sort_keys=True, default=str, separators=(",", ":")).encode())
    h.update(b"\x00")
    h.update(str(unprojected).encode())
    h.update(b"\x00")
    h.update(json.dumps(add_dict, sort_keys=True, default=str, separators=(",", ":")).encode())
    return h.hexdigest()


def _cond_cache_dir(key):
    return os.path.join(COND_CACHE_ROOT, key[:2], key)


def _cond_cache_save(key, cond):
    d = _cond_cache_dir(key)
    os.makedirs(d, exist_ok=True)
    tensors = {}
    meta_list = []
    for i, (t, m) in enumerate(cond):
        tensors["cond%d" % i] = t.contiguous()
        meta = {}
        t_idx = 0
        for mk, mv in (m or {}).items():
            if isinstance(mv, tuple):
                mv = list(mv)
            if hasattr(mv, "detach") and hasattr(mv, "cpu"):
                tensors["meta%d_%d" % (i, t_idx)] = mv.detach().cpu().contiguous()
                meta[mk] = {"__tensor_file__": "meta%d_%d" % (i, t_idx)}
                t_idx += 1
            else:
                try:
                    json.dumps(mv)
                    meta[mk] = mv
                except Exception:
                    meta[mk] = {"__unserial__": type(mv).__name__}
        meta_list.append(meta)
    from safetensors.torch import save_file
    save_file(tensors, os.path.join(d, "tensors.safetensors"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"entries": meta_list, "ntensors": len(cond)}, f)


def _cond_cache_load(key):
    d = _cond_cache_dir(key)
    tpath = os.path.join(d, "tensors.safetensors")
    mpath = os.path.join(d, "meta.json")
    if not (os.path.isfile(tpath) and os.path.isfile(mpath)):
        return None
    from safetensors.torch import load_file
    ts = load_file(tpath)
    meta_json = json.load(open(mpath))
    out = []
    for i, meta in enumerate(meta_json["entries"]):
        t = ts.get("cond%d" % i)
        if t is None:
            return None
        m = {}
        for mk, mv in meta.items():
            if isinstance(mv, dict) and "__tensor_file__" in mv:
                m[mk] = ts.get(mv["__tensor_file__"])
                if m[mk] is None:
                    return None
            else:
                m[mk] = mv
        out.append([t, m])
    return out


def _module_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters(recurse=True))


def _move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    return value


def _layer_pre_hook(module, args, kwargs):
    device = module._h3_mp_device
    # comfy-kitchen's CUDA backend exports quantized tensors through DLPack.
    # DLPack requires the current CUDA device to match the tensor's device,
    # while moving activations alone does not update PyTorch's current device.
    torch.cuda.set_device(device)
    return _move_tensors(args, device), _move_tensors(kwargs, device)


class H3MultiGPUPlacement:
    def __init__(self, clip, gpu_ids):
        self.clip = clip
        self.devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
        clip_model = getattr(clip.cond_stage_model, "qwen3vl_32b", None)
        if clip_model is None or not hasattr(clip_model, "transformer"):
            raise ValueError("The selected text encoder is not a MiniMax H3 Qwen3-VL-32B checkpoint")
        self.qwen = clip_model.transformer
        self.retain_cpu_weights = os.environ.get("H3_MP_RETAIN_CPU_WEIGHTS", "0") == "1"
        # state_dict() returns detached tensors sharing the original CPU
        # storages. Holding those references lets the module's .to(cuda)
        # allocate GPU copies without discarding the CPU masters. After the
        # encode we can rebind parameters to these tensors instead of copying
        # the full model back over PCIe.
        self.cpu_state = self.qwen.state_dict() if self.retain_cpu_weights else None
        if self.cpu_state is not None and any(value.device.type != "cpu" for value in self.cpu_state.values() if torch.is_tensor(value)):
            raise RuntimeError("H3 retained weight snapshot must be captured on CPU")
        self.decoder = self.qwen.model
        self.layers = list(self.decoder.layers)
        self.layer_devices = self._plan_layer_devices()
        for layer, device in zip(self.layers, self.layer_devices):
            layer._h3_mp_device = device
            layer.register_forward_pre_hook(_layer_pre_hook, with_kwargs=True)

    def _plan_layer_devices(self):
        fixed = _module_bytes(self.qwen.visual) + _module_bytes(self.decoder.embed_tokens)
        layer_sizes = [_module_bytes(layer) for layer in self.layers]
        target = (fixed + sum(layer_sizes)) / len(self.devices)
        device_index = 0
        used = fixed
        placements = []
        for index, layer_size in enumerate(layer_sizes):
            remaining_layers = len(layer_sizes) - index
            remaining_devices = len(self.devices) - device_index
            if device_index + 1 < len(self.devices) and used + layer_size > target and remaining_layers >= remaining_devices:
                device_index += 1
                used = 0
            placements.append(self.devices[device_index])
            used += layer_size
        return placements

    def offload_embedding(self):
        """Keep the token embedding table off the GPUs between encodes.

        It is read once per encode for the token lookup; holding its ~1.5 GiB on
        the first GPU leaves the Ray sampler without room once image/video
        conditioning makes the packed sequence longer. `dispatch()` puts it back
        on the GPU before the next encode.
        """
        self.decoder.embed_tokens.to(torch.device("cpu"))
        torch.cuda.empty_cache()

    def dispatch(self):
        first_device = self.devices[0]
        # Text-to-video never runs the vision tower; keeping its int4 weights on
        # the CPU frees ~1.1 GiB on the first GPU, which is what lets the Ray
        # sampler fit next to the resident UNet/CLIP shards. Set
        # H3_MP_KEEP_VISUAL_CPU=0 when running image/video-reference workflows.
        if os.environ.get("H3_MP_KEEP_VISUAL_CPU", "1") == "1":
            self.qwen.visual.to(torch.device("cpu"))
        else:
            self.qwen.visual.to(first_device)
        self.decoder.embed_tokens.to(first_device)
        if self.decoder.norm is not None:
            self.decoder.norm.to(self.devices[-1])
        for layer, device in zip(self.layers, self.layer_devices):
            layer.to(device)
        torch.cuda.empty_cache()

        totals = {str(device): 0 for device in self.devices}
        totals[str(first_device)] += _module_bytes(self.qwen.visual) + _module_bytes(self.decoder.embed_tokens)
        for layer, device in zip(self.layers, self.layer_devices):
            totals[str(device)] += _module_bytes(layer)
        logging.info("H3 Qwen model-parallel placement: %s", ", ".join(f"{device}={size / 2**30:.2f} GiB" for device, size in totals.items()))

    def offload(self):
        if self.cpu_state is None:
            logging.warning("H3 Qwen: offload_after_encode requested but CPU weight masters were not retained; set H3_MP_RETAIN_CPU_WEIGHTS=1 to enable it.")
            return
        self.qwen.load_state_dict(self.cpu_state, strict=True, assign=True)
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            return
        self.qwen.visual.to("cpu")
        self.decoder.embed_tokens.to("cpu")
        if self.decoder.norm is not None:
            self.decoder.norm.to("cpu")
        for layer in self.layers:
            layer.to("cpu")
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def _parse_gpu_ids(value):
    try:
        gpu_ids = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError("gpu_ids must be a comma-separated list such as 0,1,2,3") from exc
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must contain at least one unique GPU index")
    count = torch.cuda.device_count()
    if any(gpu_id < 0 or gpu_id >= count for gpu_id in gpu_ids):
        raise ValueError(f"gpu_ids must be within the visible CUDA range 0-{count - 1}")
    return gpu_ids


class H3MultiGPUCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "gpu_ids": ("STRING", {"default": "0,1,2,3"}),
                "offload_after_encode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"

    def load_clip(self, clip_name, gpu_ids, offload_after_encode=True):
        selected_gpus = _parse_gpu_ids(gpu_ids)
        cpu = torch.device("cpu")
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options={"load_device": cpu, "offload_device": cpu, "initial_device": cpu, "dtype": torch.bfloat16},
            disable_dynamic=True,
        )
        placement = H3MultiGPUPlacement(clip, selected_gpus)
        original_encode = clip.encode_from_tokens_scheduled

        def encode_model_parallel(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
            cache_key = _cond_cache_key(clip_name, tokens, unprojected, add_dict)
            try:
                hit = _cond_cache_load(cache_key)
            except Exception as exc:
                logging.warning("h3_cond_cache: load failed %s: %s", cache_key[:16], exc)
                hit = None
            if hit is not None:
                logging.info("h3_cond_cache: HIT  %s", cache_key[:16])
                return hit
            started = time.monotonic()
            _log_vram("clip:before-dispatch")
            placement.dispatch()
            dispatched = time.monotonic()
            _log_vram("clip:after-dispatch")
            torch.cuda.set_device(placement.devices[0])
            original_load_model = self.load_model
            original_load_device = self.patcher.load_device
            self.load_model = types.MethodType(lambda clip_self, tokens={}: clip_self.patcher, self)
            self.patcher.load_device = placement.devices[0]
            try:
                output = original_encode(tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar)
                encoded = time.monotonic()
                _log_vram("clip:after-encode")
                output = _move_tensors(output, cpu)
                output_copied = time.monotonic()
            finally:
                self.load_model = original_load_model
                self.patcher.load_device = original_load_device
                if offload_after_encode:
                    placement.offload()
                else:
                    placement.offload_embedding()
            finished = time.monotonic()
            logging.info(
                "H3 Qwen model-parallel timing: dispatch=%.3fs encode=%.3fs output_to_cpu=%.3fs offload=%.3fs total=%.3fs",
                dispatched - started,
                encoded - dispatched,
                output_copied - encoded,
                finished - output_copied,
                finished - started,
            )
            try:
                _cond_cache_save(cache_key, output)
                logging.info("h3_cond_cache: MISS %s (cached)", cache_key[:16])
            except Exception as exc:
                logging.warning("h3_cond_cache: save failed %s: %s", cache_key[:16], exc)
            return output

        clip.encode_from_tokens_scheduled = types.MethodType(encode_model_parallel, clip)
        clip.h3_model_parallel_placement = placement
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "H3MultiGPUCLIPLoader": H3MultiGPUCLIPLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MultiGPUCLIPLoader": "MiniMax H3 Multi-GPU CLIP Loader",
}
