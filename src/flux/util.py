import os
from dataclasses import dataclass

import torch
import json
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_sft

from optimum.quanto import requantize

from .model import Flux, FluxParams
from .controlnet import ControlNetFlux
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder
from .annotator.dwpose import DWposeDetector
from .annotator.mlsd import MLSDdetector
from .annotator.canny import CannyDetector
from .annotator.midas import MidasDetector
from .annotator.hed import HEDdetector
from .annotator.tile import TileDetector
from .annotator.zoe import ZoeDetector

import json

model_path_config = json.load('./model_path.json')

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
#Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

class Annotator:
    def __init__(self, name: str, device: str):
        if name == "canny":
            processor = CannyDetector()
        elif name == "openpose":
            processor = DWposeDetector(device)
        elif name == "depth":
            processor = MidasDetector()
        elif name == "hed":
            processor = HEDdetector()
        elif name == "hough":
            processor = MLSDdetector()
        elif name == "tile":
            processor = TileDetector()
        elif name == "zoe":
            processor = ZoeDetector()
        self.name = name
        self.processor = processor

    def __call__(self, image: Image, width: int, height: int):
        image = np.array(image)
        detect_resolution = max(width, height)
        image, remove_pad = resize_image_with_pad(image, detect_resolution)

        image = np.array(image)
        if self.name == "canny":
            result = self.processor(image, low_threshold=100, high_threshold=200)
        elif self.name == "hough":
            result = self.processor(image, thr_v=0.05, thr_d=5)
        elif self.name == "depth":
            result = self.processor(image)
            result, _ = result
        else:
            result = self.processor(image)

        result = HWC3(remove_pad(result))
        result = cv2.resize(result, (width, height))
        return result


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="XLabs-AI/flux-dev-fp8",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux-dev-fp8.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    with torch.device(device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(model_path_config['Flux'], device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return model

def load_flow_model2(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")

    with torch.device(device):
        model = Flux(configs[name].params)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(model_path_config['Flux'], device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return model

def load_flow_model_quintized(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    json_path = hf_hub_download(configs[name].repo_id, 'flux_dev_quantization_map.json')


    model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device='cpu')
    with open(json_path, "r") as f:
        quantization_map = json.load(f)
    print("Start a quantization process...")
    requantize(model, sd, quantization_map, device=device)
    print("Model is quantized!")
    return model

def load_controlnet(name, device, transformer=None):
    with torch.device(device):
        controlnet = ControlNetFlux(configs[name].params)
    if transformer is not None:
        controlnet.load_state_dict(transformer.state_dict(), strict=False)
    return controlnet

def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder(model_path_config, max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)

def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder(model_path_config, max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:

    # Loading the autoencoder
    print("Init AE")
    with torch.device(device):
        ae = AutoEncoder(configs[name].ae_params)

    sd = load_sft(model_path_config["AutoEncoder"], device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae