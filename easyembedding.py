import os
import sys
import torch
from torch import  nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import re
import gc
from safetensors.torch import load_file, save_file, safe_open
import folder_paths
import math
import random
import hashlib
from typing import List, Dict
import textwrap
import json
import time
import shutil
import scipy
from deep_translator import GoogleTranslator
from . import sampler_edit

import warnings
try:
    import torchvision.transforms.v2 as v2
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    print("v2 Apply transformation")
except ImportError:
    from torchvision import transforms
    transform = transforms.ToTensor()
    warnings.warn("torchvision v2 Module not found, replacing with ToTensor().")

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import comfy
import comfy.samplers
import comfy.sample
import comfy.model_management as model_management
import comfy.conds
import nodes
import node_helpers
import latent_preview
import comfy.cli_args
import comfy.clip_model
import comfy.clip_vision
import comfy.sd
import comfy.utils
import comfy.hooks
import comfy.context_windows
import comfy.bg_removal_model
from comfy_api.latest import IO, UI
from comfy_api.latest._io_public import ComfyTypeIO, comfytype, Custom

#----------------------------------------
#Header Utils
#----------------------------------------


#------------------------------------------------------
# embed & Projection & custom folder_in-out settings
#------------------------------------------------------

proj_dir = os.path.join(folder_paths.base_path, "models", "proj_embeddings")
os.makedirs(proj_dir, exist_ok=True)

folder_paths.folder_names_and_paths["proj_embeddings"] = ([proj_dir], folder_paths.supported_pt_extensions)


lora_dir = folder_paths.get_folder_paths("loras")

proj_dir = folder_paths.get_folder_paths("proj_embeddings")
clip_dir = folder_paths.get_folder_paths("clip")
embeddings_dir = folder_paths.get_folder_paths("embeddings")

font_dir = os.path.join(folder_paths.base_path, "models", "fonts")
os.makedirs(font_dir, exist_ok=True)

embedload = Custom("EMBEDS")
ProjL=Custom("projection_layer")
easytext = Custom("ESLogged")
    
@comfytype(io_type="EMBEDS")
class Embeds(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="projection_layer")
class ProjL(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="ESLogged")
class Logged(ComfyTypeIO):
    Type = str

def find_tensor(obj):
    """from dict/list, find torch.Tensor"""
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, dict):
        for v in obj.values():
            t = find_tensor(v)
            if t is not None:
                return t
    elif isinstance(obj, list):
        for item in obj:
            t = find_tensor(item)
            if t is not None:
                return t
    return None

    
def save_embedding_hidden_prompt(tensor, file_name, format="pt", save_dir=None):
    """
    file_name: "token name"
    format: "pt" or "safetensors"
    save_dir: "embbedings dir"
    """


    embedding_dirs = folder_paths.get_folder_paths("embeddings")
    save_dir = save_dir or (embedding_dirs[0] if embedding_dirs else ".")
    save_path = os.path.join(save_dir, f"{file_name}.{format}")

    if format == "pt":

        torch.save({"String_to_param": {file_name: tensor}}, save_path)
    elif format == "safetensors":

        metadata = {"ss_token": file_name}
        save_file({"embedding": tensor}, save_path, metadata=metadata)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return save_path

_counter = 0

_global_embed_counter = 0

def resolve_filename(prefix: str, save_dir: str) -> str:
    base = prefix.replace("%number", "")
    existing = [f for f in os.listdir(save_dir) if f.startswith(base)]
    number = len(existing) + 1
    return prefix.replace("%number", str(number))


def resolve_embfilename(prefix: str, save_dir: str) -> str:
    base = prefix.replace("%number", "")
    existing = [f for f in os.listdir(save_dir) if f.startswith(base) and f.endswith(".safetensors")]

    max_num = 0
    for f in existing:
        m = re.search(r"_(\d+)\.safetensors$", f)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num

    next_num = max_num + 1

    if "%number" in prefix:
        filename = prefix.replace("%number", str(next_num)) + ".safetensors"
    else:
        filename = f"{prefix}_{next_num}.safetensors"

    return filename

def embedding_to_conditioning(em_projected, condscale=1.0, neg_scale=1):
    if not torch.is_tensor(em_projected):
        raise TypeError("embedding_tensor는 torch.Tensor여야 합니다.")
    em_scaled = em_projected.to(dtype=torch.float32) * condscale
    
    if not isinstance(neg_scale, (int, float)):
        raise ValueError("neg_scale은 숫자여야 합니다.")
    if neg_scale <= 0:
        raise ValueError("neg_scale은 0보다 커야 합니다.")
    if neg_scale > 50:
        raise ValueError("neg_scale은 50 이하만 허용됩니다.")

    em_scaled = em_scaled/neg_scale

    if em_scaled.ndim == 2:
        em_scaled = em_scaled.unsqueeze(0)
    print("[Debug] em_projected:", type(em_projected), em_projected.shape if torch.is_tensor(em_projected) else em_projected)
    print("[Debug] em_scaled:", type(em_scaled), em_scaled.shape if torch.is_tensor(em_scaled) else em_scaled)
    shape_info = tuple(em_scaled.shape)
    return [[em_scaled, {}, shape_info]]

def prompt_to_embedding(clip_model, prompt_text, device="cpu"):

    if clip_model is None:
        raise ValueError("clip_model이 필요합니다.")
    if not prompt_text:
        raise ValueError("prompt_text가 필요합니다.")

    # ComfyUI CLIP
    if hasattr(clip_model, "encode_from_tokens_scheduled"):
        tokens = clip_model.tokenize(prompt_text)
        cond = clip_model.encode_from_tokens_scheduled(tokens)
        return cond[0][0].to(dtype=torch.float32, device="cpu")

    # HuggingFace/Diffusers CLIP
    if hasattr(clip_model, "__call__"):
        tokens = clip_model.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
        outputs = clip_model(**tokens)
        return outputs.last_hidden_state.to(dtype=torch.float32, device="cpu")

    raise RuntimeError(f"지원하지 않는 clip_model 타입: {type(clip_model)}")

#------------------------------------------------------
# image & Mask settings
#------------------------------------------------------

def ensure_mask_tensor(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, dict):
        if "noise_mask" in t:
            t = t["noise_mask"]
        elif "mask" in t:
            t = t["mask"]
        else:
            raise ValueError("Unsupported dict format for mask")
    if isinstance(t, (list, np.ndarray)):
        t = torch.as_tensor(np.asarray(t, dtype=np.float32))
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask type or shape: {t.shape}")
    return t.float()

def normalize_mask(mask: torch.Tensor, ref_tensor: torch.Tensor):

    if mask.ndim == 2:
        mask = mask.unsqueeze(-1).expand(-1, -1, ref_tensor.shape[-1])
        mask = mask.unsqueeze(0)

    elif mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, ref_tensor.shape[-1])
        mask = mask.unsqueeze(0)

    elif mask.ndim == 4:
        if mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, -1, ref_tensor.shape[-1])

    else:
        raise ValueError("Unsupported mask dimensions")

    return mask

def safe_to_torch(image):
    try:
        if isinstance(image, torch.Tensor):
            # torch.Tensor → float32 [0,1]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            return image.float().clamp(0.0, 1.0)

        elif isinstance(image, np.ndarray):
            # numpy → torch.Tensor
            arr = torch.from_numpy(image).float()
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr.unsqueeze(0) if arr.ndim == 3 else arr

        elif isinstance(image, Image.Image):
            # PIL.Image → torch.Tensor
            arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            arr = arr[None, ...]  # add batch
            return torch.from_numpy(arr)

        else:
            raise TypeError(f"Unsupported type: {type(image)}")

    except Exception as e:
        print("safe_to_torch conversion failed:", e)
        raise TypeError("Unsupported image type")

def to_tensor_output(tensor):

    # torch image
    if isinstance(tensor, torch.Tensor):
        # 2D → (1,1,H,W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        # 3D → (1,C,H,W)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        # 4D → pass
        elif tensor.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected tensor ndim: {tensor.ndim}")
        return tensor.float().clamp(0.0, 1.0)

    # numpy image
    elif isinstance(tensor, np.ndarray):
        arr = torch.from_numpy(tensor).float()
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = arr.unsqueeze(0).unsqueeze(0)
        elif arr.ndim == 3:
            arr = arr.unsqueeze(0)
        elif arr.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected numpy ndim: {arr.ndim}")
        return arr

    # pil image
    elif isinstance(tensor, Image.Image):
        arr = np.array(tensor.convert("RGB")).astype(np.float32) / 255.0
        arr = arr[None, ...]  # (1,H,W,C)
        return torch.from_numpy(arr)

    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")

def to_torch_image(image):
    """
    input->Torch float32 [0,1] tensor
    """
    if isinstance(image, torch.Tensor):
        return image.float().clamp(0.0, 1.0)
    elif isinstance(image, np.ndarray):
        arr = torch.from_numpy(image).float()
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr.unsqueeze(0) if arr.ndim == 3 else arr
    elif isinstance(image, Image.Image):
        arr = torch.from_numpy(np.array(image.convert("RGB"))).float() / 255.0
        return arr.unsqueeze(0)
    else:
        raise TypeError("Unsupported image type")

def to_tensor_image_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]  # add batch
    return torch.from_numpy(arr)

def to_numpy_image(image):
    if isinstance(image, torch.Tensor):
        arr = image[0].cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return arr
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        return image.astype(np.uint8)
    else:
        raise TypeError("Unsupported image type")

def resize_image(image_tensor, size):
    """
    image_tensor: torch.Tensor (batch, height, width, channels)
    size: (new_w, new_h) 
    """
    arr = safe_to_torch(image_tensor)
    new_w, new_h = size
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"image size error: {(new_w, new_h)}")
    resized = cv2.resize(to_numpy_image(arr), (new_w, new_h), interpolation=cv2.INTER_AREA)
    return to_tensor_output(Image.fromarray(resized))

def detect_face(pil_img, return_bbox=False):

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5)

    if return_bbox:
        return faces
    else:
        return len(faces) > 0

def gaussian_blur(tensor, kernel_size=5, sigma=2):
    k = kernel_size // 2
    x = torch.arange(-k, k+1, dtype=torch.float32)
    gauss = torch.exp(-(x**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
    kernel2d = kernel2d / kernel2d.sum()
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)

    tensor = tensor.unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(tensor, kernel2d, padding=k)
    return blurred.squeeze()

def dilate_mask(mask: torch.Tensor, iterations=1):
    """Mask dilation"""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        pass
    else:
        mask = mask.view(1, 1, mask.shape[-2], mask.shape[-1])

    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)
    for _ in range(iterations):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask > 0).float()

    return mask

def apply_mask_mode(mask: torch.Tensor, mode: str):
    """Mask_mode"""
    if mask is None:
        return None

    if mode == "normal":
        return mask.float().clamp(0.0, 1.0)
    elif mode == "small_spread":
        return dilate_mask(mask, iterations=1)
    elif mode == "big_spread":
        return dilate_mask(mask, iterations=3)
    elif mode == "blur":
        return gaussian_blur(mask.float(), kernel_size=5, sigma=2).clamp(0.0, 1.0)
    elif mode == "clear":
        emphasized = mask.float().clamp(0.0, 1.0) * 1.5
        return emphasized.clamp(0.0, 1.0)
    else:
        return mask.float().clamp(0.0, 1.0)

def get_mask_bbox(mask_tensor: torch.Tensor):

    m = mask_tensor.squeeze()
    coords = torch.nonzero(m > 0.5)

    if coords.numel() == 0:
        return None

    y_min, x_min = coords.min(dim=0)[0]
    y_max, x_max = coords.max(dim=0)[0]

    return int(x_min.item()), int(y_min.item()), int(x_max.item()), int(y_max.item())


def scale_bbox_to_latent(bbox, orig_size, latent_size=(64, 64)):

    h_lat, w_lat = latent_size
    H_orig, W_orig = orig_size

    if bbox is None:
        return (h_lat, w_lat, 0, 0)

    x_min, y_min, x_max, y_max = bbox

    x_min_lat = int(x_min * w_lat / W_orig)
    y_min_lat = int(y_min * h_lat / H_orig)
    x_max_lat = int(x_max * w_lat / W_orig)
    y_max_lat = int(y_max * h_lat / H_orig)

    x_min_lat = max(0, min(x_min_lat, w_lat - 1))
    y_min_lat = max(0, min(y_min_lat, h_lat - 1))
    x_max_lat = max(x_min_lat + 1, min(x_max_lat, w_lat))
    y_max_lat = max(y_min_lat + 1, min(y_max_lat, h_lat))

    # area = (height, width, start_y, start_x)
    area_height = y_max_lat - y_min_lat
    area_width = x_max_lat - x_min_lat

    return (max(1, area_height), max(1, area_width), y_min_lat, x_min_lat)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def ensure_rgb(image_np):
    if len(image_np.shape) == 2:  # [H,W]
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 1:  # [H,W,1]
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:
        return image_np

def color_transfer(base_np, target_np):
    base_rgb   = ensure_rgb(base_np)
    target_rgb = ensure_rgb(target_np)

    arr_lab  = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for i in range(3):
        arr_mean, arr_std   = arr_lab[:,:,i].mean(), arr_lab[:,:,i].std()
        base_mean, base_std = base_lab[:,:,i].mean(), base_lab[:,:,i].std()
        arr_lab[:,:,i] = (arr_lab[:,:,i] - arr_mean) * (base_std / (arr_std+1e-5)) + base_mean
    arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr_lab, cv2.COLOR_LAB2RGB)

def image_to_vector(image_arr, threshold=127):
    if len(image_arr.shape) == 3 and image_arr.shape[2] == 3:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    elif len(image_arr.shape) == 2:  # grayscale
        gray = image_arr
    else:
        gray = image_arr.astype(np.uint8)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def resize_vector(contours, new_width, new_height, orig_width, orig_height):
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    scaled_contours = []
    for cnt in contours:
        if cnt.shape[0] > 0:
            scaled = cnt.astype(np.float32) * [scale_x, scale_y]
            scaled = scaled.astype(np.int32)
            scaled_contours.append(scaled)
    return scaled_contours

def vector_to_image(contours, width, height, base_image=None, draw_color=(0,0,0), thickness=1):
    if base_image is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = base_image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            
    cv2.drawContours(canvas, contours, -1, draw_color, thickness, lineType=cv2.LINE_AA)
    return canvas

def vector_resize_mask(mask_np, new_w, new_h, resize_mode="pixelbox", threshold=127):
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "pixelbox": cv2.INTER_AREA
    }

    # 1. Mask Scale Safe Transform (If the range is 0.0–1.0, scale it to 0–255.)
    if mask_np.max() <= 1.0:
        mask_np_u8 = (mask_np * 255).astype(np.uint8)
    else:
        mask_np_u8 = mask_np.astype(np.uint8)

    base_resized = cv2.resize(mask_np_u8, (new_w, new_h), interpolation=interp_map[resize_mode])

    mask_bin = (mask_np_u8 > threshold).astype(np.uint8) * 255

    contours = image_to_vector(mask_bin, threshold)
    
    # 2. Lowered the area filter standard to preserve fine lines. (> 50 -> > 1)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1]

    if valid_contours:
        scaled_contours = resize_vector(valid_contours, new_w, new_h, mask_np.shape[1], mask_np.shape[0])
        mask_resized_np = vector_to_image(scaled_contours, new_w, new_h,
                                            base_image=base_resized,
                                            draw_color=(255, 255, 255),
                                            thickness=1)
    else:
        target_h, target_w = base_resized.shape[:2]
        edges = cv2.Canny(mask_bin, 50, 150, apertureSize=3)
        if target_w <= 128 or target_h <= 128:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.erode(edges, kernel, iterations=1)
        edgeline = cv2.resize(edges, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        edgeline = edgeline.astype(base_resized.dtype)

        if len(base_resized.shape) == 3 and base_resized.shape[2] == 3:
            edgeline_color = cv2.cvtColor(edgeline, cv2.COLOR_GRAY2BGR)
            mask_resized_np = cv2.bitwise_or(base_resized, edgeline_color)
        else:
            edgeline_color = edgeline
            mask_resized_np = cv2.bitwise_or(base_resized, edgeline_color)

    if mask_resized_np.ndim == 3 and mask_resized_np.shape[2] == 3:
        mask_resized_np = cv2.cvtColor(mask_resized_np, cv2.COLOR_BGR2GRAY)

    _, mask_resized_np = cv2.threshold(mask_resized_np, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    mask_resized_np = cv2.morphologyEx(mask_resized_np, cv2.MORPH_CLOSE, kernel)

    return mask_resized_np

def vector_resize_mask_3d_final(mask_tensor, target_d, target_h, target_w, resize_mode="pixelbox", is_video_model=False, threshold=127):
    if is_video_model:
        raise ValueError("\n[LatentMaskPrep Error] Video models are not supported.")

    if hasattr(mask_tensor, "cpu"):
        mask_np = mask_tensor.squeeze().cpu().numpy()
    else:
        mask_np = np.array(mask_tensor)

    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()

    mask_2d_resized = vector_resize_mask(mask_np, target_w, target_h, resize_mode=resize_mode, threshold=threshold)

    mask_2d_resized = ensure_mask_tensor(mask_2d_resized)
    resized_2d = F.avg_pool2d(mask_2d_resized, kernel_size=3, stride=1, padding=1)
    resized_3d = resized_2d.unsqueeze(2)

    return torch.clip(resized_3d, 0.0, 1.0)


def vector_resize_norm_mask(mask_np, new_w, new_h, resize_mode="pixelbox", threshold=127):
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "pixelbox": cv2.INTER_AREA
    }

    if mask_np.max() <= 1.0:
        mask_np_u8 = (mask_np * 255).astype(np.uint8)
    else:
        mask_np_u8 = mask_np.astype(np.uint8)

    base_resized = cv2.resize(mask_np_u8, (new_w, new_h), interpolation=interp_map[resize_mode])

    mask_bin = (mask_np_u8 > threshold).astype(np.uint8) * 255

    contours = image_to_vector(mask_bin)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

    if valid_contours:
        scaled_contours = resize_vector(valid_contours, new_w, new_h, mask_np.shape[1], mask_np.shape[0])
        mask_resized_np = vector_to_image(scaled_contours, new_w, new_h,
                                          base_image=base_resized,
                                          draw_color=(255,255,255),
                                          thickness=1)
    else:
        # fallback
        edges = cv2.Canny(mask_bin, 50, 150)
        edgeline = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        edgeline_color = cv2.cvtColor(edgeline, cv2.COLOR_GRAY2BGR)
        mask_resized_np = cv2.bitwise_or(base_resized, edgeline_color)

    return mask_resized_np

def progressbar_update(pbar, step, total_steps):
    pbar.update(step+1)


#------------------------------------------------------
# prompt settings
#------------------------------------------------------
def normalize_and_encode(clip, text, weight, device):

    if not text or not str(text).strip():
        model = clip.patcher.model
    
        if hasattr(model, 'embed_tokens'):
            clip_dim = model.embed_tokens.weight.shape[1]
        elif hasattr(model, 'token_embedding'):
            clip_dim = model.token_embedding.weight.shape[1]
        elif hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            clip_dim = model.config.hidden_size
        return [[torch.zeros((1, 77, clip_dim), device=device), {"pooled_output": torch.zeros((1, 1280), device=device), "weight": 0}]]

    tokens = clip.tokenize(text)
    
    cond = clip.encode_from_tokens_scheduled(tokens)

    cond_scaled = []
    for item in cond:
        tensor = item[0]
        cond_dict = item[1].copy() if len(item) > 1 else {}
        
        scaled_tensor = tensor * weight
        
        cond_dict["weight"] = weight

        if "pooled_output" not in cond_dict or cond_dict["pooled_output"] is None:
            cond_dict["pooled_output"] = torch.zeros((1, 1280), device=device)
            
        cond_scaled.append([scaled_tensor, cond_dict])
    
    return cond_scaled

def weighted_concat(base_cond, new_cond):
    # cond concat : base_cond, new_cond
    tensor1, dict1 = base_cond[0]
    tensor2, dict2 = new_cond[0]
    
    w1, w2 = dict1.get("weight", 1.0), dict2.get("weight", 1.0)
    
    concat_tensor = torch.cat([tensor1, tensor2], dim=1)
    
    p1, p2 = dict1.get("pooled_output"), dict2.get("pooled_output")
    new_pooled = (p1 * w1 + p2 * w2) / (w1 + w2) if (p1 is not None and p2 is not None) else (p1 or p2)
    
    return [[concat_tensor, {"pooled_output": new_pooled, "weight": w1 + w2}]]

def apply_conditioning_combine(cond_1, cond_2):

    return cond_1 + cond_2

def apply_conditioning_concat(cond_to, cond_from):

    out = []
    cond_from_tensor = cond_from[0][0]
    for i in range(len(cond_to)):
        t1 = cond_to[i][0]
        if cond_from_tensor.shape[1] < t1.shape[1]:
            pad_len = t1.shape[1] - cond_from_tensor.shape[1]
            padding = torch.zeros((1, pad_len, t1.shape[2]),
                                  device=t1.device, dtype=t1.dtype)
            cond_from_tensor_padded = torch.cat([cond_from_tensor, padding], dim=1)
        else:
            cond_from_tensor_padded = cond_from_tensor[:, :t1.shape[1]]
        tw = torch.cat((t1, cond_from_tensor_padded), 1)
        out.append([tw, cond_to[i][1].copy()])
    return out

def apply_conditioning_average(cond_to, cond_from, strength, device, dtype):

    out = []
    cond_from_tensor = cond_from[0][0]
    pooled_from = cond_from[0][1].get("pooled_output", None)

    for i in range(len(cond_to)):
        t1 = cond_to[i][0]
        pooled_to = cond_to[i][1].get("pooled_output", pooled_from)

        t0 = cond_from_tensor[:, :t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            padding = torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]), device=device, dtype=dtype)
            t0 = torch.cat([t0, padding], dim=1)

        tw = torch.mul(t1, strength) + torch.mul(t0, (1.0 - strength))
        t_to = cond_to[i][1].copy()
        if pooled_from is not None and pooled_to is not None:
            t_to["pooled_output"] = torch.mul(pooled_to, strength) + torch.mul(pooled_from, (1.0 - strength))
        
        out.append([tw, t_to])
    return out

def process_fields(clip, cond_list, manual_mode, strength=1.0):

    valid_conds = [c for c in cond_list if c is not None and isinstance(c, list) and len(c) > 0]
    
    if not valid_conds:
        return None

    current_cond = valid_conds[0]

    device = current_cond[0][0].device
    dtype = current_cond[0][0].dtype
    
    for next_cond in valid_conds[1:]:
        if manual_mode == "concat":
            current_cond = apply_conditioning_concat(current_cond, next_cond)
            
        elif manual_mode == "combine":
            current_cond = apply_conditioning_combine(current_cond, next_cond)
            
        elif manual_mode == "average":
            current_cond = apply_conditioning_average(current_cond, next_cond, strength, device, dtype)
            
    return current_cond

def zip_and_combine_conds(conds: list, new_conds: list):
    combined_conds = []
    if conds:
        combined_conds = apply_conditioning_combine(combined_conds, conds)
    if new_conds:
        combined_conds = apply_conditioning_combine(combined_conds, new_conds)
    return combined_conds

def zip_and_combine_conds_multi(conds: list, new_conds1: list, new_conds2: list):
    combined_conds = []
    if conds:
        combined_conds = apply_conditioning_combine(combined_conds, conds)
    if new_conds1:
        combined_conds = apply_conditioning_combine(combined_conds, new_conds1)
    if new_conds2:
        combined_conds = apply_conditioning_combine(combined_conds, new_conds2)
    return combined_conds

#----------------------------------------
#custom translator settings
#----------------------------------------
BASE_DIR = os.path.dirname(__file__)
TRANS_DIR = os.path.join(BASE_DIR, "translatetext")
os.makedirs(TRANS_DIR, exist_ok=True)

def get_trans_files():
    if not os.path.exists(TRANS_DIR):
        return []
    return [f for f in os.listdir(TRANS_DIR) if f.endswith(".txt")]

def parse_multi_language_glossary(content):
    # (basekey) \n - \n (lang) \n - \n (translate)
    pattern = r"\((.*?)\)\n-\n(.*?)\n-\n\((.*?)\)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    # dictionary output: { "basekey": { "en": "translate", "ja": "translate" } }
    glossary = {}
    for original, lang, translation in matches:
        if original not in glossary:
            glossary[original] = {}
        glossary[original][lang.strip()] = translation.strip()
    return glossary
#----------------------------------------
#custom sampler settings
#----------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"

def get_safe_conditioning(cond, device, model, cfg):
    if cond is None or (isinstance(cond, list) and len(cond) == 0):
        cond = model.cond_stage_model.get_empty_conditioning()

    updated_cond = node_helpers.conditioning_set_values(cond, {"strength": cfg})

    return updated_cond

def get_safe_conditioning_hooked(cond, device, model, cfg):
    if cond is None or (isinstance(cond, list) and len(cond) == 0):
        cond = model.cond_stage_model.get_empty_conditioning()

    updated_cond = node_helpers.conditioning_set_values(cond, {"strength": cfg})

    target_custom_keys = [
        "reference_weight", "palette_mode", "ref_lat_style_type", 
        "ref_lat_transfer_str", "ref_latent_features", "palette_features", 
        "style_type", "style_str", "layernum", "layers"
    ]

    combined_hooks = comfy.hooks.HookGroup()
    extracted_masks = []
    
    for item in updated_cond:
        c_dict = item[1]

        if "hooks" in c_dict and c_dict["hooks"] is not None:
            combined_hooks = combined_hooks.clone_and_combine(c_dict["hooks"])

        found_custom_data = {k: c_dict[k] for k in target_custom_keys if k in c_dict}
        if found_custom_data:
            custom_hook = comfy.hooks.Hook()
            if not hasattr(custom_hook, "conditioning_modifiers"):
                custom_hook.conditioning_modifiers = {}
            
            custom_hook.conditioning_modifiers.update(found_custom_data)
            
            temp_group = comfy.hooks.HookGroup()
            temp_group.hooks = [custom_hook]
            combined_hooks = combined_hooks.clone_and_combine(temp_group)

        if "mask" in item[1] and item[1]["mask"] is not None:
            mask = item[1]["mask"]
            strength = c_dict.get("mask_strength", 1.0)
            if strength >= 0.6: 
                extracted_masks.append(c_dict["mask"])
                extracted_masks.append(mask)

        if "concat_mask" in item[1] and item[1]["concat_mask"] is not None:
            concat_mask = item[1]["concat_mask"]
            extracted_masks.append(concat_mask)

    hooks = combined_hooks if len(combined_hooks) > 0 else None

    return updated_cond, hooks, extracted_masks

def prepare_noise_inner_safe(latent_image, generator, device, noise_inds=None):
    if noise_inds is None:
        return torch.randn(latent_image.size(), generator=generator, dtype=latent_image.dtype,
                           layout=latent_image.layout, device=device)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], generator=generator,
                                       dtype=latent_image.dtype, layout=latent_image.layout, device=device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    return torch.cat(noises, axis=0)

def prepare_noise_safe(latent_image, generator, device, noise_inds=None):

    if getattr(latent_image, "is_nested", False):
        tensors = latent_image.unbind()
        noises = [prepare_noise_inner_safe(t, generator, device, noise_inds) for t in tensors]
        return comfy.nested_tensor.NestedTensor(noises)
    else:
        return prepare_noise_inner_safe(latent_image, generator, device, noise_inds)

def parse_seed(seed_str: str) -> int:
    MAX_SEED = 2**63 - 1


    if not seed_str:
        return 0

    try:
        seed_val = int(seed_str)
        if 0 <= seed_val <= MAX_SEED:
            return seed_val
        if seed_val < 0:
            print(f"\n[Notice] The seed value ({seed_val}) is negative. It will be converted or clamped.")
        else:
            print(f"\n[Notice] The seed value exceeds MAX_SEED. It will be mapped via hash.")
            
        seed_str = str(seed_val)
        seed_str = str(seed_val)
    except (ValueError, TypeError):
        seed_str = str(seed_str)

    print(f"\n [Notice] The seed value is not a number. It will be replaced with the text hash value: '{seed_str}'")
    hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed_val =  max(0, min(int.from_bytes(hash_bytes[:8], "big"), MAX_SEED))
    return seed_val

def normalized_latentnoise(noise):
    # 1. std check
    noise_std = noise.std(dim=tuple(range(1, noise.dim())), keepdim=True) + 1e-6
    # 2. normalize_noise
    norm_noise = noise/ noise_std
    return norm_noise

def safe_rescale_latent(processed_latent, max_std=1.0):
    # 1. std check
    dims = list(range(1, processed_latent.ndim))
    current_std = torch.std(processed_latent, dim=dims, keepdim=True)
    current_std = torch.clamp(current_std, min=1e-6)

    scale_factor = max_std / current_std
    norm_noise = processed_latent * scale_factor

    return norm_noise

def match_minstd(source, target):
    dims = tuple(range(1, source.dim()))
    target_mean = target.mean(dim=dims, keepdim=True)
    target_std = target.std(dim=dims, keepdim=True) + 1e-6

    source_mean = source.mean(dim=dims, keepdim=True)
    source_std = source.std(dim=dims, keepdim=True) + 1e-6

    normalized_source = (source - source_mean) / source_std
    return (normalized_source * target_std) + target_mean

def get_processed_noise(noise, normalize_noise):

    if normalize_noise == "none":
        return noise
    elif normalize_noise == "soft":
        max_std = 1
        noise = safe_rescale_latent(noise, max_std)
    elif normalize_noise == "standard":
        noise = normalized_latentnoise(noise)
    return noise

def calculate_custom_latent_weights(s_val, e_val, num_steps, latent_mode="linear", device="cpu"):
    """
    options=["linear", "ease_in", "EASE_OUT", "EASE_IN_OUT"]
    """
    weights = []
    for i in range(num_steps):
        progress = i / max(num_steps - 1, 1)
        
        if latent_mode == 'ease_in':
            factor = 0.05 + 0.95 * (1 - progress)
        elif latent_mode == 'EASE_OUT':
            factor = 0.05 + 0.95 * progress
        elif latent_mode == 'EASE_IN_OUT':
            factor = 0.05 + 0.95 * (1 - abs(progress - 0.5) / 0.5)
        else:  # 'linear'
            factor = progress ** 0.8

        factor = min(max(factor, 0.0), 1.0)
        val = s_val + (e_val - s_val) * factor
        weights.append(val)
        
    return torch.tensor(weights, dtype=torch.float32, device=device)

def get_current_weight(step_idx, transfer_str, w):
    if hasattr(transfer_str, "__getitem__") and len(transfer_str) > step_idx:
        return transfer_str[step_idx]
    return w

def get_style_transfer_hooks(transfer_str, style_type, palette_features, w=1.0, latent_shape=None, latent_dim=None):
    if style_type in ("Pass", None) or palette_features is None:
        return lambda q,k,v,extra: (q,k,v)

    mu_p = palette_features.get("mean")
    sigma_p = palette_features.get("std")
    edges = palette_features.get("edges")
    colors = palette_features.get("colors")
    brightness = palette_features.get("brightness")

    stat_dims = (3, 4) if latent_dim == 5 else (2, 3)

    def apply_style_hook(q, k, v, extra_options):
        step_idx = extra_options.get("transformer_options", {}).get("current_step", 0)
        
        if hasattr(transfer_str, "__getitem__") and len(transfer_str) > step_idx:
            curr_val = transfer_str[step_idx]
            current_weight = curr_val.item() if isinstance(curr_val, torch.Tensor) else curr_val
        else:
            current_weight = w

        effective_weight = current_weight * w
        if effective_weight <= 0.0:
            return q, k, v

        if style_type == "Edge" and edges is not None:
            return q * (1.0 + effective_weight * edges), k, v
        elif style_type == "Color" and colors is not None:
            return q, k, v + effective_weight * colors
        elif style_type == "Brightness" and brightness is not None:
            return q, k, v * (1.0 + effective_weight * (brightness - 1.0))
        elif style_type == "AdaIN" and mu_p is not None and sigma_p is not None:
            mu_curr = v.mean(dim=stat_dims, keepdim=True)
            sigma_curr = v.std(dim=stat_dims, keepdim=True) + 1e-6
            v_norm = (v - mu_curr) / sigma_curr
            v_ada = v_norm * (sigma_p * effective_weight + sigma_curr * (1-effective_weight)) + \
                    (mu_p * effective_weight + mu_curr * (1-effective_weight))
            return q, k, v_ada
            
        return q, k, v

    return apply_style_hook

def build_style_transfer_model_options(img_guide_options, sigmas, extra_options=None, latent_shape=None, noise_mask=None, latent_dim=None):
    palette_features = img_guide_options.get("palette_features", None)
    style_type = img_guide_options.get("style_type", "Pass")
    style_str = img_guide_options.get("style_str", 1.0)

    device=sigmas.device
    max_val = float(ref_weight) * float(transfer_str)
    s_val = 0.00
    e_val = float(max_val)
    method = "linear"  
    weights = calculate_custom_latent_weights(s_val, e_val, len(sigmas), method, device)
    weight_map = weights.to(device=sigmas.device, dtype=torch.float32)

    style_hook_img = get_style_transfer_hooks(weight_map, style_type, palette_features, 1.0, latent_shape=latent_shape, latent_dim=latent_dim)

    def style_patch(args):
        sigma = args["sigma"]
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().item()

        idx = torch.argmin(torch.abs(sigmas - sigma)).item()
        idx = min(max(idx, 0), len(weight_map) - 1)

        current_weight = weight_map[idx]
        if isinstance(current_weight, torch.Tensor):
            current_weight = current_weight.item()

        cond_out = args["cond_out"]
        
        if style_hook_img is not None and current_weight > 0.0 and style_type != "Pass":
            if isinstance(cond_out, (list, tuple)) and len(cond_out) == 3:
                q, k, v = cond_out
            else:
                q = k = v = cond_out

            q_out, k_out, v_styled = style_hook_img(q, k, v, {"transformer_options": {"current_step": idx}})
            
            if isinstance(cond_out, (list, tuple)):
                v_final = v * (1.0 - current_weight) + v_styled * current_weight
                return (q_out, k_out, v_final)
            else:
                return cond_out * (1.0 - current_weight) + v_styled * current_weight
        
        if isinstance(cond_out, (list, tuple)) and len(cond_out) == 3:
            q, k, v = cond_out
            return (q, k, v * current_weight)
        return cond_out * current_weight

    is_flow_model = (latent_dim == 5) or (extra_options and extra_options.get("is_flow", False))

    layers_dict = img_guide_options.get("layers", {}) # {"q": "...", "k": "...", "v": "..."}
    layernum = img_guide_options.get("layernum", 0)
    
    patches = {}

    if layers_dict and layernum > 0:
        for i in range(layernum):
            for target_type, base_name in layers_dict.items():
                if not base_name:
                    continue

                dynamic_key = base_name
                if "layers." in dynamic_key:
                    parts = dynamic_key.split("layers.")
                    sub_parts = parts[1].split(".")
                    sub_parts[0] = str(i) # Index replacement
                    dynamic_key = parts[0] + "layers." + ".".join(sub_parts)

                patches[dynamic_key] = style_patch
                
    else:
        # Fall-back
        if is_flow_model:
            patches["self_attn.q_proj"] = style_patch
            patches["self_attn.k_proj"] = style_patch
            patches["self_attn.v_proj"] = style_patch

            style_q = style_patch
            style_k = style_patch
            style_v = style_patch

    
        else:
            patches["middle_block"] = [style_patch]

    model_options = {
        "transformer_options": {
            "patches": patches
        }
    }

    
    if extra_options:
        model_options["transformer_options"].update(extra_options)
        
    return model_options

def get_latent_transfer_hooks(transfer_str, style_type, ref_latent_features, weight_map, w=1.0, latent_shape=None, latent_dim=None):
    stat_dims = (3, 4) if latent_dim == 5 else (2, 3)

    def apply_latent_hook(q, k, v, extra_options):
        if style_type == "Pass" or style_type is None:
            return q, k, v

        step_idx = extra_options.get("transformer_options", {}).get("current_step", 0)
        
        if weight_map is not None and len(weight_map) > step_idx:
            current_weight = weight_map[step_idx]
            if isinstance(current_weight, torch.Tensor):
                current_weight = current_weight.item()
        else:
            current_weight = transfer_str[step_idx] if hasattr(transfer_str, "__getitem__") and len(transfer_str) > step_idx else w

        effective_weight = current_weight * w
        if effective_weight <= 0.0:
            return q, k, v

        v_target = v 
        v_styled = v_target

        if style_type == "Latent_AdaIN":
            mu_ref = ref_latent_features.get("mean", None)
            sigma_ref = ref_latent_features.get("std", None)
            if mu_ref is not None and sigma_ref is not None:
                mu_curr = v_target.mean(dim=stat_dims, keepdim=True)
                sigma_curr = v_target.std(dim=stat_dims, keepdim=True) + 1e-6
                v_norm = (v_target - mu_curr) / sigma_curr
                v_styled = v_norm * (sigma_ref * effective_weight + sigma_curr * (1 - effective_weight)) + \
                           (mu_ref * effective_weight + mu_curr * (1 - effective_weight))

        elif style_type == "Spatial_Attention":
            ref_v = ref_latent_features.get("v_feat", None)
            if ref_v is not None:
                if ref_v.shape[2:] != v_target.shape[2:]:
                    interp_mode = "trilinear" if latent_dim == 5 else "bilinear"
                    ref_v = F.interpolate(ref_v, size=v_target.shape[2:], mode=interp_mode, align_corners=False if interp_mode=="bilinear" else None)
                spatial_mask = torch.sigmoid(ref_v.mean(dim=1, keepdim=True))
                v_styled = v_target * (1.0 - spatial_mask * effective_weight) + ref_v * (spatial_mask * effective_weight)

        elif style_type == "Residual_Blending":
            ref_latent_base = ref_latent_features.get("base", None)
            if ref_latent_base is not None:
                if ref_latent_base.shape[2:] != v_target.shape[2:]:
                    interp_mode = "trilinear" if latent_dim == 5 else "bilinear"
                    ref_latent_base = F.interpolate(ref_latent_base, size=v_target.shape[2:], mode=interp_mode, align_corners=False if interp_mode=="bilinear" else None)
                v_styled = v_target + (ref_latent_base - v_target) * effective_weight

        elif style_type == "Channel_Selective":
            ref_latent_base = ref_latent_features.get("base", None)
            if ref_latent_base is not None and v_target.shape[1] >= 4:
                v_styled = v_target.clone()
                struct_weight = effective_weight
                if latent_dim == 5:
                    v_styled[:, :2, :, :, :] = v_styled[:, :2, :, :, :] * (1.0 - struct_weight) + ref_latent_base[:, :2, :, :, :] * struct_weight
                else:
                    v_styled[:, :2, :, :] = v_styled[:, :2, :, :] * (1.0 - struct_weight) + ref_latent_base[:, :2, :, :] * struct_weight

        return q, k, v_styled

    return apply_latent_hook

def build_style_transfer_lat_model_options(latent_guide_options, sigmas, extra_options=None, noise_mask=None, latent_shape=None, latent_dim=None):
    ref_weight = latent_guide_options.get("reference_weight", 1.00)
    method = latent_guide_options.get("palette_mode", "ease_in_out")
    style_type = latent_guide_options.get("ref_lat_style_type", "Pass")
    latent_features = latent_guide_options.get("ref_latent_features", None)
    transfer_str = latent_guide_options.get("ref_lat_transfer_str", 1.0)

    max_val = float(ref_weight) * float(transfer_str)
    device=sigmas.device
    s_val = 0.00
    e_val = float(max_val)
    weights = calculate_custom_latent_weights(s_val, e_val, len(sigmas), method, device)
    weight_map = weights.to(device=sigmas.device, dtype=torch.float32)

    style_hook_fn = get_latent_transfer_hooks(transfer_str, style_type, latent_features, weight_map, 1.0, latent_shape=latent_shape, latent_dim=latent_dim)
    
    def style_patch(args):
        sigma = args["sigma"]
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().item()
        idx = torch.argmin(torch.abs(sigmas - sigma)).item()
        idx = min(max(idx, 0), len(weight_map) - 1)
        
        cond_out = args["cond_out"]
        
        if style_hook_fn is not None:
            if isinstance(cond_out, (list, tuple)) and len(cond_out) == 3:
                q, k, v = cond_out
            else:
                q = k = v = cond_out

            q_out, k_out, v_out = style_hook_fn(q, k, v, {"transformer_options": {"current_step": idx}})
            
            if isinstance(cond_out, (list, tuple)):
                return (q_out, k_out, v_out)
            else:
                return v_out
                
        return cond_out

    is_flow_model = (latent_dim == 5) or (extra_options and extra_options.get("is_flow", False))

    layers_dict = latent_guide_options.get("layers", {}) # {"q": "...", "k": "...", "v": "..."}
    layernum = latent_guide_options.get("layernum", 0)
    
    patches = {}

    if layers_dict and layernum > 0:
        for i in range(layernum):
            for target_type, base_name in layers_dict.items():
                if not base_name:
                    continue

                dynamic_key = base_name
                if "layers." in dynamic_key:
                    parts = dynamic_key.split("layers.")
                    sub_parts = parts[1].split(".")
                    sub_parts[0] = str(i)  # Index replacement
                    dynamic_key = parts[0] + "layers." + ".".join(sub_parts)

                patches[dynamic_key] = style_patch
    else:
        if is_flow_model:
            patches["self_attn.q_proj"] = style_patch
            patches["self_attn.k_proj"] = style_patch
            patches["self_attn.v_proj"] = style_patch
        else:
            patches["middle_block"] = [style_patch]

    model_options = {
        "transformer_options": {
            "patches": patches
        }
    }
    
    if extra_options and isinstance(extra_options, dict):
        model_options["transformer_options"].update(extra_options)

    return model_options

def build_model_options(sigmas, pos_hook=None, neg_hook=None, noise_mask=None, extra_options=None, latent_shape=None, latent_dim=None):
    model_options = {"hooks": {}, "transformer_options": {}}
    
    if pos_hook: 
        model_options["hooks"]["pos_conditioning"] = pos_hook
    if neg_hook: 
        model_options["hooks"]["neg_conditioning"] = neg_hook

    latent_guide_options = {}
    img_guide_options = {}

    if pos_hook is not None:
        pos_dict = getattr(pos_hook, "conditioning_modifiers", {}) if hasattr(pos_hook, "conditioning_modifiers") else {}
        if not pos_dict and isinstance(pos_hook, dict):
            pos_dict = pos_hook
            
        for key in ["reference_weight", "palette_mode", "ref_lat_style_type", "ref_lat_transfer_str", "ref_latent_features", "layernum", "layers"]:
            if key in pos_dict:
                latent_guide_options[key] = pos_dict[key]

        for key in ["palette_features", "style_type", "style_str", "layernum", "layers"]:
            if key in pos_dict:
                img_guide_options[key] = pos_dict[key]

    if neg_hook is not None:
        neg_dict = getattr(neg_hook, "conditioning_modifiers", {}) if hasattr(neg_hook, "conditioning_modifiers") else {}
        if not neg_dict and isinstance(neg_hook, dict):
            neg_dict = neg_hook
            
        if "reference_latents" in neg_dict:
            if latent_guide_options:
                print(f"\033[33m[Warning]\033[0m 네거티브 컨디셔닝에서도 라텐트 가이드가 감지되었습니다. 안전을 위해 포지티브 기준 가이드만 적용됩니다.")
            else:
                print(f"\033[31m[Warning]\033[0m 포지티브가 아닌 네거티브에 라텐트 가이드가 연결되었습니다. 의도치 않은 결과가 나올 수 있습니다.")
                for key in ["reference_weight", "palette_mode", "ref_lat_style_type", "ref_lat_transfer_str", "ref_latent_features", "layernum", "layers"]:
                    if key in neg_dict:
                        latent_guide_options[key] = neg_dict[key]
            
        if "palette_features" in neg_dict:
            if img_guide_options:
                print(f"\033[33m[Warning]\033[0m 네거티브 컨디셔닝에서도 이미지 가이드가 감지되었습니다. 안전을 위해 포지티브 기준 가이드만 적용됩니다.")
            else:
                print(f"\033[31m[Warning]\033[0m 포지티브가 아닌 네거티브에 이미지 가이드가 연결되었습니다. 의도치 않은 결과가 나올 수 있습니다.")
                for key in ["palette_features", "style_type", "style_str", "layernum", "layers"]:
                    if key in neg_dict:
                        img_guide_options[key] = neg_dict[key]

    # ----------------------------------------------------
    # Final data injection
    # ----------------------------------------------------
    if latent_guide_options:
        lat_model_opts = build_style_transfer_lat_model_options(latent_guide_options, sigmas, extra_options, noise_mask, latent_shape=None, latent_dim=None)
        if "transformer_options" in lat_model_opts:
            model_options["transformer_options"].update(lat_model_opts["transformer_options"])

    if img_guide_options:
        if latent_guide_options:
            print(f"\033[33m[Warning]\033[0m 라텐트 가이드와 이미지 가이드가 동시에 감지되었습니다. 안전을 위해 라텐트 가이드만 적용됩니다.")
            pass
        else:
            img_model_opts = build_style_transfer_model_options(img_guide_options, sigmas, extra_options, noise_mask, latent_shape=None, latent_dim=None)
            if "transformer_options" in img_model_opts:
                model_options["transformer_options"].update(img_model_opts["transformer_options"])

    if extra_options and isinstance(extra_options, dict):
        model_options["transformer_options"].update(extra_options)
        
    return model_options

def get_sigmas(model, scheduler, steps, denoise, device="cpu"):

    model_sampling = model.get_model_object("model_sampling")
    
    if denoise < 1.0:
        new_steps = int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, new_steps).to(device)
        return sigmas[-(steps + 1):]
    
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).to(device)
    return sigmas

def erode_mtensor(mask_tensor, kernel_size=5, sigma=2, iterations=1, tapered_corners=True):
    inv = 1.0 - mask_tensor
    if tapered_corners:
        for _ in range(iterations):
            blurred = gaussian_blur(inv, kernel_size=kernel_size, sigma=sigma)
            inv = (blurred > 0.3).float()
    else:
        stride=1
        for _ in range(iterations):
            inv = F.max_pool2d(inv, kernel_size, stride=1, padding=kernel_size//2)
    return 1.0 - inv

def prepare_noise_mask(mask_list, threshold=0.6, target_size=None, get_shrinkmask=True, findkey=False):
    if not mask_list:
        return None
    valid_masks = [m for m in mask_list if m.max() >= threshold]
    
    if not valid_masks:
        return None
    cond_mask = torch.max(torch.stack(valid_masks), dim=0).values

    if findkey:
        return None
    if get_shrinkmask:
        c_mask = erode_mtensor(ensure_mask_tensor(cond_mask), kernel_size=3, iterations=1, tapered_corners=False)
    else:
        c_mask = ensure_mask_tensor(cond_mask)

    if target_size is not None:
        noise_mask = F.interpolate(c_mask, size=target_size, mode='bilinear')
        return noise_mask.clamp(0.0, 1.0)
    
    return c_mask.clamp(0.0, 1.0)

def get_layer_count(clip):
    try:
        patcher = getattr(clip, "patcher", None)
        model_weights = getattr(patcher, "model", None) if patcher else None
        target_keys = None
        
        if hasattr(model_weights, "state_dict"):
            target_keys = model_weights.state_dict().keys()
        elif hasattr(clip, "cond_stage_model"):
            target_keys = dict(clip.cond_stage_model.named_parameters()).keys()
            
        if target_keys:
            layer_indices = set()
            found_attn_keys = {}

            targets = {
                "q": ["q_proj", "query", "to_q", "_q"],
                "k": ["k_proj", "key", "to_k", "_k"],
                "v": ["v_proj", "value", "to_v", "_v"]
            }
            
            for name in target_keys:
                name_lower = name.lower()
                
                if "layers." in name:
                    try:
                        idx = int(name.split("layers.")[1].split(".")[0])
                        layer_indices.add(idx)
                    except ValueError:
                        pass

                for target_key, keywords in targets.items():
                    if target_key not in found_attn_keys:
                        for kw in keywords:
                            if kw in name_lower:
                                found_attn_keys[target_key] = name
                                break
                                
            layer_count = len(layer_indices) if layer_indices else 0
            return layer_count, found_attn_keys
            
    except Exception as e:
        print(f"[get_layer_and_attn_keys] failed: {e}")
        
    return 0, {}

#----------------------------------------
# Textbox Node Settings
#----------------------------------------
bubblelayout= Custom("b_layout")

@comfytype(io_type="b_layout")
class bubblelayout(ComfyTypeIO):
    Type = torch.Tensor

BASE_DIR = os.path.dirname(__file__)
b_layout_dir = os.path.join(BASE_DIR, "bubble_layout")

COLOR_MAP = {
    "A": (255, 0, 0),   # R → bubble outer color
    "B": (0, 255, 0),   # G →bubble lines
    "C": (0, 0, 255),   # B →bubble inner
}

def get_bubble_layout_files():
    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(b_layout_dir):
        for f in os.listdir(b_layout_dir):
            full_path = os.path.join(b_layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)
    return files

def load_bubble_layout(filename):
    ref_path = os.path.join(b_layout_dir, filename)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Bubble layout not found: {ref_path}")
    
    img = Image.open(ref_path).convert("RGB")
    return to_torch_image(img)

#----------------------------------------
#Projection Node Settings
#----------------------------------------

class EasyProjLayerLoad(IO.ComfyNode):
    @classmethod
    def define_schema(cls):

        files = []
        base_dirs = folder_paths.get_folder_paths("proj_embeddings")
        for d in base_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        
                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)


        return IO.Schema(
            node_id="EasyProjLayerLoad",
            display_name="프로젝션 레이어 로더",
            category="커스텀임베딩/임베드",
            description="저장된 임베딩 보정치 파일을 가져옵니다.",
            inputs=[
                IO.Combo.Input(
                    "proj_layername",
                    options=files,
                    default=files[0] if files else "",
                    tooltip="불러올 proj레이어 파일을 선택하세요"
                ),
            ],
            outputs=[
                ProjL.Output("projection_layer", tooltip="프로젝션 레이어 출력"),
                IO.Int.Output("in_dim", tooltip="저장된 입력 차원"),
                IO.Int.Output("out_dim", tooltip="저장된 출력 차원"),
                IO.Combo.Output(
                    "clip_type",
                    options=[
                        "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
                        "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream",
                        "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image",
                        "flux2", "ovis"
                    ],
                    tooltip="저장된 모델 유형"
                )
            ],
        )

    @classmethod
    def execute(cls, proj_layername) -> IO.NodeOutput:

        base_dirs = folder_paths.get_folder_paths("proj_embeddings")
        filename = None
        for d in base_dirs:
            candidate = os.path.join(d, proj_layername)
            if os.path.exists(candidate):
                filename = candidate
                break

        if filename is None:
            raise FileNotFoundError(f"{proj_layername}.safetensors not found in proj_embeddings paths")

        data = load_file(filename)

        with safe_open(filename, framework="pt", device="cpu") as f:
            metadata = f.metadata()

        weight = data.get("weight", None)
        bias = data.get("bias", None)

        try:
            in_dim = int(metadata.get("meta_in_dim"))
        except Exception:
            in_dim = weight.shape[1] if weight is not None else 0

        try:
            out_dim = int(metadata.get("meta_out_dim"))
        except Exception:
            out_dim = weight.shape[0] if weight is not None else 0

        clip_type = metadata.get("meta_type", "unknown")

        proj_layer = None
        if weight is not None and bias is not None and weight.numel() > 0:
            proj_layer = torch.nn.Linear(in_dim, out_dim)
            proj_layer.weight.data = weight
            proj_layer.bias.data = bias

        return IO.NodeOutput(proj_layer, in_dim, out_dim, clip_type)


        
#----------------------------------------

class EasyProjectionLayer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        # embeddings folder scan
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        files = []
        for d in embedding_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        
                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)

                    
        return IO.Schema(
            node_id="EasyProjectionLayer",
            display_name="프로젝션 레이어",
            category="커스텀임베딩/임베드",
            description="임베딩 차원을 조건 기대값에 맞게 선형 변환하고 처리합니다.\n"
                        "주로 SD1.X는 768, SD2.X는 1024, SD3-SDXL은 2048,\n"
                        "Flux는 4096 근처(변동), Wan은 4096~8192 사이값(변동)입니다.\n"
                        "차원이 같아도 in_dim/out_dim을 명시해야 Projection이 적용됩니다.\n"
                        "레이어 로더에서의 출력값을 연결해서 쓸 수도 있습니다.\n"
                        "어댑터에도 연결은 됩니다. 하지만 작동 테스트는 아직 하지 않았습니다.\n"
                        "임베딩으로 출력시 조건 강도는 적용되지 않습니다.\n"
                        "사용할 임베딩의 차원정보는 체커 노드를 참고하세요.",
            inputs=[
                embedload.Input("EMBEDS", tooltip="임베딩 입력"),
                ProjL.Input("projection_layer", tooltip="프로젝션 레이어 입력", optional=True),
                IO.Clip.Input("clip_model", tooltip="컨디셔닝 처리용 클립 연결", optional=True),
                IO.Int.Input("in_dim", default="768", tooltip="원본 임베딩의 dim값"),
                IO.Int.Input("out_dim", default="768", tooltip="변경할 임베딩의 dim값"),
                IO.Combo.Input("clip_type",
                               options=["none", "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
                                        "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream",
                                        "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image",
                                        "flux2", "ovis"],
                               default="none",
                               tooltip="유형 정리"),
                IO.Combo.Input("save_mode",
                               options=["none","save"],
                               default="none",
                               tooltip="보정된 임베딩 레이어 정보를 파일로 저장할지 여부"),
                IO.Combo.Input("output_settings",
                               options=["conditioning","embedding output"],
                               default="conditioning",
                               tooltip="출력 방식"),
                IO.Int.Input("cond_strength", default=10, min=1, max=12,
                             tooltip="조건 강도 (1~12), 0.1~1.2로 처리됩니다."),
                IO.Int.Input("neg_scale", default=1, min=1, max=50,
                             tooltip="부정 조건 추가 약화 강도. 나누기로 처리되며, (1~50)까지 처리됩니다.\n"
                             "네거티브 임베딩을 연결할 때 사용하세요. 기본값인 1로 두면 적용되지 않습니다."),
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="보정 임베딩 전달"),
                embedload.Output("EMBEDS", tooltip="보정용 embedding 출력")
            ],
            
        )

    @classmethod
    def execute(cls, EMBEDS, in_dim=768, out_dim=768,
                projection_layer=None, clip_model=None, clip_type="none",
                save_mode="none", output_settings="conditioning", cond_strength=10,
                neg_scale=1, proj_layername="") -> IO.NodeOutput:

        embedding = EMBEDS

        if proj_layername:
            embedding_name = proj_layername
        elif isinstance(embedding, dict):
            if "name" in embedding:
                embedding_name = embedding["name"]
            elif "filename" in embedding:
                embedding_name = os.path.splitext(os.path.basename(embedding["filename"]))[0]
            else:
                embedding_name = "unnamed"
        else:
            embedding_name = "unnamed"

        # layer load/create
        if projection_layer is not None and clip_type == "none":
            projected = projection_layer(embedding)
            proj = projection_layer
        else:
            proj = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
            projected = proj(embedding) if proj is not None else embedding

        if save_mode == "save":
            save_dir = os.path.join(folder_paths.base_path, "models", "proj_embeddings")
            os.makedirs(save_dir, exist_ok=True)

            filename = resolve_filename(f"proj_{clip_type}_{embedding_name}_%number", save_dir) + ".safetensors"
            filepath = os.path.join(save_dir, filename)

            tensors = {}
            if proj is not None:
                tensors["weight"] = proj.weight.cpu()
                tensors["bias"]   = proj.bias.cpu()

            save_file(
                tensors,
                filepath,
                metadata={
                    "meta_type": clip_type,
                    "meta_in_dim": str(in_dim),
                    "meta_out_dim": str(out_dim),
                    "meta_embedding_name": embedding_name,
                    "meta_proj_exists": str(proj is not None)
                }
            )


        if output_settings == "embedding output":

            return IO.NodeOutput(None, projected)
        else:

            condscale = cond_strength / 10.0
            em_projected = projected

            conditioning = embedding_to_conditioning(em_projected, condscale, neg_scale)

            return IO.NodeOutput(conditioning, None)




 
#----------------------------------------       

class EasyEmbeddingLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        files = []
        for d in embedding_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".pt") or f.endswith(".safetensors"):

                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)


        return IO.Schema(
            node_id="EasyEmbeddingLoader",
            display_name="임베딩 로더",
            category="커스텀임베딩/임베드",
            description="embeddings 폴더에서 지정한 임베딩(.pt/.safetensors)을 불러와 Projection 출력으로 전달합니다.\n"
                        "절대 체크포인트 통합모델을 집어넣지 마세요. 오버플로우로 블루스크린이 터질 수 있습니다.\n"
                        "어댑터에도 연결은 됩니다. 하지만 작동 테스트는 아직 하지 않았습니다.",
            inputs=[
                IO.Combo.Input("embedding_name", options=files, default=files[0] if files else "",
                               tooltip="불러올 임베딩 파일을 선택하세요 (확장자 포함)"),
                IO.Combo.Input("clip_type",
                               options=["stable_diffusion","stable_cascade","sd3","stable_audio",
                                        "mochi","ltxv","pixart","cosmos","lumina2","wan","hidream",
                                        "chroma","ace","omnigen2","qwen_image","hunyuan_image",
                                        "flux2","ovis"],
                               default="stable_diffusion", tooltip="모델 유형"),
                IO.Combo.Input("mode",
                               options=["cpu","gpu"], default="cpu", tooltip="실행 장치"),
                IO.Combo.Input("with_setting", options=["off", "with_name", "with_embset"], default="off",
                               tooltip="임베딩 파일명 문자열 출력 여부/임베딩세팅 전달여부")
            ],
            outputs=[
                embedload.Output("EMBEDS", tooltip="보정용 embedding 출력"),
                IO.String.Output("embedding_name", tooltip="불러온 파일 이름 (옵션)"),
                IO.Combo.Output(
                    "clip_type",
                    options=[
                        "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
                        "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream",
                        "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image",
                        "flux2", "ovis"
                    ],
                    tooltip="모델 유형"
                )
            ],
        )

    @classmethod
    def execute(cls, embedding_name, clip_type="stable_diffusion", mode="cpu", with_setting="off") -> IO.NodeOutput:
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        tensor = None
        file_path = None

        for d in embedding_dirs:
            candidate = os.path.join(d, embedding_name)
            if os.path.exists(candidate):
                file_path = candidate
                break

        if file_path is None:
            raise FileNotFoundError(f"{embedding_name} not found")

        if file_path.endswith(".pt"):
            loaded = torch.load(file_path, map_location="cpu")
            if isinstance(loaded, torch.Tensor):
                tensor = loaded
            elif isinstance(loaded, dict):
                if "embedding" in loaded and isinstance(loaded["embedding"], torch.Tensor):
                    tensor = loaded["embedding"]
                else:
                    for v in loaded.values():
                        if isinstance(v, torch.Tensor):
                            tensor = v
                            break
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, torch.Tensor):
                        tensor = item
                        break
                    elif isinstance(item, dict):
                        if "embedding" in item and isinstance(item["embedding"], torch.Tensor):
                            tensor = item["embedding"]
                            break
                        else:
                            for v in item.values():
                                if isinstance(v, torch.Tensor):
                                    tensor = v
                                    break

        elif file_path.endswith(".safetensors"):
            tensors = load_file(file_path)
            if "embedding" in tensors and isinstance(tensors["embedding"], torch.Tensor):
                tensor = tensors["embedding"]
            else:
                for v in tensors.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break

        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise FileNotFoundError(f"{embedding_name} invalid format (no tensor found)")

        device = torch.device("cpu") if mode == "cpu" else torch.device("cuda")
        tensor = tensor.to(device)

        if with_setting == "with_name":
            return IO.NodeOutput(tensor, embedding_name, "")
        elif with_setting == "with_embset":
            return IO.NodeOutput(tensor, "", clip_type)
        else:
            return IO.NodeOutput(tensor, "", "")


#----------------------------------------
#Text Prompt Node Settings
#----------------------------------------
class EasyClipTextEncodeSimple(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextEncodeSimple",
            display_name="CLIP 텍스트 인코더 (단순형)",
            category="커스텀임베딩/프롬프트",
            description="긍정/부정 프롬프트만 처리하는 단순형 텍스트 인코더 노드.\n"
                "강도 문법은 (keywords:weight factor)식으로 쓰시면 됩니다.\n"
                "텍스트 임베딩으로 저장하시려면 출력을 임베딩으로 옮기신 뒤 이름을 적고 emb_opt를 정하여 노드실행을 하시면 됩니다.\n"
                "(텍스트임베딩 출력노드 겸용입니다. 실행버튼이 노드에 존재합니다.)",
            inputs=[
                IO.Clip.Input("clip", tooltip="컨디셔닝 처리용 클립 연결"),
                IO.String.Input("prompt_text", multiline=True, tooltip="프롬프트 입력란"),
                IO.Combo.Input("output_settings", options=["conditioning", "embedding output"], default="conditioning", tooltip="출력설정"),
                IO.String.Input("file_name", default="easyemb_%number", tooltip="파일명 지정"),
                IO.Combo.Input("emb_opt", options=["basic", "textsave"], default="basic", tooltip="텍스트임베딩 프롬프트 기록 여부"),
            ],
            is_output_node=True,
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="조건 출력"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt_text=None, output_settings="conditioning", file_name="easyemb_%number", emb_opt="basic") -> IO.NodeOutput:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cond = normalize_and_encode(clip, prompt_text, 1.0, device) if prompt_text else (None, 0)

        # layer count
        layer_count, layernames = get_layer_count(clip)
            
        if output_settings == "conditioning":
            if layer_count > 0:
                cond_final = []
                cond_final.extend(node_helpers.conditioning_set_values(cond, {"layernum": layer_count, "layers": layernames}, append=False))
            else:
                cond_final = cond
            return IO.NodeOutput(cond_final)

        elif output_settings == "embedding output":
            text = prompt_text if prompt_text is not None else ""
            if text:
                embeds = prompt_to_embedding(clip_model=clip, prompt_text=text, device="cpu")
                em_tensor = embeds.float()
                if emb_opt == "textsave":
                    save_dir = os.path.join(folder_paths.base_path,"models","embeddings")
                    os.makedirs(save_dir, exist_ok=True)
                    filename = resolve_embfilename(file_name, save_dir)
                    filepath = os.path.join(save_dir, filename)
                    metadata = {"original_text": text, "embed_shape": str(list(em_tensor.shape))}
                    try:
                        save_file({"embedding": em_tensor}, filepath, metadata=metadata)
                        print(f"{CYAN}{BOLD}[EasyClipTextEncodeSimple]{RESET} Save complete: {filepath}")
                    except Exception as e:
                        print(f"{CYAN}{BOLD}[EasyClipTextEncodeSimple]{RESET} Save failure: {e}")

            return IO.NodeOutput(None)

#----------------------------------------

class EasyClipTextEncodeTokenInfo(IO.ComfyNode):

    @classmethod
    def get_max_tokens(cls, clip):
        try:
            return clip.tokenizer.max_length
        except:
            return 77


    @classmethod
    def get_token_count_only(cls, clip, text_list):
        total_tokens = 0
        for text in text_list:
            if text and str(text).strip():

                tokens = clip.tokenize(text)
                total_tokens += sum(len(v) for v in tokens.values())
        return total_tokens
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextEncodeTokenInfo",
            display_name="CLIP 텍스트 인코더 (토큰 확인)",
            category="커스텀임베딩/프롬프트",
            description="긍정/부정 프롬프트를 처리하고 사용한 토큰 수를 간단히 측정하는 노드.\n"
                "강도 문법은 (keywords:weight factor)식으로 쓰시면 됩니다.\n"
                "토큰 사용량을 확인하시려면 아웃풋의 토큰 출력에 텍스트 노드를 연결하고 아웃풋 세팅 스위치를 켜시면 됩니다.\n"
                "(출력노드로도 사용할순 있습니다. 그경우 cmd에 정보를 출력합니다.)\n"
                "토큰이 무시될 가능성을 줄이기 위해, 슬롯에 맞게 입력하는 걸 권장합니다.",
            inputs=[
                IO.Clip.Input("clip", tooltip="컨디셔닝 처리용 클립 연결"),
                IO.String.Input("pos_keyword", multiline=True, tooltip="긍정 프롬프트"),
                IO.String.Input("neg_keyword", multiline=True, tooltip="부정 프롬프트"),
                IO.String.Input("pos_text", multiline=True, tooltip="긍정 설명문. 문장 형태로 기입합니다."),
                IO.String.Input("pos_stylekey", multiline=False, tooltip="긍정 스타일/아티스트. @형태로 입력합니다."),
                IO.String.Input("pos_calltype", multiline=False, tooltip="긍정 호출키.<호출타입:keyword:str>형태로 입력하세요."),
                IO.String.Input("neg_stylekey", multiline=False, tooltip="부정 스타일/아티스트. @형태로 입력합니다."),
                IO.String.Input("neg_calltype", multiline=False, tooltip="부정 호출키.<호출타입:keyword:str>형태로 입력하세요."),
                IO.Combo.Input("output_settings", options=["conditioning", "token_check"], default="conditioning", tooltip="출력설정"),
                IO.Combo.Input("check_prompt", options=["off", "positive", "negative", "all"], default="off", tooltip="토큰 측정 대상 프롬프트 설정"),
                IO.Combo.Input("manual_mode", options=["concat", "combine", "average"], default="concat", tooltip="컨디셔닝 결합방식")
            ],
            is_output_node=True,
            outputs=[
                IO.Conditioning.Output("pos_cond", tooltip="긍정 조건"),
                IO.Conditioning.Output("neg_cond", tooltip="부정 조건"),
                IO.String.Output("token_check_text", tooltip="확인된 예상 사용 토큰 수")            
            ]
        )

    @classmethod
    def execute(cls, clip, pos_keyword=None, neg_keyword=None, pos_text=None, pos_stylekey=None, pos_calltype=None, 
                neg_stylekey=None, neg_calltype=None, output_settings="conditioning", check_prompt="off", manual_mode="concat") -> IO.NodeOutput:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # layer count
        layer_count, layernames = get_layer_count(clip)

        pos_fields = []

        if pos_keyword and pos_keyword.strip():
            pos_fields.append(normalize_and_encode(clip, pos_keyword, 1.0, device))
        if pos_text and pos_text.strip():
            pos_fields.append(normalize_and_encode(clip, pos_text, 1.0, device))
        if pos_stylekey and pos_stylekey.strip():
            pos_fields.append(normalize_and_encode(clip, pos_stylekey, 1.0, device))
        if pos_calltype and pos_calltype.strip():
            pos_fields.append(normalize_and_encode(clip, pos_calltype, 1.0, device))


        neg_fields = []

        if neg_keyword and neg_keyword.strip():
            neg_fields.append(normalize_and_encode(clip, neg_keyword, 1.0, device))
        if neg_stylekey and neg_stylekey.strip():
            neg_fields.append(normalize_and_encode(clip, neg_stylekey, 1.0, device))
        if neg_calltype and neg_calltype.strip():
            neg_fields.append(normalize_and_encode(clip, neg_calltype, 1.0, device))


        if output_settings == "conditioning":

            Pos_cond = process_fields(clip, pos_fields, manual_mode, strength=1.0)
            Neg_cond = process_fields(clip, neg_fields, manual_mode, strength=1.0)
            
            if Pos_cond is None:
                Pos_cond = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)

            if Neg_cond is None:
                Neg_cond = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)

            if layer_count > 0:
                Pos_cond_final = []
                Pos_cond_final.extend(node_helpers.conditioning_set_values(Pos_cond, {"layernum": layer_count, "layers": layernames}, append=False))
                Neg_cond_final = []
                Neg_cond_final.extend(node_helpers.conditioning_set_values(Neg_cond, {"layernum": layer_count, "layers": layernames}, append=False))
            else:
                Pos_cond_final = Pos_cond
                Neg_cond_final = Neg_cond

            return IO.NodeOutput(Pos_cond_final, Neg_cond_final, None)

        elif output_settings == "token_check":
            pos_tokens = 0
            neg_tokens = 0
            pos_texts = [pos_keyword, pos_text, pos_stylekey, pos_calltype]
            neg_texts = [neg_keyword, neg_stylekey, neg_calltype]
            
            if check_prompt in ["positive", "all"]:
                pos_tokens = cls.get_token_count_only(clip, [t for t in pos_texts if t])
            if check_prompt in ["negative", "all"]:
                neg_tokens = cls.get_token_count_only(clip, [t for t in neg_texts if t])

            
            total_tokens = pos_tokens + neg_tokens
            max_tokens = cls.get_max_tokens(clip)

            lines = []
            if check_prompt in ["positive", "all"]:
                lines.append(f"Positive tokens: {pos_tokens}\n")
            if check_prompt in ["negative", "all"]:
                lines.append(f"Negative tokens: {neg_tokens}\n")
            if check_prompt == "all":
                lines.append(f"Total tokens: {total_tokens}\n")

            info_text = "\n".join(lines)

            if total_tokens > max_tokens:
                info_text += f"\n[!] WARNING: The total number of tokens ({total_tokens}) has exceeded the model limit ({max_tokens})!"

            print(info_text.strip())
            return IO.NodeOutput(None, None, info_text.strip())

#----------------------------------------

class EasyTranslate(IO.ComfyNode):


    @staticmethod
    def google_translate(text, target_lang="en"):
        try:
            result = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return result
        except Exception as e:
            print(f"[Translate] failed translation: {e}")
            return text

    @staticmethod
    def show_progress(stage, translated=None):
        stages = {
            "start": "Start translate...",
            "waiting": "Waiting for response...",
            "complete": "Translation complete.",
            "print": f"Translated text: {translated}" if translated else "No text"
        }
        print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} {stages[stage]}")

    @staticmethod
    def save_translation_log(request, result, file_name, save_dir):
        filename = resolve_filename(file_name, save_dir)
        filepath = os.path.join(save_dir, f"{filename}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"({request})\n-\n({result})")

    @staticmethod
    def save_as_json_dict(original, translated, file_name, lang):
        filename = resolve_filename(file_name, TRANS_DIR)
        file_path = os.path.join(TRANS_DIR, f"{filename}")
    
        # 1. data loading (else empty dict create)
        data = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    # Which line, at which point, the grammar is incorrect 
                    print(f"[EasyTranslate]{RESET} ERROR: Dictionary file syntax error! Needs verification: {e.msg} at line {e.lineno}, col {e.colno}")
                    # Return at least an empty dictionary so the system does not crash when a file is corrupted.
                    data = {}

        # 2. data update (If the base key does not exist, create it; if it exists, add the corresponding language.)
        if original not in data:
            data[original] = {}
        data[original][lang] = translated

        # 3. JSON dict style save (indent=4)
        if os.path.exists(file_path):
            shutil.copy(file_path, f"{file_path}.bak")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
        print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} Saved to dictionary file: {file_path}")

    @classmethod
    def define_schema(cls):
        # TRANS_DIR, search *.txt
        files = []
        if os.path.exists(TRANS_DIR):
            for f in os.listdir(TRANS_DIR):
                if f.endswith(".txt"):
                    files.append(f)
        return IO.Schema(
            node_id="EasyTranslate",
            display_name="번역지원노드",
            category="커스텀임베딩",
            description="구글 번역 지원 및 커스텀 번역파일을 로드해 읽는 노드.\n"
                "불리언 스위치를 통해 구글번역을 활용할지, 커스텀 유저 딕셔너리를 쓸지 고를 수 있습니다.\n"
                "번역 결과는 스위치 여부에 따라 저장할지 말지가 결정됩니다.",
            inputs=[
                IO.Boolean.Input("custom_dict", default=False, tooltip="번역 모드. false인 경우 구글 번역으로 진행하며, 커스텀 딕셔너리의 경우 파일을 준비해야 합니다.\n"
                                 "TIP: 사전 파일은 JSON 표준 형식을 따릅니다. 따옴표와 쉼표를 주의해서 수정해주세요. 실수로 파일이 깨지면 같은 폴더에 생성된 .bak 파일을 확인하세요."),
                IO.Combo.Input("file_name", options=files, default=files[0] if files else "customdict.txt", tooltip="사전 파일 선택. txtdict_save로 저장 시 이 이름을 그대로 사용합니다"),
                IO.String.Input("savefile_name", default="customdict_%number", tooltip="저장 파일명 지정.딕셔너리 제작 때에는 적용되지 않습니다."),
                IO.Combo.Input("save_opt", options=["basic", "textsave", "txtdict_save"], default="basic", tooltip="번역 내역 기록 여부. basic=pass, textsave=번역을 일반텍스트로 출력,\n"
                               "txtdict_save=번역을 텍스트 딕셔너리 구조로 저장."),
                IO.String.Input("translate_input", multiline=True, tooltip="번역할 원문 텍스트. 구글 번역은 문장을 써도 되지만,\n"
                                "커스텀 딕셔너리일 경우는 키워드만 쓰는게 나을 수 있습니다.", optional=True),
                IO.Combo.Input("target_language", options = ["skip", "en", "ja", "zh-CN", "zh-TW"], default="skip", tooltip="주요 타겟 언어 선택. skip 선택 시 번역하지 않습니다."),
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True,
            outputs=[
                IO.String.Output("translated_text", tooltip="번역된 텍스트")
            ],
        )

    @classmethod
    def execute(cls, custom_dict=False, file_name="customdict.txt", savefile_name="customdict_%number",
                save_opt="basic", translate_input=None, target_language="skip") -> IO.NodeOutput:


        request_text = translate_input or ""
        translated = request_text
        filename = resolve_filename(file_name, TRANS_DIR)
        file_path = os.path.join(TRANS_DIR, f"{filename}")
        
        # 1. Glossary
        if custom_dict:
            # glossary_file 
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} ERROR: Dictionary file syntax error! {e.msg} at line {e.lineno}, col {e.colno}")
                    print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} TIP: Check the backup file (.bak) if the dictionary is corrupted.")
                    data = {}
                
                except Exception as e:
                    print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} Dictionary file load failed: {e}")
                    data = {}
            else:
                print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} No dictionary file: {file_path}")
                data = {}
            # translate
            if request_text in data:
                translated = data[request_text].get(target_language, request_text)
            else:
                print(f"{CYAN}{BOLD}[EasyTranslate]{RESET} No match keyword: {file_path}")
                translated = request_text

        # 2. google translate (API)
        else:
            if target_language != "skip":
                cls.show_progress("start")
                translated = cls.google_translate(request_text, target_language)
                cls.show_progress("complete")
                cls.show_progress("print", translated)
                
                if save_opt == "textsave":
                    # output/translations
                    save_dir = os.path.join(folder_paths.base_path, "output", "translations")
                    os.makedirs(save_dir, exist_ok=True)
                    cls.save_translation_log(request_text, translated, savefile_name, save_dir)

                # dict file save
                elif save_opt == "txtdict_save":
                    # dictionary update (file name no add count)
                    cls.save_as_json_dict(request_text, translated, file_name, target_language)

        return IO.NodeOutput(translated)


#----------------------------------------

class EasyClipTextEncodeADV(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextEncodeADV",
            display_name="CLIP 텍스트 인코더 어드밴스",
            category="커스텀임베딩/프롬프트",
            description=
                "글로벌 프롬프트와 긍정_부정텍스트 프롬프트를 동시에 처리하는 목적으로 만든 노드. 강도 문법은 (keywords:weight factor)식으로 쓰시면 됩니다.\n"
                "주의: 서브태그 슬롯에 문장형 텍스트를 과도하게 넣으면 토큰 오버플로우가 발생할 수 있습니다.",
            inputs=[
                IO.Clip.Input("clip", tooltip="컨디셔닝 처리용 클립 연결"),
                IO.String.Input("pos_keyword", multiline=True, tooltip="긍정 프롬프트"),
                IO.String.Input("neg_keyword", multiline=True, tooltip="부정 프롬프트"),
                IO.String.Input("pos_text", multiline=True, tooltip="긍정 설명문. 문장 형태로 기입합니다."),
                IO.String.Input("pos_stylekey", multiline=False, tooltip="긍정 스타일/아티스트. @형태로 입력합니다."),
                IO.String.Input("pos_calltype", multiline=False, tooltip="긍정 호출키.<호출타입:keyword:str>형태로 입력하세요."),
                IO.String.Input("neg_stylekey", multiline=False, tooltip="부정 스타일/아티스트. @형태로 입력합니다."),
                IO.String.Input("neg_calltype", multiline=False, tooltip="부정 호출키.<호출타입:keyword:str>형태로 입력하세요."),
                IO.String.Input("global_background", multiline=True, tooltip="전역 배경 프롬프트"),
                IO.Combo.Input("manual_mode", options=["concat", "combine", "average"], default="concat", tooltip="컨디셔닝 결합방식")
            ],
            outputs=[
                IO.Conditioning.Output("Pos_cond", tooltip="최종 Pos CLIP 조건"),
                IO.Conditioning.Output("Neg_cond", tooltip="최종 Neg CLIP 조건"),
            ]
        )

    @classmethod
    def execute(cls, clip, pos_keyword=None, neg_keyword=None, pos_text=None, pos_stylekey=None,
                pos_calltype=None, neg_stylekey=None, neg_calltype=None, global_background=None, 
                manual_mode="concat") -> IO.NodeOutput:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if clip is None:
            raise ValueError("The CLIP model was not connected. A CLIP input is needed.")

        # layer count
        layer_count, layernames = get_layer_count(clip)


        pos_fields = []

        if pos_keyword and pos_keyword.strip():
            pos_fields.append(normalize_and_encode(clip, pos_keyword, 1.0, device))
        if pos_text and pos_text.strip():
            pos_fields.append(normalize_and_encode(clip, pos_text, 1.0, device))
        if pos_stylekey and pos_stylekey.strip():
            pos_fields.append(normalize_and_encode(clip, pos_stylekey, 1.0, device))
        if pos_calltype and pos_calltype.strip():
            pos_fields.append(normalize_and_encode(clip, pos_calltype, 1.0, device))
        if global_background and global_background.strip():
            pos_fields.append(normalize_and_encode(clip, global_background, 1.0, device))

        neg_fields = []
        if neg_keyword and neg_keyword.strip():
            neg_fields.append(normalize_and_encode(clip, neg_keyword, 1.0, device))
        if neg_stylekey and neg_stylekey.strip():
            neg_fields.append(normalize_and_encode(clip, neg_stylekey, 1.0, device))
        if neg_calltype and neg_calltype.strip():
            neg_fields.append(normalize_and_encode(clip, neg_calltype, 1.0, device))

        Pos_cond = process_fields(clip, pos_fields, manual_mode, strength=1.0)
        Neg_cond = process_fields(clip, neg_fields, manual_mode, strength=1.0)


        if Pos_cond is None:
            Pos_cond = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        if Neg_cond is None:
            Neg_cond = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)

        if layer_count > 0:
            Pos_cond_final = []
            Pos_cond_final.extend(node_helpers.conditioning_set_values(Pos_cond, {"layernum": layer_count, "layers": layernames}, append=False))
            Neg_cond_final = []
            Neg_cond_final.extend(node_helpers.conditioning_set_values(Neg_cond, {"layernum": layer_count, "layers": layernames}, append=False))
        else:
            Pos_cond_final = Pos_cond
            Neg_cond_final = Neg_cond

        return IO.NodeOutput(Pos_cond_final, Neg_cond_final)

#----------------------------------------

class EasyClipTextMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextMask",
            display_name="CLIP 텍스트 마스크",
            category="커스텀임베딩/프롬프트",
            description=(
                "마스크 조건에만 특정 텍스트 프롬프트를 적용하는 CLIP 인코더.\n"
                "mask_strength는 마스크 영역, extra_strength는 마스크 외만 적용됩니다.\n"
                "마스크가 없거나 style/quality가 모두 basic이면 전역 프롬프트는 적용되지 않습니다."
            ),
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("text", multiline=True, tooltip="마스크 영역에 적용할 텍스트 프롬프트.\n"
                                "마스크를 연결하지 않으면 기본 프롬프트로 기능하며,\n"
                                "마스크를 연결했지만 이곳에 요청사항을 입력하지 않은 경우 랜덤 출력물이 나옵니다."),
                IO.String.Input("style", multiline=True, tooltip="스타일 텍스트. 공통적으로 유지하고 싶은 내용을 기입해야 하고, 마스크를 연결한 경우 입력을 필수로 해야 합니다.", optional=True),
                IO.Mask.Input("mask", tooltip="적용할 마스크 (선택 사항)", optional=True),
                IO.Boolean.Input("resize_mask", default=False, tooltip="마스크를 레이턴트 크기에 맞춰 리사이즈합니다. 비활성화 시 원본 해상도를 유지합니다."),
                IO.Combo.Input("cond_avg", options=["skip", "concat", "combine", "average"], default="skip", tooltip="조건 합치기"),
                IO.Combo.Input("mask_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
                               default="5", tooltip="마스크 적용 강도.(1~30), 0.5~5.0으로 처리됩니다."),
                IO.Combo.Input("extra_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="10",
                                tooltip="마스크 외 영역 강도 (1~20), 0.1~1.0으로 처리됩니다."),
                IO.Boolean.Input("set_area_to_bounds", default=False, tooltip="켰을 시 프롬프트 키워드가 마스크 영역 내에서만 처리되도록 영역을 제한합니다.")
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="최종 CLIP 조건")
            ]
        )		

    @classmethod
    def execute(cls, clip, text, mask=None, style=None, resize_mask=False, cond_avg="skip", mask_strength="5", extra_strength="10", set_area_to_bounds=False) -> IO.NodeOutput:

        if not text or not text.strip():
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: The text is empty.")
            return IO.NodeOutput([])

        strength = 0.5 + (int(mask_strength) - 1) * (4.5 / 29.0)
        ex_strength = 0.1 + (int(extra_strength) - 1) * (0.9 / 19.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # layer count
        layer_count, layernames = get_layer_count(clip)

        mask_fields = []
        stylecond = None
        if text and text.strip():
            textcond = normalize_and_encode(clip, text, 1.0, device)
            mask_fields.append(textcond)
        if style and style.strip():
            stylecond = normalize_and_encode(clip, style, 1.0, device)
            mask_fields.append(stylecond)

        if not mask_fields:
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: Both the text and style are empty.")
            return IO.NodeOutput([])
        elif cond_avg == "skip" or len(mask_fields) < 2:
            cond_main = mask_fields[0] if mask_fields else None
        else:
            cond_main = process_fields(clip, mask_fields, cond_avg, 1)
        
        condset = {}
        cond_final = []
        if mask is None:
            if layer_count > 0:
                condset.update({"layernum": layer_count, "layers": layernames})
                cond_final.extend(node_helpers.conditioning_set_values(cond_main, condset, append=False))
            else:
                cond_final.extend(cond_main)

            return IO.NodeOutput(cond_final)

        mask_proc = ensure_mask_tensor(mask) 
        
        bbox = get_mask_bbox(mask_proc.squeeze(1))
        area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])

        # 5. Cond extend
        # mask settings
        condset.update({"mask": mask_proc,"mask_strength": strength, "area": area, "set_area_to_bounds": set_area_to_bounds})
        if layer_count > 0:
            condset.update({"layernum": layer_count, "layers": layernames})
        
        cond_final.extend(node_helpers.conditioning_set_values(textcond, condset, append=False))

        if stylecond is not None:
            style_dict = {"global_mask_c": True, "mask_strength": ex_strength}

            style_cond_processed = node_helpers.conditioning_set_values(stylecond, style_dict, append=False)

            cond_final = cond_final + style_cond_processed

        return IO.NodeOutput(cond_final)


#----------------------------------------

class EasyClipMultiTextMask(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipMultiTextMask",
            display_name="CLIP 멀티 텍스트 마스크",
            category="커스텀임베딩/프롬프트",
            description=(
                "마스크 조건에만 특정 텍스트 프롬프트를 적용하는 CLIP 인코더.\n"
                "mask_strength는 마스크 영역, extra_strength는 마스크 외만 적용됩니다.\n"
                "마스크가 없거나 style/quality가 모두 basic이면 전역 프롬프트는 적용되지 않습니다.\n"
                "주의)멀티 마스크인 경우 단순한 배경만 있는 이미지가 아니면 이미지가 깨질 가능성이 높습니다.\n"
                "이 노드는 샘플러 스텝을 20 이상 주는걸 고려해 짜여있습니다. 그 미만의 스텝일 경우 샘플링 중 터집니다."
            ),
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("text", multiline=True, tooltip="마스크 영역에 적용할 텍스트 프롬프트.\n"
                                "마스크를 연결하지 않으면 기본 프롬프트로 기능하며,\n"
                                "마스크를 연결했지만 이곳에 요청사항을 입력하지 않은 경우 랜덤 출력물이 나옵니다."),
                IO.String.Input("sub_text", multiline=True, tooltip="두번째 마스크 영역에 적용할 텍스트 프롬프트.\n"
                                "두번째 마스크를 연결하지 않으면 적용되지 않습니다."),
                IO.String.Input("style", multiline=True, tooltip="스타일 텍스트. 공통적으로 유지하고 싶은 내용을 기입해야 하고, 마스크를 연결한 경우 입력을 필수로 해야 합니다.", optional=True),
                IO.Mask.Input("mask", tooltip="적용할 마스크 (선택 사항)", optional=True),
                IO.Mask.Input("sub_mask", tooltip="적용할 마스크 (선택 사항).첫번째 마스크가 없을경우 적용되지 않습니다.", optional=True),
                IO.Boolean.Input("resize_mask", default=False, tooltip="마스크를 레이턴트 크기에 맞춰 리사이즈합니다. 비활성화 시 원본 해상도를 유지합니다."),
                IO.Combo.Input("cond_avg", options=["skip", "concat", "combine", "average"], default="skip", tooltip="조건 합치기"),
                IO.Combo.Input("mask_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
                               default="5", tooltip="마스크 적용 강도.(1~30), 0.5~5.0으로 처리됩니다."),
                IO.Combo.Input("extra_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="10",
                                tooltip="마스크 외 영역 강도 (1~20), 0.1~1.0으로 처리됩니다."),
                IO.Boolean.Input("set_area_to_bounds", default=False, tooltip="켰을 시 프롬프트 키워드가 마스크 영역 내에서만 처리되도록 영역을 제한합니다. 마스크가 두개일 경우는 충돌이 발생할 수 있습니다.")
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="최종 CLIP 조건")
            ]
        )		

    @classmethod
    def execute(cls, clip, text, mask=None, sub_text=None, sub_mask=None, style=None, resize_mask=False, cond_avg="skip", mask_strength="5", extra_strength="10", set_area_to_bounds=False) -> IO.NodeOutput:

        if not text or not text.strip():
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: The text is empty.")
            return IO.NodeOutput([])

        strength = 0.5 + (int(mask_strength) - 1) * (4.5 / 29.0)
        ex_strength = 0.1 + (int(extra_strength) - 1) * (0.9 / 19.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mask_fields = []
        stylecond = None
        if text and text.strip():
            textcond = normalize_and_encode(clip, text, 1.0, device)
            mask_fields.append(textcond)
        if style and style.strip():
            stylecond = normalize_and_encode(clip, style, 1.0, device)
            mask_fields.append(stylecond)

        if not mask_fields:
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: Both the text and style are empty.")
            return IO.NodeOutput([])
        elif cond_avg == "skip" or len(mask_fields) < 2:
            cond_main = mask_fields[0] if mask_fields else None
        else:
            cond_main = process_fields(clip, mask_fields, cond_avg, strength=strength)

        # layer count
        layer_count, layernames = get_layer_count(clip)

        cond_final = []
        cond_dict1 = {}
        if mask is None:
            if layer_count > 0:
                cond_dict1.update({"layernum": layer_count, "layers": layernames})
                cond_final.extend(node_helpers.conditioning_set_values(cond_main, cond_dict1, append=False))
            else:
                cond_final.extend(cond_main)

            return IO.NodeOutput(cond_final)

        mask_proc = ensure_mask_tensor(mask)
        
        bbox = get_mask_bbox(mask_proc.squeeze(1))
        area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])

        # 5. Cond extend
        if layer_count > 0:
            cond_dict1.update({"layernum": layer_count, "layers": layernames})
        # mask settings
        cond_dict1.update({"mask": mask_proc, "mask_strength": strength, "area": area, "set_area_to_bounds": set_area_to_bounds})
        style_dict = {
            "global_mask_c": True,
            "mask_strength": ex_strength
        }

        sub_mask_proc = torch.zeros_like(mask_proc)
        has_sub = False
        sub_cond = None
        cond_dict2 = None
        if sub_mask is not None and sub_text and sub_text.strip():
            sub_textcond = normalize_and_encode(clip, sub_text, 1.0, device)
            sub_mask_proc = ensure_mask_tensor(sub_mask)

            sub_bbox = get_mask_bbox(sub_mask_proc.squeeze(1))
            sub_area = scale_bbox_to_latent(sub_bbox, sub_mask_proc.shape[-2:])

            cond_dict2 = {
                "mask": sub_mask_proc,
                "mask_strength": strength,
                "area": sub_area,
                "set_area_to_bounds": set_area_to_bounds
            }

                
            has_sub = True

        if has_sub:
            #textcond "start_percent": 0.2, "end_percent": 0.5
            #sub_textcond "start_percent": 0.5, "end_percent": 0.8
            cond_one = sampler_edit.editcond_set_values_with_timestep_range(textcond, cond_dict1, start_percent=0.2, end_percent=0.5)
            sub_cond = sampler_edit.editcond_set_values_with_timestep_range(sub_textcond, cond_dict2, start_percent=0.5, end_percent=0.8)
            cond_final.extend(cond_one)
            cond_final.extend(sub_cond)
        else:
            #textcond "start_percent": 0.3, "end_percent": 0.8
            cond_one = sampler_edit.editcond_set_values_with_timestep_range(textcond, cond_dict1, start_percent=0.3, end_percent=0.8)
            cond_final.extend(cond_one)

        # style set(global settings)
        if stylecond is not None:
            #stylecond "start_percent": 0.0, "end_percent": 1.0
            style_cond_processed = sampler_edit.editcond_set_values_with_timestep_range(stylecond, style_dict, start_percent=0.0, end_percent=1.0)

            cond_final = cond_final + style_cond_processed

        return IO.NodeOutput(cond_final)


#----------------------------------------

class EasyClipTextMaskNoArea(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextMaskNoArea",
            display_name="CLIP 텍스트 마스크(하드마스크)",
            category="커스텀임베딩/프롬프트",
            description=(
                "마스크 조건에만 특정 텍스트 프롬프트를 적용하는 CLIP 인코더.\n"
                "mask_strength는 마스크 영역, extra_strength는 마스크 외만 적용됩니다.\n"
                "마스크가 없거나 style/quality가 모두 basic이면 전역 프롬프트는 적용되지 않습니다.\n"
                "에어리어 연산을 하지 않으므로 인페인팅에 사용하려면 키워드를 세밀하게 쓸 필요가 있습니다."
            ),
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("text", multiline=True, tooltip="마스크 영역에 적용할 텍스트 프롬프트.\n"
                                "마스크를 연결하지 않으면 기본 프롬프트로 기능하며,\n"
                                "마스크를 연결했지만 이곳에 요청사항을 입력하지 않은 경우 랜덤 출력물이 나옵니다."),
                IO.String.Input("style", multiline=True, tooltip="스타일 텍스트. 공통적으로 유지하고 싶은 내용을 기입해야 하고, 마스크를 연결한 경우 입력을 필수로 해야 합니다.", optional=True),
                IO.Mask.Input("mask", tooltip="적용할 마스크 (선택 사항)", optional=True),
                IO.Combo.Input("mask_mode", options=["normal", "small_spread", "big_spread", "blur", "clear"], default="normal", tooltip="마스크 적용 모드", optional=True),
                IO.Boolean.Input("resize_mask", default=True, tooltip="마스크를 레이턴트 크기에 맞춰 리사이즈합니다. 비활성화 시 원본 해상도를 유지합니다."),
                IO.Combo.Input("cond_avg", options=["skip", "concat", "combine", "average"], default="skip", tooltip="조건 합치기"),
                IO.Combo.Input("mask_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
                               default="5", tooltip="마스크 적용 강도.(1~30), 0.5~5.0으로 처리됩니다."),
                IO.Combo.Input("extra_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="10",
                                tooltip="마스크 외 영역 강도 (1~20), 0.1~1.0으로 처리됩니다."),
                IO.Boolean.Input("maskbreaker", default=False, tooltip="대상이 공백 라텐트 캔버스일 경우에 사용합니다. 스타일(배경)과 텍스트의 적용 범위제한을 해제합니다. 훅 샘플러의 마스크 영역 제한을 해제하는데 사용됩니다."),
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="최종 CLIP 조건")
            ]
        )		

    @classmethod
    def execute(cls, clip, text,style=None, mask=None, mask_mode="normal", resize_mask=True, cond_avg="skip", mask_strength="5", extra_strength="10", maskbreaker=False) -> IO.NodeOutput:

        if not text or not text.strip():
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: The text is empty.")
            return IO.NodeOutput([])

        strength = 0.5 + (int(mask_strength) - 1) * (4.5 / 29.0)
        ex_strength = 0.1 + (int(extra_strength) - 1) * (0.9 / 19.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mask_fields = []
        stylecond = None
        if text and text.strip():
            textcond = normalize_and_encode(clip, text, 1.0, device)
            mask_fields.append(textcond)
        if style and style.strip():
            stylecond = normalize_and_encode(clip, style, 1.0, device)
            mask_fields.append(stylecond)

        if not mask_fields:
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: Both the text and style are empty.")
            return IO.NodeOutput([])
        elif cond_avg == "skip" or len(mask_fields) < 2:
            cond_main = mask_fields[0] if mask_fields else None
        else:
            cond_main = process_fields(clip, mask_fields, cond_avg, 1)

        # layer count
        layer_count, layernames = get_layer_count(clip)

        cond_final = []
        cond_dict = {}
        if mask is None:
            if layer_count > 0:
                cond_dict.update({"layernum": layer_count, "layers": layernames})
                cond_final.extend(node_helpers.conditioning_set_values(cond_main, cond_dict, append=False))
            else:
                cond_final.extend(cond_main)

            return IO.NodeOutput(cond_final)

        mask_proc = ensure_mask_tensor(mask)
        if resize_mask:
            h, w = mask_proc.shape[-2:]
            target_h, target_w = h // 8, w // 8
            mask_proc = F.interpolate(mask_proc, size=(target_h, target_w), mode='area')
            mask_proc = mask_proc.clamp(0.0, 1.0)
        mask_proc = apply_mask_mode(mask_proc, mask_mode)

        # 5. Cond extend
        if layer_count > 0:
            cond_dict.update({"layernum": layer_count, "layers": layernames})
        # mask settings
        cond_dict.update({"mask": mask_proc, "mask_strength": strength, "set_area_to_bounds": False})
        cond_final.extend(node_helpers.conditioning_set_values(textcond, cond_dict, append=False))

        if stylecond is not None:
            style_dict = {"global_mask_c": True, "mask_strength": ex_strength}
            if maskbreaker:
                style_dict.update({"masksetkey": maskbreaker})

            style_cond_processed = node_helpers.conditioning_set_values(stylecond, style_dict, append=False)

            cond_final = cond_final + style_cond_processed

        return IO.NodeOutput(cond_final)


#----------------------------------------

class EasyClipTextMask_And_Image_Using_Vision(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] > 4 and arr.shape[3] <= 4:
            # (B, H, W, C) -> (B, C, H, W)
            arr = arr.permute(0, 3, 1, 2)
        elif arr.shape[1] > 4 and arr.shape[2] <= 4:
            # (B, C, H, W)
            arr = arr.permute(0, 2, 1, 3)
            
        return arr.float()

    @classmethod
    def ensure_ref_image_tensor_hwc(cls, arr):

        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)  # (H, W) -> (1, H, W, 1)
        elif arr.dim() == 3:
            if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
                arr = arr.unsqueeze(0).permute(0, 2, 3, 1) # (C, H, W) -> (1, H, W, C)
            else:
                arr = arr.unsqueeze(0) # (H, W, C) -> (1, H, W, C)
        elif arr.dim() != 4:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            arr = arr.permute(0, 2, 3, 1)

        return arr.float()

    @classmethod
    def apply_mask(cls, target_shape, mask):
        if mask.shape[-1] != target_shape[-1] or mask.shape[-2] != target_shape[-2]:
            mask = F.interpolate(mask.float(), size=target_shape, mode="nearest")
        return mask

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextMask_And_Image_Using_Vision",
            display_name="CLIP 텍스트 마스크(참고이미지 적용, Clip Vision 내장 필수)",
            category="커스텀임베딩/프롬프트",
            description=(
                "마스크 조건에만 특정 텍스트 프롬프트를 적용하는 CLIP 인코더.\n"
                "mask_strength는 마스크 영역, extra_strength는 마스크 외만 적용됩니다.\n"
                "ref_image가 없다면 마스크 프롬프트로서 기능합니다.(ref_image를 사용하려면 모델이 clipvision을 지원해야합니다.)\n"
                "마스크가 없거나 style/quality가 모두 basic이면 전역 프롬프트는 적용되지 않습니다.\n"
            ),
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("text", multiline=True, tooltip="마스크 영역에 적용할 텍스트 프롬프트.\n"
                                "마스크를 연결하지 않으면 기본 프롬프트로 기능하며,\n"
                                "마스크를 연결했지만 이곳에 요청사항을 입력하지 않은 경우 랜덤 출력물이 나옵니다."),
                IO.String.Input("style", multiline=True, tooltip="스타일 텍스트. 공통적으로 유지하고 싶은 내용을 기입해야 하고, 마스크를 연결한 경우 입력을 필수로 해야 합니다.", optional=True),
                IO.Mask.Input("mask", tooltip="적용할 마스크 (선택 사항)", optional=True),
                IO.Image.Input("ref_Image", tooltip="참고할 레퍼런스 이미지 (선택 사항)", optional=True),
                IO.Boolean.Input("resize_mask", default=False, tooltip="마스크를 이미지 크기에 맞춰 리사이즈합니다. 비활성화 시 원본 해상도를 유지합니다.(모델이 clipvision을 지원해야합니다)"),
                IO.Combo.Input("cond_avg", options=["skip", "concat", "combine", "average"], default="skip", tooltip="조건 합치기. 마스크가 있을 경우 작동하지 않습니다."),
                IO.Combo.Input("mask_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
                               default="5", tooltip="마스크 적용 강도.(1~30), 0.5~5.0으로 처리됩니다."),
                IO.Combo.Input("extra_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="10",
                                tooltip="마스크 외 영역 강도 (1~20), 0.1~1.0으로 처리됩니다."),
                IO.Boolean.Input("set_area_to_bounds", default=False, tooltip="켰을 시 프롬프트 키워드가 마스크 영역 내에서만 처리되도록 영역을 제한합니다.")
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="최종 CLIP 조건")
            ]
        )		

    @classmethod
    def execute(cls, clip, text, style=None, mask=None, ref_Image=None, resize_mask=False, cond_avg="skip", mask_strength="5", extra_strength="10", set_area_to_bounds=False) -> IO.NodeOutput:

        if not text or not text.strip():
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: The text is empty.")
            return IO.NodeOutput([])

        strength = 0.5 + (int(mask_strength) - 1) * (4.5 / 29.0)
        ex_strength = 0.1 + (int(extra_strength) - 1) * (0.9 / 19.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_keys = [
            "vision_model", "patch_embedding", "visual_projection",
            "conv1.weight", "class_embedding", "vision.encoder"
        ]
        vision_support = False
        if isinstance(clip, dict):
            vision_support = any(any(vk in key for vk in vision_keys) for key in clip.keys())
        elif hasattr(clip, "__dict__"):
            vision_support = any(any(vk in attr for vk in vision_keys) for attr in dir(clip))
    
        mask_fields = []
        stylecond = None
        if text and text.strip():
            textcond = normalize_and_encode(clip, text, 1.0, device)
            mask_fields.append(textcond)
        if style and style.strip():
            stylecond = normalize_and_encode(clip, style, 1.0, device)
            mask_fields.append(stylecond)
        num_conds = len(mask_fields)
        textcond = mask_fields[0] if num_conds > 0 else None
        stylecond = mask_fields[1] if num_conds > 1 else None

        # layer count
        layer_count, layernames = get_layer_count(clip)

        cond_final = []
        cond_dict = {}
        # 2. [Pre-branch filtering]
        if mask is None or num_conds == 1:
            if num_conds == 1 or cond_avg == "skip":
                cond_main = textcond
                print(f"{CYAN}[EasyClipTextMask] Since there is only one input text, it is passed as a regular prompt.{RESET}")
            else:
                cond_main = process_fields(clip, mask_fields, cond_avg, 1.0)

            if layer_count > 0:
                cond_dict.update({"layernum": layer_count, "layers": layernames})
                cond_final.extend(node_helpers.conditioning_set_values(cond_main, cond_dict, append=False))
            else:
                cond_final.extend(cond_main)

            return IO.NodeOutput(cond_final)

        else:
            
            processed_image = None
            mask_proc = ensure_mask_tensor(mask)  # ndim==4
            if layer_count > 0:
                cond_dict.update({"layernum": layer_count, "layers": layernames})

        if ref_Image is not None:
            if not vision_support:
                print("[EasyClipTextMask] Currently, CLIP does not support Vision input. Switch to text/mask mode.")
                bbox = get_mask_bbox(mask_proc)
                area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])


            else:
                processed_image_samples = cls.ensure_image_tensor(ref_Image)
                processed_image = cls.ensure_ref_image_tensor_hwc(processed_image_samples)
                images = [processed_image[:, :, :, :3]]
                tokens = clip.tokenize(text, images=images)
                cond_main = clip.encode_from_tokens_scheduled(tokens)
                if resize_mask:
                    target_spatial_shape = processed_image_samples.shape[-2:]
                    mask_proc = cls.apply_mask(target_spatial_shape, mask_proc)
                bbox = get_mask_bbox(mask_proc)
                area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])

        else:
            bbox = get_mask_bbox(mask_proc)
            area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])
        
        cond_dict.update({"mask": mask_proc, "mask_strength": strength, "area": area, "set_area_to_bounds": set_area_to_bounds})
        cond_final.extend(node_helpers.conditioning_set_values(cond_main, cond_dict, append=False))

        if stylecond:            
            style_inject_dict = {"global_mask_c": True, "mask_strength": ex_strength}
            style_cond_processed = node_helpers.conditioning_set_values(stylecond, style_inject_dict, append=False)

            cond_final = cond_final + style_cond_processed

        return IO.NodeOutput(cond_final)

#----------------------------------------

class EasyClipTextMask_And_Image(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] > 4 and arr.shape[3] <= 4:
            # (B, H, W, C) -> (B, C, H, W)
            arr = arr.permute(0, 3, 1, 2)
        elif arr.shape[1] > 4 and arr.shape[2] <= 4:
            # (B, C, H, W)
            arr = arr.permute(0, 2, 1, 3)
            
        return arr.float()

    @classmethod
    def ensure_ref_image_tensor_hwc(cls, arr):

        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)  # (H, W) -> (1, H, W, 1)
        elif arr.dim() == 3:
            if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
                arr = arr.unsqueeze(0).permute(0, 2, 3, 1) # (C, H, W) -> (1, H, W, C)
            else:
                arr = arr.unsqueeze(0) # (H, W, C) -> (1, H, W, C)
        elif arr.dim() != 4:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            arr = arr.permute(0, 2, 3, 1)

        return arr.float()

    @classmethod
    def apply_mask(cls, target_shape, mask):
        if mask.shape[-1] != target_shape[-1] or mask.shape[-2] != target_shape[-2]:
            mask = F.interpolate(mask.float(), size=target_shape, mode="nearest")
        return mask

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextMask_And_Image",
            display_name="CLIP 텍스트 마스크(참고이미지 적용)",
            category="커스텀임베딩/프롬프트",
            description=(
                "실험노드입니다. 고정된 결과가 나오지 않아 노드등록은 하지 않습니다.\n"
                "마스크 조건에만 특정 텍스트 프롬프트를 적용하는 CLIP 프롬프트.\n"
                "mask_strength는 마스크 영역, extra_strength는 마스크 외만 적용됩니다.\n"
                "ref_image가 없다면 마스크 프롬프트로서 기능합니다.(ref_image를 사용하려면 모델이 clipvision을 지원해야합니다.)\n"
                "마스크가 없거나 style/quality가 모두 basic이면 전역 프롬프트는 적용되지 않습니다.\n"
            ),
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("text", multiline=True, tooltip="마스크 영역에 적용할 텍스트 프롬프트.\n"
                                "마스크를 연결하지 않으면 기본 프롬프트로 기능하며,\n"
                                "마스크를 연결했지만 이곳에 요청사항을 입력하지 않은 경우 랜덤 출력물이 나옵니다."),
                IO.String.Input("style", multiline=True, tooltip="스타일 텍스트. 공통적으로 유지하고 싶은 내용을 기입해야 하고, 마스크를 연결한 경우 입력을 필수로 해야 합니다.", optional=True),
                IO.Mask.Input("mask", tooltip="적용할 마스크 (선택 사항)", optional=True),
                IO.Image.Input("ref_Image", tooltip="참고할 레퍼런스 이미지 (선택 사항)", optional=True),
                IO.Combo.Input("style_type", options=["Pass", "Edge", "Color", "Brightness", "AdaIN"], default="Pass", tooltip="적용할 스타일 전이 모드, Pass는 이미지 처리를 취소합니다."),
                IO.Combo.Input("style_str", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], default="0", tooltip="스타일 전이 강도.0이면 이미지 처리가 취소됩니다."),
                IO.Boolean.Input("resize_mask", default=False, tooltip="마스크를 이미지 크기에 맞춰 리사이즈합니다. 비활성화 시 원본 해상도를 유지합니다."),
                IO.Combo.Input("cond_avg", options=["skip", "concat", "combine", "average"], default="skip", tooltip="조건 합치기. 마스크가 있을 경우 작동하지 않습니다."),
                IO.Combo.Input("mask_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
                               default="5", tooltip="마스크 적용 강도.(1~30), 0.5~5.0으로 처리됩니다."),
                IO.Combo.Input("extra_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="10",
                                tooltip="마스크 외 영역 강도 (1~20), 0.1~1.0으로 처리됩니다."),
                IO.Boolean.Input("set_area_to_bounds", default=False, tooltip="켰을 시 프롬프트 키워드가 마스크 영역 내에서만 처리되도록 영역을 제한합니다.")
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="최종 CLIP 조건")
            ]
        )		

    @classmethod
    def execute(cls, clip, text, style=None, mask=None, ref_Image=None, style_type="Pass", resize_mask=False, cond_avg="skip", mask_strength="5", extra_strength="10", set_area_to_bounds=False) -> IO.NodeOutput:

        if not text or not text.strip():
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: The text is empty.")
            return IO.NodeOutput([])

        strength = 0.5 + (int(mask_strength) - 1) * (4.5 / 29.0)
        ex_strength = 0.1 + (int(extra_strength) - 1) * (0.9 / 19.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mask_fields = []
        stylecond = None
        if text and text.strip():
            textcond = normalize_and_encode(clip, text, 1.0, device)
            mask_fields.append(textcond)
        if style and style.strip():
            stylecond = normalize_and_encode(clip, style, 1.0, device)
            mask_fields.append(stylecond)
        num_conds = len(mask_fields)
        textcond = mask_fields[0] if num_conds > 0 else None
        stylecond = mask_fields[1] if num_conds > 1 else None

        # layer count
        layer_count, layernames = get_layer_count(clip)

        cond_final = []
        cond_dict = {}
        # 2. [Pre-branch filtering]
        if mask is None or num_conds == 1:
            if num_conds == 1 or cond_avg == "skip":
                cond_main = textcond
                print(f"{CYAN}[EasyClipTextMask] Since there is only one input text, it is passed as a regular prompt.{RESET}")
            else:
                cond_main = process_fields(clip, mask_fields, cond_avg, 1.0)

            if layer_count > 0:
                cond_dict.update({"layernum": layer_count, "layers": layernames})
                cond_final.extend(node_helpers.conditioning_set_values(cond_main, cond_dict, append=False))
            else:
                cond_final.extend(cond_main)

            return IO.NodeOutput(cond_final)
            
        # If mask, cond == 2
        cond_main = textcond

        mask_proc = ensure_mask_tensor(mask)  # ndim==4
        if layer_count > 0:
            cond_dict.update({"layernum": layer_count, "layers": layernames})

        palette_features = None
        if ref_Image is not None:
            style_strength_val = float(style_str) / 10.0
            if style_strength_val > 0 and style_type != "Pass":
                image = cls.ensure_image_tensor(ref_Image) # (B, C, H, W)
                images = cls.ensure_ref_image_tensor_hwc(image)
                ref_pixels = images[:, :, :, :3]  # (B, H, W, C)
                mu = ref_pixels.mean(dim=(1, 2), keepdim=True)
                sigma = ref_pixels.std(dim=(1, 2), keepdim=True) + 1e-6
        
                gray = ref_pixels.mean(dim=-1, keepdim=True)
                edge_y = torch.abs(gray[:, :-1, :, :] - gray[:, 1:, :, :])
                edge_x = torch.abs(gray[:, :, :-1, :] - gray[:, :, 1:, :])


                mean_edge_y = edge_y.mean(dim=(1, 2), keepdim=True)
                mean_edge_x = edge_x.mean(dim=(1, 2), keepdim=True)
                edges = (mean_edge_y + mean_edge_x) * 0.5
                brightness = gray.mean(dim=(1, 2), keepdim=True)
                color_vector = ref_pixels.mean(dim=(1, 2), keepdim=True)

                palette_features = {
                    
                    "mean": mu,
                    "std": sigma,
                    "edges": edges,
                    "colors": color_vector,
                    "brightness": brightness
                }
                if resize_mask:
                    target_spatial_shape = image.shape[-2:]
                    mask_proc = cls.apply_mask(target_spatial_shape, mask_proc)
            else:
                print(f"[EasyClipTextMask] The image Guide has been skipped. (value: str={style_strength_val}, type={style_type})")

        bbox = get_mask_bbox(mask_proc)
        area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])

        cond_dict.update({"mask": mask_proc, "mask_strength": strength, "area": area, "set_area_to_bounds": set_area_to_bounds})

        style_inject_dict = {"global_mask_c": True, "mask_strength": ex_strength}

        if ref_Image is not None:
            cond_dict.update({"palette_features": palette_features, "style_type": style_type, "style_str": style_strength_val})

        cond_final.extend(node_helpers.conditioning_set_values(cond_main, cond_dict, append=False))

        if stylecond:
            style_cond_processed = node_helpers.conditioning_set_values(stylecond, style_inject_dict, append=False)

            cond_final = cond_final + style_cond_processed

        return IO.NodeOutput(cond_final)

#----------------------------------------
class EasyClipTextMask_And_latent(IO.ComfyNode):

    @classmethod
    def apply_mask(cls, latent_image, mask):
        if mask.shape[-1] != latent_image.shape[-1] or mask.shape[-2] != latent_image.shape[-2]:
            mask = F.interpolate(mask.float(), size=latent_image.shape[-2:], mode="nearest")
        return mask

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyClipTextMask_And_latent",
            display_name="CLIP 텍스트 마스크(라텐트 융합 빌더)",
            category="커스텀임베딩/프롬프트",
            description=(
                "실험노드입니다. 고정된 결과가 나오지 않아 노드등록은 하지 않습니다.\n"
                "1. 마스크드 텍스트 컨디셔닝을 빌드하고,\n"
                "2. 가공된 레퍼런스 라텐트 텐서와 가이드 훅 정보를 담은 독립 라텐트 컨디셔닝을 빌드한 뒤,\n"
                "3. 이 둘을 cond_avg로 정밀 결합하고,\n"
                "4. 최종 배경 스타일 조건을 병합(extend)하여 반환합니다.\n"
                "AdaIN/Spatial의 경우는 구도와 화풍이 비슷해야 이미지 붕괴를 줄일 수 있습니다."
            ),
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("text", multiline=True, tooltip="마스크 영역에 적용할 텍스트 프롬프트."),
                IO.String.Input("style", multiline=True, tooltip="스타일 텍스트. 공통적으로 유지하고 싶은 내용을 기입해야 하고, 마스크를 연결한 경우 입력을 필수로 해야 합니다.", optional=True),
                IO.Mask.Input("mask", tooltip="적용할 마스크 (선택 사항)", optional=True),
                IO.Latent.Input("ref_latent", tooltip="참고할 레퍼런스 라텐트 (선택 사항)", optional=True),
                IO.Boolean.Input("resize_mask", default=False, tooltip="마스크를 레이턴트 크기에 맞춰 리사이즈합니다."),
                IO.Combo.Input("cond_avg", options=["concat", "combine", "average"], default="combine", tooltip="일반 컨디셔닝 병합용. 텍스트 마스크,레퍼런스라텐트 사용시는 기능이 잠깁니다."),
                IO.Combo.Input("mask_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
                               default="5", tooltip="마스크 적용 강도.(1~30), 0.5~5.0으로 처리됩니다."),
                IO.Combo.Input("extra_strength", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="10",
                                tooltip="마스크 외 영역 강도 (1~20), 0.1~1.0으로 처리됩니다."),
                IO.Combo.Input("transfer_str", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="0", tooltip="샘플 라텐트의 가이딩 강도."),
                IO.Combo.Input("style_type", options=["Pass", "Latent_AdaIN", "Spatial_Attention", "Residual_Blending", "Channel_Selective"], default="Pass", tooltip="라텐트 가이딩 정보 유형"),
                IO.Combo.Input("latent_mode", options=["default", "linear", "ease_in", "EASE_OUT", "EASE_IN_OUT"], default="default", tooltip="라텐트 가이딩 정보 유형"),
                IO.Float.Input("Latent_activation_value", default=1.0, min=0.0, max=1.0, step=0.1, tooltip="라텐트 가이드 강도 (훅 영향력)"),
                IO.Boolean.Input("set_area_to_bounds", default=False, tooltip="켰을 시 프롬프트 키워드가 마스크 영역 내에서만 처리되도록 영역을 제한합니다.")
            ],
            outputs=[
                IO.Conditioning.Output("conditioning", tooltip="최종 융합 조건")
            ]
        )

    @classmethod
    def execute(cls, clip, text, style=None, mask=None, ref_latent=None, resize_mask=False, cond_avg="combine", mask_strength="5", extra_strength="10",
                transfer_str="0", style_type="Pass", latent_mode="default", Latent_activation_value=1.0, set_area_to_bounds=False) -> IO.NodeOutput:

        if not text or not text.strip():
            print(f"{CYAN}{BOLD}[EasyClipTextMask]{RESET} Warning: The text is empty.")
            return IO.NodeOutput([])

        strength = 0.5 + (int(mask_strength) - 1) * (4.5 / 29.0)
        ex_strength = 0.1 + (int(extra_strength) - 1) * (0.9 / 19.0)

        device = comfy.model_management.get_torch_device()

        # Prepare encoding source (initialize `stylecond` to `None` to prevent `NameError`)
        textcond = normalize_and_encode(clip, text, 1.0, device)
        stylecond = normalize_and_encode(clip, style, 1.0, device) if style and style.strip() else None

        # layer count
        layer_count, layernames = get_layer_count(clip)

        inject_dict={}
        cond_final = []
        # If no mask, Returns a simple text condition 
        if mask is None:
            if layer_count > 0:
                inject_dict.update({"layernum": layer_count, "layers": layernames})
                cond_final.extend(node_helpers.conditioning_set_values(textcond, inject_dict, append=False))
            else:
                cond_final.extend(textcond)

            return IO.NodeOutput(cond_final)

        mask_proc = ensure_mask_tensor(mask)
        bbox = get_mask_bbox(mask_proc)
        area = scale_bbox_to_latent(bbox, mask_proc.shape[-2:])

        # ----------------------------------------------------
        # step 1: pure [Masked Text Conditioning]
        # ----------------------------------------------------
        if layer_count > 0:
            inject_dict.update({"layernum": layer_count, "layers": layernames})

        inject_dict.update({
            "mask": mask_proc,
            "mask_strength": strength,
            "area": area,
            "set_area_to_bounds": set_area_to_bounds})

        style_inject_dict = {"global_mask_c": True, "mask_strength": ex_strength}

        # ----------------------------------------------------
        # step 2: [Latent Conditioning + Guide Hook]
        # ----------------------------------------------------
        # design a Latent-specific conditioning structure intended purely for guidance purposes.
        if ref_latent is not None:
            try:
                t_str_val = int(transfer_str)/20
            except (ValueError, TypeError):
                t_str_val = 0

            if (t_str_val > 0 and 
                style_type != "Pass" and 
                latent_mode != "default" and 
                Latent_activation_value > 0.0):
                
                lat_samples = ref_latent.get("samples", None)
                
                ref_latent_features = {}
                if lat_samples is not None:
                    lat_samples = lat_samples.to(device)

                    if lat_samples.dim() == 5:
                        mu = lat_samples.mean(dim=(3, 4), keepdim=True)
                        sigma = lat_samples.std(dim=(3, 4), keepdim=True) + 1e-6

                    elif lat_samples.dim() == 4:
                        mu = lat_samples.mean(dim=(2, 3), keepdim=True)
                        sigma = lat_samples.std(dim=(2, 3), keepdim=True) + 1e-6

                    ref_latent_features = {
                        "mean": mu,
                        "std": sigma,
                        "v_feat": lat_samples, # Spatial Attention
                        "base": lat_samples    # Residual / Channel Selective
                    }

                inject_dict.update({
                    "reference_latents": lat_samples,
                    "ref_latent_features": ref_latent_features, 
                    "ref_lat_transfer_str": t_str_val,  
                    "ref_lat_style_type": style_type,
                    "reference_weight": Latent_activation_value,
                    "palette_mode": latent_mode,
                })
            else:
                print(f"[EasyClipTextMask] The Latent Guide has been skipped. (value: str={transfer_str}, type={style_type}, mode={latent_mode}, act={Latent_activation_value})")

        # ----------------------------------------------------
        # step 3: [Cond avg]
        # ----------------------------------------------------
        cond_final.extend(node_helpers.conditioning_set_values(textcond, inject_dict, append=False))

        # ----------------------------------------------------
        # step 4: [style cond Extend]
        # ----------------------------------------------------
        if stylecond is not None:
            style_cond_processed = node_helpers.conditioning_set_values(stylecond, style_inject_dict, append=False)

            cond_final = cond_final + style_cond_processed

        return IO.NodeOutput(cond_final)

#----------------------------------------

class SimpleMaskAreaPrep(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] > 4 and arr.shape[3] <= 4:
            # (B, H, W, C) -> (B, C, H, W)
            arr = arr.permute(0, 3, 1, 2)
        elif arr.shape[1] > 4 and arr.shape[2] <= 4:
            # (B, C, H, W)
            arr = arr.permute(0, 2, 1, 3)
            
        return arr.float()

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "invert_check": "mask invert setting check...",
            "changedim": "change ndim...",
        }
        print(f"{CYAN}{BOLD}[AlphaMaskPrep]{RESET} {stages[stage]}")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SimpleMaskAreaPrep",
            display_name="간이 마스크영역 준비기",
            category="커스텀임베딩/샘플링",
            description="x축 y축을 입력하고 배치 좌표를 입력하여 마스크 프롬프트에서 사용가능한 사각형 마스크영역을 만듭니다.",
            inputs=[
                IO.Image.Input("target_image", tooltip="캔버스 준비용 이미지"),
                IO.Int.Input("mask_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크의 X축 길이"),
                IO.Int.Input("mask_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크의 Y축 길이"),
                IO.Int.Input("left_x", default=1, min=0, max=2048, step=1, tooltip="마스크를 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 이미지보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top_y", default=1, min=0, max=2048, step=1, tooltip="마스크를 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 이미지보다 작아질 경우 여백은 보정됩니다."),
                IO.Combo.Input("set_mask", options=["default","invert"], default="default", tooltip="마스크 반전 옵션"),
                IO.Combo.Input("set_maskdim", options=["2dim","3dim","4dim"], default="4dim", tooltip="마스크 ndim 세팅"),
                IO.Combo.Input("output_switch", options=["dict_mask","mask_tensor"], default="dict_mask", tooltip="마스크 출력 옵션"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 위에 마스크 영역을 오버레이하여 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("noise_mask", tooltip="생성된 간이 마스크"),
            ],
        )

    @classmethod
    def execute(cls, target_image, mask_x, mask_y, left_x, top_y, set_mask="default", set_maskdim="4dim", output_switch="dict", preview_mode=False) -> IO.NodeOutput:

        cls.show_data_preparation_progress("start")
        total_steps = 3
        pbar = comfy.utils.ProgressBar(int(total_steps))        
        progressbar_update(pbar, 0, total_steps)

        img_tensor = cls.ensure_image_tensor(target_image)
        _, _, img_h, img_w = img_tensor.shape
        box_x = min(max(mask_x, 1), 2048)
        box_y = min(max(mask_y, 1), 2048)
        l_x = min(max(left_x, 0), 2048)
        t_y = min(max(top_y, 0), 2048)
        r_x = img_w - box_x -l_x
        b_y = img_h - box_y -t_y

        canvas = torch.zeros((1, img_h, img_w), dtype=torch.float32)

        canvas[:, t_y:t_y+box_y, l_x:l_x+box_x] = 1.0
        
        mask = canvas
        cls.show_data_preparation_progress("invert_check")
        progressbar_update(pbar, 1, total_steps)

        if set_mask == "invert":
            mask = 1.0 - mask

        progressbar_update(pbar, 2, total_steps)
        cls.show_data_preparation_progress("changedim")
        if set_maskdim == "2dim": # [H, W]
            mask = mask.squeeze()
        elif set_maskdim == "3dim": # [B, H, W]
            mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
        else: # 4dim [B, 1, H, W]
            pass

        progressbar_update(pbar, 3, total_steps)
        overlay_result = img_tensor.clone()

        if output_switch == "dict_mask":
            if preview_mode:
                alpha = 0.4
                color_rgb = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=img_tensor.device).view(1, 3, 1, 1)
                raw_mask_idx = canvas.expand_as(img_tensor)
                overlay_result = img_tensor * (1 - raw_mask_idx * alpha) + color_rgb * (raw_mask_idx * alpha)
                overlay_result = overlay_result.clamp(0, 1)

                preview_img_permuted = overlay_result.permute(0, 2, 3, 1)
                return IO.NodeOutput({"noise_mask": mask},ui=UI.PreviewImage(preview_img_permuted))
            return IO.NodeOutput({"noise_mask": mask},)
        if preview_mode:
            alpha = 0.4
            color_rgb = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=img_tensor.device).view(1, 3, 1, 1)
            raw_mask_idx = canvas.expand_as(img_tensor)
            overlay_result = img_tensor * (1 - raw_mask_idx * alpha) + color_rgb * (raw_mask_idx * alpha)
            overlay_result = overlay_result.clamp(0, 1)

            preview_img_permuted = overlay_result.permute(0, 2, 3, 1)
            return IO.NodeOutput(mask,ui=UI.PreviewImage(preview_img_permuted))
        return IO.NodeOutput(mask,)

#----------------------------------------

class AutoMaskPrep(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] > 4 and arr.shape[3] <= 4:
            # (B, H, W, C) -> (B, C, H, W)
            arr = arr.permute(0, 3, 1, 2)
        elif arr.shape[1] > 4 and arr.shape[2] <= 4:
            # (B, C, H, W)
            arr = arr.permute(0, 2, 1, 3)
            
        return arr.float()

    @classmethod
    def ensure_image_nhwc(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            # (B, C, H, W) -> (B, H, W, C)
            arr = arr.permute(0, 2, 3, 1)
            
        return arr.float()

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "model_check": "model check...",
            "mask_settings": "Waiting for response...",
            "changedim": "change ndim...",
        }
        print(f"{CYAN}{BOLD}[AutoMaskPrep]{RESET} {stages[stage]}")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AutoMaskPrep",
            display_name="오토 마스크 준비기",
            category="커스텀임베딩/샘플링",
            description="입력 이미지를 모델을 참고해서 앰플리피케이션, dim 정리를 해서 다른 노드에 바로 사용가능한 상태로 만듭니다.",
            inputs=[
                IO.Image.Input("target_image", tooltip="마스크를 준비할 이미지"),
                IO.BackgroundRemoval.Input("bg_removal_model", tooltip="마스크 생성에 사용할 배경 제거 모델.모델이 있는 경우 해당 모델을 기준으로 배경영역 마스크를 준비합니다.", optional=True),
                IO.Mask.Input("guide_mask", tooltip="외부에서 가져온 가이드 마스크 (선택 사항. 모델이 연결되어 있을 경우 무시됩니다.)", optional=True),
                IO.Combo.Input("set_mask", options=["default","invert"], default="default", tooltip="마스크 반전 옵션"),
                IO.Combo.Input("area_adjust", options=["Heavy_Shrink", "Shrink", "Light_Shrink", "Default", "Light_Dilate", "Dilate", "Heavy_Dilate"], 
                               default="Default", tooltip="마스크 영역을 수축하거나 팽창하여 다듬습니다."),
                IO.Int.Input("min_boxsize", default=50, min=0, max=512, step=1, tooltip="OpenCV 박스 추출 시 무시할 최소 크기.(노이즈 제거용. 모델이 연결되어 있을 경우 무시됩니다.)\n"
                             "검출을 위해선 이미지의 크기보다 작아야 합니다."),
                IO.Float.Input("amp_setting", default=1.000, min=0.000, max=1.999, step=0.001, tooltip="앰플리피케이션 적용치"),
                IO.Combo.Input("set_maskdim", options=["2dim","3dim","4dim"], default="4dim", tooltip="마스크 ndim 세팅"),
                IO.Float.Input("alpha_setting", default=1.0, min=0.0, max=1.0, step=0.1, tooltip="알파 강도 적용치"),
                IO.Boolean.Input("binary_switch", default=True, tooltip="마스크 경계를 깔끔하게 0과 1의 이진 데이터로 고정할지를 확인합니다."),
                IO.Combo.Input("output_switch", options=["dict_mask","mask_tensor"], default="mask_tensor", tooltip="마스크 출력 옵션"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo], 
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask", tooltip="마스크"),
            ],
        )

    @classmethod
    def execute(cls, target_image, bg_removal_model=None, guide_mask=None, set_mask="default", area_adjust="default", min_boxsize=50, amp_setting=1.000, set_maskdim="4dim", alpha_setting=1.0, binary_switch=True, output_switch="mask_tensor", show_preview=False) -> IO.NodeOutput:

        cls.show_data_preparation_progress("start")
        total_steps = 5
        pbar = comfy.utils.ProgressBar(int(total_steps))        
        progressbar_update(pbar, 0, total_steps)

        image = cls.ensure_image_nhwc(target_image)
        target_B, target_H, target_W, target_C = image.shape

        min_size = min(max(min_boxsize, 0), 512)

        cls.show_data_preparation_progress("model_check")      
        progressbar_update(pbar, 1, total_steps)

        if bg_removal_model is not None:
            if guide_mask is not None:
                print(f"{CYAN}{BOLD}[AutoMaskPrep]{RESET} Since bg_removal_model is connected, guide_mask is ignored.")
                pass
            image_sample = image.permute(0, 2, 3, 1)
            samp_mask = bg_removal_model.encode_image(image_sample)
            samp_mask = samp_mask.unsqueeze(1)
            if samp_mask.dim() == 4:
                if samp_mask.shape[3] <= 4 and samp_mask.shape[3] < samp_mask.shape[1] and samp_mask.shape[3] < samp_mask.shape[2]:
                    samp_mask = samp_mask.permute(0, 3, 1, 2)
                else:
                    pass

        else:
            if guide_mask is not None:
                samp_mask = ensure_mask_tensor(guide_mask)
                if samp_mask.dim() == 3:
                    samp_mask = samp_mask.unsqueeze(1)
                if samp_mask.shape[-2:] != (target_H, target_W):
                    samp_mask = F.interpolate(samp_mask.float(), size=(target_H, target_W), mode="nearest")

            else:
                img_np = (target_image[0].cpu().numpy() * 255).astype(np.uint8)
                image_file = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
                H, W, C = image_file.shape

                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = np.ones((5, 5), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                sure_bg = cv2.dilate(opening, kernel, iterations=3)

                contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                mask_canvas = torch.zeros((1, H, W), dtype=torch.float32)

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < min_size or h < min_size:
                        continue
                    cv2.drawContours(mask_canvas[0].numpy(), [cnt], -1, 1.0, thickness=cv2.FILLED)
            
                samp_mask = ensure_mask_tensor(mask_canvas)

        cls.show_data_preparation_progress("mask_settings")      
        progressbar_update(pbar, 2, total_steps)
        mask = ensure_mask_tensor(samp_mask)

        amp_set = min(max(amp_setting, 0.000), 1.999)
        alpha_set = min(max(alpha_setting, 0.0), 1.0)

        if area_adjust != "Default":
            # kernel pixel size (Heavy: 3px, Normal: 2px, Light: 1px)
            if area_adjust == "Heavy_Shrink":
                k_size, k_pad, op = 7, 3, "shrink"
            elif area_adjust == "Shrink":
                k_size, k_pad, op = 5, 2, "shrink"
            elif area_adjust == "Light_Shrink":
                k_size, k_pad, op = 3, 1, "shrink"
            elif area_adjust == "Light_Dilate":
                k_size, k_pad, op = 3, 1, "dilate"
            elif area_adjust == "Dilate":
                k_size, k_pad, op = 5, 2, "dilate"
            elif area_adjust == "Heavy_Dilate":
                k_size, k_pad, op = 7, 3, "dilate"
            else:
                op = "none"

            if op == "dilate":
                mask = F.max_pool2d(mask, kernel_size=k_size, stride=1, padding=k_pad)
            elif op == "shrink":
                mask = -F.max_pool2d(-mask, kernel_size=k_size, stride=1, padding=k_pad)
            
            mask = torch.clamp(mask, 0.0, 1.0)

        if amp_setting != 1.0:
            mask = mask * amp_set

        if binary_switch:
            mask = (mask >= threshold_val).float()
            mask = torch.clamp(mask, 0.0, 1.0)

        if set_mask == "invert":
            mask = 1.0 - mask

        if alpha_setting != 1.0:
            mask = mask * alpha_set

        progressbar_update(pbar, 3, total_steps)
        cls.show_data_preparation_progress("changedim")
        if set_maskdim == "2dim": # [H, W]
            mask = mask.squeeze()
        elif set_maskdim == "3dim": # [B, H, W]
            mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
        else: # 4dim [B, 1, H, W]
            pass

        progressbar_update(pbar, 4, total_steps)
        if output_switch == "dict_mask":
            return IO.NodeOutput({"noise_mask": mask})
        if show_preview:
            preview_mask = ensure_mask_tensor(mask).repeat(1, 3, 1, 1)
            preview_mask = cls.ensure_image_nhwc(preview_mask)
            return IO.NodeOutput(mask, ui=UI.PreviewImage(preview_mask))
        else:
            return IO.NodeOutput(mask)

#----------------------------------------

class AutoMaskPrep_ADV(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] > 4 and arr.shape[3] <= 4:
            # (B, H, W, C) -> (B, C, H, W)
            arr = arr.permute(0, 3, 1, 2)
        elif arr.shape[1] > 4 and arr.shape[2] <= 4:
            # (B, C, H, W)
            arr = arr.permute(0, 2, 1, 3)
            
        return arr.float()

    @classmethod
    def ensure_image_nhwc(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            # (B, C, H, W) -> (B, H, W, C)
            arr = arr.permute(0, 2, 3, 1)
            
        return arr.float()

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "model_check": "model check...",
            "mask_settings": "Waiting for response...",
            "changedim": "change ndim...",
        }
        print(f"{CYAN}{BOLD}[AutoMaskPrep]{RESET} {stages[stage]}")

    @classmethod
    def apply_anti_aliasing(cls, mask: torch.Tensor, blur_sigma: float = 1.0) -> torch.Tensor:

        device = mask.device
    
        # 1. 배치 및 채널 차원을 떼어내어 2차원 [H, W] numpy 배열로 변환
        # (B, C, H, W) 또는 (1, 1, H, W) 구조에 유연하게 대응
        mask_np = mask.detach().cpu().squeeze().numpy()
    
        if mask_np.ndim != 2:
            # 차원이 예상치 못하게 꼬인 경우를 대비한 안전 장치
            return mask

        # 2. 커널 사이즈 계산 (sigma 값에 비례하여 홀수 크기로 자동 설정)
        k_size = int(blur_sigma * 3) * 2 + 1
    
        # 3. 가우시안 블러 적용 (경계를 부드러운 그라데이션으로 만듦)
        blurred = cv2.GaussianBlur(mask_np, (k_size, k_size), blur_sigma)
    
        # 4. 외곽의 너무 퍼진 잔상을 잡아주기 위한 콘트라스트(Smoothstep) 보정
        # 0.2 ~ 0.8 구간의 경사도를 가파르게 만들어 중심부는 진하게, 외곽은 부드럽게 유지
        # (선택 사항이지만 이 과정을 거치면 뭉개짐 없이 칼같은 안티앨리어싱이 됨)
        min_val, max_val = blurred.min(), blurred.max()
        if max_val > min_val:
            normalized = (blurred - min_val) / (max_val - min_val)
            # S자 커브(Smoothstep)로 경계면 대비 극대화
            smoothed = normalized * normalized * (3.0 - 2.0 * normalized)
            final_np = min_val + smoothed * (max_val - min_val)
        else:
            final_np = blurred

        # 5. 다시 원래의 텐서 형태와 디바이스로 복구
        final_tensor = torch.from_numpy(final_np).float().to(device)
    
        # 원래 가지고 있던 차원 구조로 복원 (예: [1, 1, H, W])
        while final_tensor.dim() < mask.dim():
            final_tensor = final_tensor.unsqueeze(0)
        
        return final_tensor
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AutoMaskPrep_ADV",
            display_name="오토 마스크 준비기(고급)",
            category="커스텀임베딩/샘플링",
            description="입력 이미지를 참고하여 마스크를 자동 추출하고 다듬습니다. 중립 회색 전처리 옵션이 추가되었습니다.",
            inputs=[
                IO.Image.Input("target_image", tooltip="마스크를 준비할 이미지 (NHWC)"),
                IO.BackgroundRemoval.Input("bg_removal_model", tooltip="마스크 생성에 사용할 배경 제거 모델 (선택)", optional=True),
                IO.Mask.Input("guide_mask", tooltip="외부에서 가져온 가이드 마스크 (선택 사항)", optional=True),
                IO.Float.Input("neutral_blend", default=0.0, min=0.0, max=1.0, step=0.1, tooltip="마스크 추출 전 원본에 중립 회색을 섞어 과도한 광원/색상 간섭을 짓누릅니다 (0: 원본 유지, 0.5~0.7: 추천)"),
                IO.Combo.Input("set_mask", options=["default","invert"], default="default", tooltip="마스크 반전 옵션"),
                IO.Combo.Input("area_adjust", options=["Heavy_Shrink", "Shrink", "Light_Shrink", "Default", "Light_Dilate", "Dilate", "Heavy_Dilate"], 
                               default="Default", tooltip="마스크 영역을 수축하거나 팽창하여 다듬습니다."),
                IO.Int.Input("min_boxsize", default=50, min=0, max=512, step=1, tooltip="OpenCV 박스 추출 시 무시할 최소 크기"),
                IO.Float.Input("amp_setting", default=1.000, min=0.000, max=1.999, step=0.001, tooltip="앰플리피케이션 적용치"),
                IO.Combo.Input("set_maskdim", options=["2dim","3dim","4dim"], default="4dim", tooltip="마스크 ndim 세팅"),
                IO.Float.Input("alpha_setting", default=1.0, min=0.0, max=1.0, step=0.1, tooltip="알파 강도 적용치"),
                IO.Boolean.Input("binary_switch", default=True, tooltip="마스크 경계를 0과 1의 이진 데이터로 고정"),
                IO.Combo.Input("output_switch", options=["dict_mask","mask_tensor"], default="mask_tensor", tooltip="마스크 출력 옵션"),
                IO.Float.Input("threshold_val", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="마스크 판정을 내릴 임계값 (0에 가까울수록 넓게, 1에 가까울수록 좁게 잡음)"),
                IO.Boolean.Input("antialiasing", default=False, tooltip="안티앨리어싱 마스크. 경계부가 깨지거나 흐릿할 경우 씁니다."),
                IO.Float.Input("blur_sigma", default=1.0, min=0.1, max=5.0, step=0.1, tooltip="마스크 외곽의 계단 현상을 부드럽게 다듬는 안티 앨리어싱 강도. 안티앨리어싱 스위치가 켜져야만 적용됩니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Mask.Output("mask", tooltip="처리된 최종 마스크"),
            ],
        )

    @classmethod
    def execute(cls, target_image, bg_removal_model=None, guide_mask=None, neutral_blend=0.0, set_mask="default", area_adjust="default", min_boxsize=50, amp_setting=1.000,
                set_maskdim="4dim", alpha_setting=1.0, binary_switch=True, output_switch="mask_tensor", threshold_val=0.5, antialiasing=False, blur_sigma=1.0, show_preview=False) -> IO.NodeOutput:


        cls.show_data_preparation_progress("start")
        total_steps = 5
        pbar = comfy.utils.ProgressBar(int(total_steps))        
        progressbar_update(pbar, 0, total_steps)

        image = cls.ensure_image_nhwc(target_image)
        target_B, target_H, target_W, target_C = image.shape

        min_size = min(max(min_boxsize, 0), 512)

        if neutral_blend > 0.0:
            solid_gray = torch.full_like(image, 0.5)
            # Blends the original image with neutral gray (0.5) to flatten color saturation and light source noise
            prep_image = image * (1.0 - neutral_blend) + solid_gray * neutral_blend
        else:
            prep_image = image

        cls.show_data_preparation_progress("model_check")      
        progressbar_update(pbar, 1, total_steps)

        if bg_removal_model is not None:
            if guide_mask is not None:
                print(f"{CYAN}{BOLD}[AutoMaskPrep]{RESET} Since bg_removal_model is connected, guide_mask is ignored.")
                pass

            samp_mask = bg_removal_model.encode_image(prep_image)
            samp_mask = samp_mask.unsqueeze(1)
            if samp_mask.dim() == 4:
                if samp_mask.shape[3] <= 4 and samp_mask.shape[3] < samp_mask.shape[1] and samp_mask.shape[3] < samp_mask.shape[2]:
                    samp_mask = samp_mask.permute(0, 3, 1, 2)
                else:
                    pass

        else:
            if guide_mask is not None:
                samp_mask = ensure_mask_tensor(guide_mask)
                if samp_mask.dim() == 3:
                    samp_mask = samp_mask.unsqueeze(1)
                if samp_mask.shape[-2:] != (target_H, target_W):
                    samp_mask = F.interpolate(samp_mask.float(), size=(target_H, target_W), mode="nearest")

            else:
                img_np = (prep_image[0].cpu().numpy() * 255).astype(np.uint8)
                image_file = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
                H, W, C = image_file.shape

                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = np.ones((5, 5), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                sure_bg = cv2.dilate(opening, kernel, iterations=3)

                contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                mask_canvas = torch.zeros((1, H, W), dtype=torch.float32)

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < min_size or h < min_size:
                        continue
                    cv2.drawContours(mask_canvas[0].numpy(), [cnt], -1, 1.0, thickness=cv2.FILLED)
            
                samp_mask = ensure_mask_tensor(mask_canvas)


        cls.show_data_preparation_progress("mask_settings")      
        progressbar_update(pbar, 2, total_steps)
        mask = ensure_mask_tensor(samp_mask)

        amp_set = min(max(amp_setting, 0.000), 1.999)
        alpha_set = min(max(alpha_setting, 0.0), 1.0)

        if area_adjust != "Default":
            # kernel pixel size (Heavy: 3px, Normal: 2px, Light: 1px)
            if area_adjust == "Heavy_Shrink":
                k_size, k_pad, op = 7, 3, "shrink"
            elif area_adjust == "Shrink":
                k_size, k_pad, op = 5, 2, "shrink"
            elif area_adjust == "Light_Shrink":
                k_size, k_pad, op = 3, 1, "shrink"
            elif area_adjust == "Light_Dilate":
                k_size, k_pad, op = 3, 1, "dilate"
            elif area_adjust == "Dilate":
                k_size, k_pad, op = 5, 2, "dilate"
            elif area_adjust == "Heavy_Dilate":
                k_size, k_pad, op = 7, 3, "dilate"
            else:
                op = "none"

            if op == "dilate":
                mask = F.max_pool2d(mask, kernel_size=k_size, stride=1, padding=k_pad)
            elif op == "shrink":
                mask = -F.max_pool2d(-mask, kernel_size=k_size, stride=1, padding=k_pad)
            
            mask = torch.clamp(mask, 0.0, 1.0)

        if amp_setting != 1.0:
            mask = mask * amp_set

        if binary_switch:
            mask = torch.clamp(mask.round(), 0.0, 1.0)

        if antialiasing:
            mask = cls.apply_anti_aliasing(mask, blur_sigma)

        if set_mask == "invert":
            mask = 1.0 - mask

        if alpha_setting != 1.0:
            mask = mask * alpha_set

        progressbar_update(pbar, 3, total_steps)
        cls.show_data_preparation_progress("changedim")
        if set_maskdim == "2dim": # [H, W]
            mask = mask.squeeze()
        elif set_maskdim == "3dim": # [B, H, W]
            mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
        else: # 4dim [B, 1, H, W]
            pass

        progressbar_update(pbar, 4, total_steps)
        if output_switch == "dict_mask":
            return IO.NodeOutput({"noise_mask": mask})
        if show_preview:
            preview_mask = ensure_mask_tensor(mask).repeat(1, 3, 1, 1)
            preview_mask = cls.ensure_image_nhwc(preview_mask)
            return IO.NodeOutput(mask, ui=UI.PreviewImage(preview_mask))
        else:
            return IO.NodeOutput(mask)

#----------------------------------------

class AutobboxMaskPrep(IO.ComfyNode):

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "model_check": "model check...",
            "mask_settings": "Waiting for response...",
            "changedim": "change ndim...",
        }
        print(f"{CYAN}{BOLD}[AutobboxMaskPrep]{RESET} {stages[stage]}")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AutobboxMaskPrep",
            display_name="오토 bbox마스크 준비기",
            category="커스텀임베딩/샘플링",
            description="입력 마스크를 기준으로 바운딩 박스를 모드에 따라 준비해 바로 사용가능한 상태로 만듭니다.",
            inputs=[
                IO.Mask.Input("target_mask", tooltip="바운딩 박스를 준비할 대상"),
                IO.Mask.Input("guide_mask", tooltip="외부에서 가져온 가이드 마스크 (선택 사항. 모델이 연결되어 있을 경우 무시됩니다.)", optional=True),
                IO.Combo.Input("set_mask", options=["default","invert"], default="default", tooltip="마스크 반전 옵션"),
                IO.Int.Input("min_boxsize", default=50, min=0, max=512, step=1, tooltip="OpenCV 박스 추출 시 무시할 최소 크기.(노이즈 제거용. 모델이 연결되어 있을 경우 무시됩니다.)\n"
                             "검출을 위해선 이미지의 크기보다 작아야 합니다."),
                IO.Combo.Input("bbox_mode", options=["default","findContours","edgelines"], default="default", tooltip="바운딩 마스크 준비 모드"),
                IO.Combo.Input("output_switch", options=["dict_mask","mask_tensor"], default="mask_tensor", tooltip="마스크 출력 옵션"),
            ],
            outputs=[
                IO.Mask.Output("mask", tooltip="마스크"),
            ],
        )

    @classmethod
    def execute(cls, target_mask, guide_mask=None, set_mask="default", min_boxsize=50, bbox_mode="default", output_switch="mask_tensor") -> IO.NodeOutput:

        cls.show_data_preparation_progress("start")
        total_steps = 5
        pbar = comfy.utils.ProgressBar(int(total_steps))        
        progressbar_update(pbar, 0, total_steps)

        samp_mask = ensure_mask_tensor(target_mask)
        target_B, target_C, target_H, target_W = samp_mask.shape

        min_size = min(max(min_boxsize, 0), 512)

        cls.show_data_preparation_progress("model_check")      
        progressbar_update(pbar, 1, total_steps)

        if guide_mask is not None:
            guide_tensor = ensure_mask_tensor(guide_mask)
            if guide_tensor.dim() == 3:
                guide_tensor = guide_tensor.unsqueeze(1)
            if samp_mask.shape[-2:] != guide_tensor.shape[-2:]:
                guide_tensor = F.interpolate(guide_tensor.float(), size=(target_H, target_W), mode="nearest")
            samp_mask = samp_mask* guide_tensor


        cls.show_data_preparation_progress("mask_settings")      
        progressbar_update(pbar, 2, total_steps)
        mask = ensure_mask_tensor(samp_mask)

        mask_np = samp_mask.squeeze().cpu().numpy()
        bbox_canvas = torch.zeros_like(samp_mask)

        if bbox_mode == "findContours":
            img_uint8 = (mask_np * 255).astype(np.uint8)
            contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w >= min_size and h >= min_size:
                    bbox_canvas[:, :, y:y+h, x:x+w] = 1.0

        elif bbox_mode == "edgelines":
            edges = cv2.Canny((mask_np * 255).astype(np.uint8), 100, 200)
            y_indices, x_indices = np.where(edges > 0)
            if len(y_indices) == 0:  # If the edge isn’t properly locked, fallback to the normal mask coordinates
                y_indices, x_indices = np.where(mask_np > 0.1)
                
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                box_h = y_max - y_min
                box_w = x_max - x_min
                if box_w >= min_size and box_h >= min_size:
                    bbox_canvas[:, :, y_min:y_max+1, x_min:x_max+1] = 1.0

        else:
            y_indices, x_indices = np.where(mask_np > 0.1)
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                box_h = y_max - y_min
                box_w = x_max - x_min
            
                if box_w >= min_size and box_h >= min_size:
                    bbox_canvas[:, :, y_min:y_max+1, x_min:x_max+1] = 1.0

        mask = ensure_mask_tensor(bbox_canvas)
            
        if set_mask == "invert":
            mask = 1.0 - mask

        progressbar_update(pbar, 4, total_steps)
        if output_switch == "dict_mask":
            return IO.NodeOutput({"noise_mask": mask})
        return IO.NodeOutput(mask)

#----------------------------------------
#embed Node Settings
#----------------------------------------

class EasyEmbedLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        files = []
        for d in embedding_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith((".bin", ".pt")):

                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)


        return IO.Schema(
            node_id="EasyEmbedLoader",
            display_name="BIN PT 임베딩 로더",
            category="커스텀임베딩/임베딩",
            description=".bin 또는 .pt 포맷 임베딩을 불러옵니다. 필요시 변환 노드에서 다른 포맷으로 저장하세요.\n"
                         "이 노드는 레이어 연결용이 아닙니다. 변환노드에만 연결하세요.\n"
                         "절대 체크포인트 통합모델을 집어넣지 마세요. 오버플로우로 블루스크린이 터질 수 있습니다.",
            inputs=[
                IO.Combo.Input("embed_name", options=files, default=files[0] if files else "",
                               tooltip="불러올 임베딩 파일 이름을 선택하세요"),
                IO.Combo.Input("with_name", options=["off","on"], default="off",
                               tooltip="파일명 출력 여부"),
            ],
            outputs=[
                embedload.Output("EMBEDS", tooltip="임베딩 객체"),
                IO.String.Output("embed_name", tooltip="불러온 파일 이름 (옵션)")
            ],
        )
    @classmethod
    def execute(cls, embed_name, with_name="off") -> IO.NodeOutput:
        embedding_dirs = folder_paths.get_folder_paths("embeddings")

        file_path = None
        for d in embedding_dirs:
            candidate = os.path.join(d, embed_name)  
            if os.path.exists(candidate):
                file_path = candidate
                break

        if file_path is None:
            raise FileNotFoundError(f"Embedding {embed_name}(.bin/.pt) not found in {embedding_dirs}")

        state_dict = torch.load(file_path, map_location="cpu")

        if with_name == "on":
            embed = os.path.splitext(embed_name)[0]
            return IO.NodeOutput(state_dict, embed)
        else:
            return IO.NodeOutput(state_dict, "")



        
#----------------------------------------

class EasyEmbedTransform(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyEmbedTransform",
            display_name="BIN/PT → EMBEDDING 변환",
            category="커스텀임베딩/임베딩",
            description=".bin 또는 .pt 포맷 임베딩을 safetensors로 변환하여 proj_embeddings에 저장합니다.\n"
                        "용량이 너무 적으면 에러방지용 더미텐서가 출력된겁니다.",
            inputs=[
                embedload.Input("EMBEDS", tooltip="BIN/PT 포맷 임베딩 객체"),
                IO.String.Input("file_name", default="easynegative",
                                tooltip="저장할 파일 이름 (확장자 제외)"),
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, EMBEDS, file_name="easynegative"):

        tensor = None
        if isinstance(EMBEDS, dict):
            if "embedding" in EMBEDS:
                tensor = EMBEDS["embedding"]
            elif "string_to_param" in EMBEDS:
                tensor = next(iter(EMBEDS["string_to_param"].values()))
            elif "params" in EMBEDS:
                tensor = EMBEDS["params"]
        elif isinstance(EMBEDS, torch.Tensor):
            tensor = EMBEDS

        metadata = {}
        if tensor is None or not isinstance(tensor, torch.Tensor):
            try:
                tensor = torch.tensor(EMBEDS)
            except Exception:
                tensor = torch.zeros((1, 1))
                metadata = {"warning": "에러 방지용 더미 텐서가 저장되었습니다"}
                print(f"{CYAN}{BOLD}[EasyEmbedTransform]{RESET} 경고: 에러 방지용 더미 텐서입니다."
                      "노드를 재점검하는걸 권장합니다.")

        if isinstance(tensor, torch.Tensor) and tensor.numel() >= 10:
            tokens = tensor.shape[0] if len(tensor.shape) > 1 else 1
            dim = tensor.shape[-1]

            if dim == 512:
                style = "Stable Diffusion 1.x"
            elif dim == 768:
                style = "Stable Diffusion 2.x"
            elif 1024 <= dim < 2000:
                style = "Stable Diffusion 3.x / T5"
            elif dim >= 2000:
                style = "Qwen / Large-scale model"
            else:
                style = "Unknown"

            metadata = {
                "embed_structure": "dict",
                "embed_shape": f"tokens={tokens}, dim={dim}",
                "tokens": str(tokens),
                "dim": str(dim),
                "embed_style": style
            }


        proj_dir = os.path.join(folder_paths.base_path, "models", "proj_embeddings")
        os.makedirs(proj_dir, exist_ok=True)
        folder_paths.folder_names_and_paths["proj_embeddings"] = (
            [proj_dir],
            folder_paths.supported_pt_extensions
        )

        out_path = os.path.join(proj_dir, f"{file_name}.safetensors")
        save_file({"embedding": tensor}, out_path, metadata=metadata)
        print(f"{CYAN}{BOLD}[EasyEmbedTransform]{RESET} saved embedding at {out_path}")

        return IO.NodeOutput(True)


#----------------------------------------


class EasyencoderChecker(IO.ComfyNode):

    @classmethod
    def find_tensor(cls, obj):
        if isinstance(obj, dict):
            for obj_v in obj.values():
                if isinstance(obj_v, torch.Tensor):
                    return obj_v
        elif isinstance(obj, torch.Tensor):
            return obj
        return None

    @classmethod
    def define_schema(cls):
        clip_dirs = folder_paths.get_folder_paths("clip")
        files = []
        for d in clip_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith((".pt", ".safetensors", ".bin")):

                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)

        return IO.Schema(
            node_id="EasyencoderChecker",
            display_name="CLIP 인코더 체크(시각화)",
            category="커스텀임베딩/특수",
            description="CLIP 인코더의 타입과 차원을 텍스트로 시각화합니다.",
            inputs=[
                IO.Combo.Input("clip_name", options=files, default=files[0] if files else "", tooltip="확인할 텍스트 인코더 파일"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="폰트 크기. 텍스트 리스트 모드일 때는 스킵됩니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="노드 위젯으로 바로 보고 싶다면 켜세요."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.String.Output("text", tooltip="CLIP 인코더 구조 텍스트 리스트")
            ],
        )

    @classmethod
    def execute(cls, clip_name, font_size=14, show_preview=False) -> IO.NodeOutput:
        clip_dirs = folder_paths.get_folder_paths("clip")
        file_name = clip_name
        clip_path = None
        tensor = None
        clip_type = "Unknown"
        clip_structure = "Unknown"
        clip_style = "Unknown"
        clip_tokensize = "Unknown"
        est_vram = "Unknown"
        vision_status = "Clip_Vision: Unknown"
        found_attn_keys = {}
        found_special_keys = []
        
        for d in clip_dirs:
            path = os.path.join(d, clip_name)
            if os.path.exists(path):
                if path.endswith(".pt"):
                    loaded = torch.load(path, map_location="cpu")
                    tensor = find_tensor(loaded)
                    clip_type = "PT"
                    clip_structure = type(loaded).__name__
                elif path.endswith(".safetensors"):
                    tensors = load_file(path)
                    tensor = find_tensor(tensors)
                    loaded = tensors
                    clip_type = "Safetensors"
                    clip_structure = type(tensors).__name__
                elif path.endswith(".bin"):
                    loaded = torch.load(path, map_location="cpu")
                    tensor = find_tensor(loaded)
                    clip_type = "BIN"
                    clip_structure = type(loaded).__name__
                break

        vision_keys = ["vision_model", "patch_embedding", "visual_projection", "conv1.weight", "class_embedding"]
        attn_keywords = ["qkv", "to_q", "to_k", "to_v", "query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "attn", "_q", "_k", "_v"]

        if tensor is not None:
            tokens = tensor.shape[0] if len(tensor.shape) > 1 else 1
            dim = tensor.shape[-1]
            
            clip_tokensize = f"tokens={tokens}, dim={dim}"
            if dim >= 2000 or tokens >= 1000:
                clip_style = "Flux / Wan / Qwen / Large-scale model"
            elif 1024 <= dim < 2000 or tokens >= 200:
                clip_style = "SD 3.x / SDXL / T5"
            elif dim == 768:
                clip_style = "SD 2.x"
            elif dim == 512:
                clip_style = "SD 1.x"
            else:
                clip_style = "Unknown / Custom"

            vision_support = any(any(vk in key for vk in vision_keys) for key in loaded)
            vision_status = "Clip_Vision: YES" if vision_support else "Clip_Vision: NO"

        layer_indices = []
        if isinstance(loaded, dict):
            for name in loaded.keys():
                if "layers." in name:
                    try:
                        idx_str = name.split("layers.")[1].split(".")[0]
                        idx = int(idx_str)
                        layer_indices.append(idx)
                    except ValueError:
                        continue

            if isinstance(loaded, dict):
                for name in loaded.keys():
                    name_lower = name.lower()
                    for kw in attn_keywords:
                        if kw in name_lower:
                            if kw not in found_attn_keys:
                                found_attn_keys[kw] = name
                            break
                    if len(found_attn_keys) >= 8:
                        break

            if not found_attn_keys:
                raw_lines.append("DEBUG: Raw Key Sample:")
                for debug_key in loaded[:5]:
                    raw_lines.append(f" > {debug_key[:40]}")

        file_size_gb = 0.0

        if clip_path:
            file_name = os.path.basename(clip_path)
            file_size_bytes = os.path.getsize(clip_path)
            file_size_gb = file_size_bytes / (1024 ** 3)

            vram_gb = file_size_gb * 1.3
            est_vram = f"~{vram_gb:.2f} GB (File: {file_size_gb:.2f} GB)"

        raw_lines = [
            f"File: {file_name}",
            f"Type: {clip_type}",
            f"Est. VRAM: {est_vram}",
            f"Spec: {clip_tokensize}",
            f"Style: {clip_style}",
            f"Status: {vision_status}"
        ]
        if layer_indices:
            min_layer = min(layer_indices)
            max_layer = max(layer_indices)
            layer_count = len(set(layer_indices))
            raw_lines.append(f"Layers: min={min_layer}, max={max_layer}, count={layer_count}")

        if found_attn_keys:
            raw_lines.append("Attn Keys (Found):")
            for pat, real_name in found_attn_keys.items():
                short_name = os.path.basename(real_name) if len(real_name) > 30 else real_name
                raw_lines.append(f"  - [{pat}] {short_name}")

        if not found_attn_keys:
            raw_lines.append("DEBUG: Raw Key Sample:")
            if isinstance(loaded, (list, tuple)):
                sample_target = loaded
            elif hasattr(loaded, "keys"):
                sample_target = list(loaded.keys())
            else:
                sample_target = []
            
            for debug_key in sample_target[:5]:
                raw_lines.append(f" > {str(debug_key)[:40]}")

        if found_special_keys:
            raw_lines.append("Special Keys (Found):")
            for sk_name in found_special_keys:
                short_sk = os.path.basename(sk_name) if len(sk_name) > 30 else sk_name
                raw_lines.append(f"  * {short_sk}")
        text_output_str = "\n".join(raw_lines)
        
        if show_preview:
            return IO.NodeOutput(text_output_str, ui={"stats":[text_output_str], "font_size":[font_size]})
        else:
            return IO.NodeOutput(text_output_str)

#----------------------------------------

class EasyEmbeddingChecker(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        files = []
        for d in embedding_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith((".pt", ".safetensors", ".bin")):

                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)
  
                    
        return IO.Schema(
            node_id="EasyEmbeddingChecker",
            display_name="임베딩 체크(시각화)",
            category="커스텀임베딩/특수",
            description="임베딩 파일 구조와 shape를 텍스트로 시각화합니다.",
            inputs=[
                IO.Combo.Input("embedding_name", options=files, default=files[0] if files else "", tooltip="확인할 임베딩 파일"),
                IO.Int.Input("font_size", default=32, min=20, max=48, tooltip="폰트 크기. 텍스트 리스트 모드일 때는 스킵됩니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="노드 위젯으로 바로 보고 싶다면 켜세요."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.String.Output("text", tooltip="임베딩 구조를 기록한 텍스트")
            ],
        )

    @classmethod
    def execute(cls, embedding_name, font_size=32, show_preview=False) -> IO.NodeOutput:
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        tensor = None
        embed_type = "Unknown"
        embed_structure = "Unknown"
        embed_style = "Unknown"
        embed_tokensize = "Unknown"

        for d in embedding_dirs:
            path = os.path.join(d, embedding_name) 
            if os.path.exists(path):
                if path.endswith(".pt"):
                    loaded = torch.load(path, map_location="cpu")
                    tensor = find_tensor(loaded)
                    embed_type = "PT"
                    embed_structure = type(loaded).__name__
                elif path.endswith(".safetensors"):
                    tensors = load_file(path)
                    tensor = find_tensor(tensors)
                    embed_type = "Safetensors"
                    embed_structure = type(tensors).__name__
                break

        if tensor is not None:
            tokens = tensor.shape[0] if len(tensor.shape) > 1 else 1
            dim = tensor.shape[-1]
            embed_tokensize = f"tokens={tokens}, dim={dim}"
            if dim >= 4000:
                embed_style = "Flux / Wan / Qwen / Large-scale model"
            elif 2048<=dim < 4000:
                embed_style = "SD3.x / SDXL / T5"
            elif 1024 <= dim < 2000:
                embed_style = "SD2.x"
            elif dim == 768:
                embed_style = "SD1.x"
            else:
                embed_style = "Unknown / Custom"

        raw_lines = [
            f"File: {embedding_name}",
            f"Embed Type: {embed_type}",
            f"Embed Structure: {embed_structure}",
            f"Embed Shape: {embed_tokensize}",
            f"Embed Style: {embed_style}"
        ]

        if embed_style == "Unknown / Custom":
            raw_lines.append("WARNING: unknown_style detected, may not work")
        text_output_str = "\n".join(raw_lines)

        if show_preview:

            return IO.NodeOutput(text_output_str, ui={"stats":[text_output_str], "font_size":[font_size]})

        else:
            return IO.NodeOutput(text_output_str)

#----------------------------------------

class EasyProjLayerChecker(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        files = []
        base_dirs = folder_paths.get_folder_paths("proj_embeddings")
        for d in base_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".safetensors"):

                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)

        return IO.Schema(
            node_id="EasyProjLayerChecker",
            display_name="프로젝션 레이어 체크(시각화)",
            category="커스텀임베딩/특수",
            description="프로젝션 레이어 파일의 차원/메타데이터를 확인하고 오류 여부를 시각화합니다.",
            inputs=[
                IO.Combo.Input("proj_file", options=files, default=files[0] if files else "", tooltip="확인할 프로젝션 레이어 파일"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="미리보기용 폰트 크기"),
                IO.Boolean.Input("show_preview", default=True, tooltip="노드 위젯으로 띄울지 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.String.Output("text", tooltip="프로젝션 레이어 구조 및 메타데이터 텍스트")
            ],
        )

    @classmethod
    def execute(cls, proj_file, font_size=14, show_preview=False) -> IO.NodeOutput:

        path = None
        for d in folder_paths.get_folder_paths("proj_embeddings"):
            candidate = os.path.join(d, proj_file)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f"{proj_file} not found")

        data = load_file(path)
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()

        weight = data.get("weight", None)
        bias = data.get("bias", None)

        try:
            in_dim = int(metadata.get("meta_in_dim", 0))
        except Exception:
            in_dim = weight.shape[1] if weight is not None else 0
        try:
            out_dim = int(metadata.get("meta_out_dim", 0))
        except Exception:
            out_dim = weight.shape[0] if weight is not None else 0

        clip_type = metadata.get("meta_type", "unknown")

        max_dim = max(in_dim, out_dim)
        if max_dim >= 4000:
            proj_style = "Flux / Wan / Qwen / Large-scale model"
        elif 2048 <= max_dim < 4000:
            proj_style = "SD3.x / SDXL / T5"
        elif 1024 <= max_dim < 2000:
            proj_style = "SD2.x"
        elif max_dim == 768:
            proj_style = "SD1.x"
        else:
            proj_style = "Unknown / Custom"

        warnings = []
        if (("meta_in_dim" not in metadata or "meta_out_dim" not in metadata) and 
            (weight is None or bias is None)):
            warnings.append("weight/bias missing (no metadata fallback)")
        if in_dim <= 0 or out_dim <= 0:
            warnings.append("invalid dimensions")
        if weight is not None and (weight.shape[1] != in_dim or weight.shape[0] != out_dim):
            warnings.append("dimension mismatch")
        if proj_style == "Unknown / Custom":
            warnings.append("unknown style detected")

        raw_lines = [
            f"File: {proj_file}",
            f"In Dim: {in_dim}",
            f"Out Dim: {out_dim}",
            f"Clip Type: {clip_type}",
            f"Proj Style: {proj_style}"
        ]

        if warnings:
            raw_lines.append("Warnings:")
            for w in warnings:
                raw_lines.append(f" ! {w}")

        text_output_str = "\n".join(raw_lines)

        if show_preview:
            return IO.NodeOutput(text_output_str, ui={"stats":[text_output_str], "font_size":[font_size]})
        else:
            return IO.NodeOutput(text_output_str)
        

#----------------------------------------
#Sampler Node Settings
#----------------------------------------

class EasySampler(IO.ComfyNode):

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "data selecting": "input data checking...",
            "waiting": "Waiting for response..."
        }
        print(f"{CYAN}{BOLD}[EasySampler]{RESET} {stages[stage]}")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasySampler",
            display_name="간단 샘플러",
            category="커스텀임베딩/샘플링",
            description="Noise 추가 여부, step 범위, leftover noise 처리 등을 제공하는 샘플러.\n"
                        "CPU 모드에서도 충돌을 최소화하며 latent를 초기화하고 샘플링을 수행합니다.\n"
                        "라데온 계열도 돌아가긴 하지만, ROCm 처리 조건이 부족하면 CPU모드로 기동될 수 있습니다.\n"
                        "사용된 시드 값은 CMD에 프린트 로그로 남아있습니다. 시드를 고정하려면 복사해서 붙여넣으면 됩니다.\n"
                        "[참고] 베이스 온리(Base Only) 모드에서는 시드가 고정되어 있어도 연산 특성상 결과물이 미세하게 달라질 수 있습니다.",
            inputs=[
                IO.Model.Input("model", tooltip="사용할 모델"),
                IO.Boolean.Input("disable_noise", default=False, tooltip="노이즈 적용여부 확인"),
                IO.String.Input("seedset", default=0, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Int.Input("steps", default=20, min=1, max=10000, step=1, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만,"
                             "너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, tooltip="CFG 스케일. 높아질수록 텍스트 처리 강도가 오르지만,\n"
                               "과하면 품질이 떨어지거나 왜곡될 수 있습니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 알고리즘"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러"),
                IO.Conditioning.Input("positive", tooltip="포지티브 컨디셔닝"),
                IO.Conditioning.Input("negative", tooltip="네거티브 컨디셔닝"),
                IO.Latent.Input("latent", tooltip="입력 latent. 빈 라텐트 이미지를 넣거나, 인코딩 또는 불러온 라텐트를 연결할 수 있습니다."),
                IO.Int.Input("height", default= 512, min=64, max=2048, step=1, tooltip="라텐트 에러날 시 대응용 높이조절.\n"
                             " 정상적으로 라텐트가 들어올 땐 작동하지 않습니다."),
                IO.Int.Input("width", default= 512, min=64, max=2048, step=1, tooltip="라텐트 에러날 시 대응용 너비조절.\n"
                             " 정상적으로 라텐트가 들어올 땐 작동하지 않습니다."),
                IO.Combo.Input("mode",
                               options=["cpu","nvidia","amd"], default="cpu", tooltip="실행 장치"),
                IO.Int.Input("start_at_step", default=0, min=0, max=10000, tooltip="스텝 시작점"),
                IO.Int.Input("end_at_step", default=10000, min=0, max=10000, tooltip="스텝 종점"),
                IO.Combo.Input("return_with_leftover_noise", options=["disable", "enable"], default="disable", tooltip="노이즈 반환 처리"),
                IO.Float.Input("denoise", default=1.00, min=0.00, max=1.00, step=0.01, tooltip="디노이즈 강도"),
                IO.Boolean.Input("normalize_noise", tooltip="xl계열에서 노이즈 번 에러 발생시 활용가능한 노이즈 안정화", default=False),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            outputs=[
                IO.Latent.Output("samples", tooltip="디노이즈 처리된 라텐트."),
            ],
        )

    @classmethod
    def execute(cls, model, latent, disable_noise, seedset=0, positive=None, negative=None,
                steps=20, cfg=8.0, sampler_name="euler", scheduler="simple",
                height=512, width=512, mode="cpu", start_at_step=0, end_at_step=10000,
                return_with_leftover_noise="disable", denoise=1.00, normalize_noise=False, clear_cache=False) -> IO.NodeOutput:

        # set device
        if mode == "cpu":
            device = "cpu"
        elif mode == "nvidia":
            device = "cuda"
        elif mode == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "cuda"
            else:
                device = "cpu" 

        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패:", e)
            elif device == "cpu":
                print("CPU 모드: GPU 캐시 초기화 생략")

            gc.collect()
            print("CPU 캐시 초기화 완료")

        cls.show_data_preparation_progress("start")

        positive = get_safe_conditioning(positive, device, model, cfg)

        negative = get_safe_conditioning(negative, device, model, cfg)

        seed_val = parse_seed(seedset)
        if seed_val == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**63 - 1))
        else:
            base_seed = seed_val

        latent_image = None
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print(f"{CYAN}{BOLD}[EasySampler]{RESET} 사용된 시드: {YELLOW}{base_seed}{RESET}")

        if isinstance(latent, dict):
            latent_image = latent.get("samples", None)
        else:
            latent_image = latent

        cls.show_data_preparation_progress("data selecting")
        if latent_image is None:
            latent_image = torch.randn((1,4,height//8,width//8),
                                   generator=generator, device=device)

        latent_image = latent_image.to(device)
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        batch_inds = latent.get("batch_index", None)

        if disable_noise:
            noise = torch.zeros_like(latent_image, device=device)
        else:
            noise = prepare_noise_safe(latent_image, generator, device, batch_inds)
            
        if normalize_noise:
            if latent_image.ndim == 5:
                #soft scale nomalized
                max_std = 1
                noise = safe_rescale_latent(noise, max_std)

            else:
                #"standard normalized"
                noise = normalized_latentnoise(noise)
        else:
            #mode = "none"
            pass

        cls.show_data_preparation_progress("waiting")
        noise_mask = latent.get("noise_mask", None)
        if noise_mask is not None:
            noise_mask = noise_mask.to(device=device, dtype=noise.dtype)
            # if dim not equal
            while noise_mask.ndim < noise.ndim:
                noise_mask = noise_mask.unsqueeze(1)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        force_full_denoise = (return_with_leftover_noise == "disable")

        model_options = {}


        comfy.samplers.cast_to_load_options(model_options, device=device, dtype=latent_image.dtype)


        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise, disable_noise=disable_noise,
            start_step=start_at_step, last_step=end_at_step,
            force_full_denoise=force_full_denoise, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=base_seed
        )


        l_prev = latent.copy()
        l_prev.pop("downscale_ratio_spacial", None)
        out = latent.copy()
        out["samples"] = samples

        del noise
        del latent_image

        gc.collect()
        
        return IO.NodeOutput(out, )

#----------------------------------------
class EasySigmaCalculator(IO.ComfyNode):

    @classmethod
    def  define_schema(cls):
        return IO.Schema(
            node_id="EasySigmaCalculator",
            display_name="시그마 준비기",
            category="커스텀임베딩/샘플링",
            description="(시험중)모델별 시그마를 선택해 준비하고 출력합니다. 이지 샘플러의 경우 full context를 켜면 여기서 입력한걸 이을 수 있습니다.",
            inputs=[
                IO.Model.Input("model", tooltip="사용할 모델", optional=True),
                IO.Combo.Input("model_type", options=["Auto", "ModelSamplingDiscrete", "ModelSamplingDiscreteEDM", "ModelSamplingContinuousEDM", "ModelSamplingContinuousV", 
                               "ModelSamplingDiscreteFlow", "StableCascadeSampling", "ModelSamplingFlux", "ModelSamplingCosmosRFlow"], 
                               default="Auto", tooltip="시그마 준비 모드. auto일 때는 모델연결이 필수이며, 다른 모드일때는모드별 시그마로 준비합니다."),
                IO.Int.Input("steps", default=20, min=1, max=10000, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만, 너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러"),
                IO.Float.Input("denoise", default=1.00, min=0.00, max=1.00, step=0.01, tooltip="디노이즈 강도"),
                IO.Boolean.Input("full_context", default=False, tooltip="스위치를 켤 시 시그마와 스텝, 디노이즈를 다음 노드에 전달합니다. 인풋정보가 같다면 샘플러에도 전달할 수 있습니다.\n"
                                 "False일때는 시그마 외의 출력이 연결될 시 오류가 날 수 있습니다"),
                IO.Combo.Input("mode", options=["cpu","nvidia","amd"], default="cpu", tooltip="실행 장치"),
            ],
            outputs=[
                IO.Sigmas.Output("sigmas", tooltip="간이 시그마 전달"),
                IO.Int.Output("steps", tooltip="전달할 스텝 수"),
                IO.Combo.Output("scheduler", tooltip="전달할 스케줄러 세팅"),
                IO.Float.Output("denoise", tooltip="전달할 디노이즈 강도")
            ],
        )

    @classmethod

    def execute(self, model=None, model_type="Auto", steps=20, scheduler="simple", denoise=1.00, full_context=False, mode="cpu"):

        # set device
        if mode == "cpu":
            device = "cpu"
        elif mode == "nvidia":
            device = "cuda"
        elif mode == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "cuda"
            else:
                device = "cpu" 

        if model is not None:
            base_sigmas = get_sigmas(model, scheduler, steps, denoise, device)
            if model_type == "Auto":
                model_typename = model.model.model_sampling
            else:
                model_typename = sampler_edit.get_sampling_by_type(model_type)
        else:
            base_sigmas = sampler_edit.get_sigma_factory(steps, denoise, scheduler, device)
            model_typename = sampler_edit.get_sampling_by_type(model_type)

        params = sampler_edit.update_params_from_sigmas(base_sigmas, model_type)


        sigmas = sampler_edit.calculate_sigmas_by_sampling(model_type, steps, denoise, scheduler, device, params)
        if full_context:
            return (sigmas, steps, scheduler, denoise) 
        else:
            return (sigmas, None, None, None)

#----------------------------------------

class LatentMaskPrep(IO.ComfyNode):

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "resizing": "resizing...",
            "feathering": "Waiting for response..."
        }
        print(f"{CYAN}{BOLD}[LatentMaskPrep]{RESET} {stages[stage]}")

    @classmethod
    def gaussian_blur(cls, tensor, kernel_size=5, sigma=2):
    
        k = kernel_size // 2
        x = torch.arange(-k, k+1, dtype=torch.float32)
        gauss = torch.exp(-(x**2)/(2*sigma**2))
        gauss = gauss / gauss.sum()
        kernel1d = gauss.unsqueeze(0)
        kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
        kernel2d = kernel2d / kernel2d.sum()
        kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)

        blurred = F.conv2d(tensor, kernel2d, padding=k)
        return blurred

    @classmethod
    def dilate_mtensor(cls, mask_tensor, kernel_size=5, sigma=2, iterations=1, tapered_corners=True):
        orig_h, orig_w = mask_tensor.shape[-2], mask_tensor.shape[-1]
        if tapered_corners:
            for _ in range(iterations):
                blurred = cls.gaussian_blur(mask_tensor, kernel_size=kernel_size, sigma=sigma)
                mask_tensor = (blurred > 0.3).float()
        else:
            stride=1
            for _ in range(iterations):
                mask_tensor = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
        mask_tensor = mask_tensor[:, :, :orig_h, :orig_w]

        return mask_tensor 

    @classmethod
    def _feather_core(cls, mask, feather_size, feather_str):

        if feather_str <= 0:
            return mask
        if feather_size <= 0:
            return mask
        kernel_size = feather_size * 2 + 1
        if kernel_size < 3:
            return mask
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.linspace(-feather_str, feather_str, kernel_size, device=mask.device)
        gauss = torch.exp(-x**2)
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)
        if mask.ndim == 2:        # [H,W]
            mask4d = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:      # [B,H,W]
            mask4d = mask.unsqueeze(1)
        elif mask.ndim == 4:      # [B,1,H,W]
            mask4d = mask
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
            
        blurred = F.conv2d(mask4d, kernel_2d, padding=kernel_size//2)
        #blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-6)
        blurred = torch.clamp(blurred, 0.0, 1.0)

        return blurred

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LatentMaskPrep",
            display_name="라텐트 마스크 준비기",
            category="커스텀임베딩/샘플링",
            description="입력 마스크를 라텐트 크기(1/8)로 리사이즈하고 페더링 처리하여 샘플러에 바로 사용할 수 있는 마스크를 출력합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="일반 크기 마스크"),
                IO.Combo.Input("mask_mode", options=["dilate", "dilate_feather", "default","light_feather","feather","hard_feather"], default="default", tooltip="마스크 처리 모드"),
                IO.Combo.Input("set_mask", options=["default","invert"], default="default", tooltip="마스크 반전 옵션"),
                IO.Combo.Input("strength", options=["0","1","2","3","4","5"], default="0", tooltip="페더링 강도"),
                IO.Int.Input("height", default=512, min=64, max=2048, step=1, tooltip="원본 이미지 높이"),
                IO.Int.Input("width", default=512, min=64, max=2048, step=1, tooltip="원본 이미지 너비"),
                IO.Latent.Input("latent_ref", tooltip="크기 참고용 Latent", optional=True),
                IO.Combo.Input("output_switch", options=["dict_mask","mask_tensor"], default="dict_mask", tooltip="마스크 출력 옵션"),
                IO.Boolean.Input("use_vector", default=False, tooltip="벡터 기반 리사이즈 사용 여부. 리사이즈시 256보다 작아지게 된다면 사용하지 않는게 낫습니다."),
                IO.Combo.Input("resize_mode", options=["nearest","bilinear","bicubic","lanczos","pixelbox"], default="pixelbox", tooltip="리사이즈 보간 모드"),
            ],
            outputs=[
                IO.Mask.Output("latent_mask", tooltip="라텐트 크기(1/8) 마스크"),
            ],
        )

    @classmethod
    def execute(cls, mask, mask_mode="default", set_mask="default", strength="0", height=512, width=512, latent_ref=None, output_switch="dict", use_vector=False, resize_mode="pixelbox") -> IO.NodeOutput:

        cls.show_data_preparation_progress("start")
        total_steps = 4
        pbar = comfy.utils.ProgressBar(int(total_steps))        
        progressbar_update(pbar, 0, total_steps)

        # channels setting      
        mask = ensure_mask_tensor(mask)

        if output_switch == "dict_mask":

            if latent_ref is not None:
                latent_d = 1
                is_video = False
                if "samples" in latent_ref:
                    latent_s = latent_ref.get("samples", None)
                    if latent_s is None:
                        raise ValueError("latent_ref provided but no 'samples' key found")
                    if latent_s.ndim == 4:  # (B,C,H,W)
                        latent_h, latent_w = latent_s.shape[-2:]
                    elif latent_s.ndim == 5:  # (B,C,D,H,W)
                        latent_d, latent_h, latent_w = latent_s.shape[2], latent_s.shape[3], latent_s.shape[4]

                        if isinstance(latent_ref, dict) and ("frame_count" in latent_ref or "fps" in latent_ref):
                            is_video = True
        
                    else:
                        raise ValueError(f"Unsupported latent_ref shape: {latent.shape}")

                progressbar_update(pbar, 1, total_steps)
                # mask resizing to latent size
                cls.show_data_preparation_progress("resizing")
                if use_vector==True:
                    if (latent_w > 0 and latent_w < 256) or (latent_h > 0 and latent_h < 256):
                        print(f"[WARNING] size is too small ({latent_w}x{latent_h}). Falling back to standard interpolation to prevent vector resizing artifacts.")
                        interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                        mask_resized = F.interpolate(mask, size=(latent_h, latent_w), mode=interp_map[resize_mode])
                        
                    else:
                        if latent_ref is not None and "samples" in latent_ref:
                            latent_s = latent_ref["samples"]
                            if latent_s.ndim == 4:  # sd type latent
                                mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
                                mask_resized_np = vector_resize_mask(mask_np, latent_w, latent_h, resize_mode)
                                if mask_resized_np.ndim == 3 and mask_resized_np.shape[2] == 3:
                                    mask_resized_np = cv2.cvtColor(mask_resized_np, cv2.COLOR_BGR2GRAY)
                                elif mask_resized_np.ndim == 2:
                                    pass
                                else:
                                    raise ValueError(f"Unexpected mask shape: {mask_resized_np.shape}")

                                mask_resized = torch.from_numpy(mask_resized_np).unsqueeze(0).unsqueeze(0).float() / 255.0
                                mask_resized = mask_resized.to(mask.device)
                            elif latent_s.ndim == 5:  # 3D/flow latent
                                mask_resized = vector_resize_mask_3d_final(mask, latent_d, latent_h, latent_w, resize_mode)
                        else:
                            # fallback: B,C,H,W
                            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
                            mask_resized = vector_resize_mask(mask_np, latent_w, latent_h, resize_mode)
                            if mask_resized_np.ndim == 3 and mask_resized_np.shape[2] == 3:
                                mask_resized_np = cv2.cvtColor(mask_resized_np, cv2.COLOR_BGR2GRAY)
                            elif mask_resized_np.ndim == 2:
                                pass
                            else:
                                raise ValueError(f"Unexpected mask shape: {mask_resized_np.shape}")
                            
                            mask_resized = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
                            mask_resized = mask_resized.to(mask.device)

                else:
                    interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                    mask_resized = F.interpolate(mask, size=(latent_h, latent_w), mode=interp_map[resize_mode])
            else:
                latent_h, latent_w = height // 8, width // 8

                progressbar_update(pbar, 1, total_steps)
                # mask resizing to latent size
                cls.show_data_preparation_progress("resizing")
                if use_vector==True:
                    if (latent_w > 0 and latent_w < 256) or (latent_h > 0 and latent_h < 256):
                        print(f"[WARNING] size is too small ({latent_w}x{latent_h}). Falling back to standard interpolation to prevent vector resizing artifacts.")
                        interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                        mask_resized = F.interpolate(mask, size=(latent_h, latent_w), mode=interp_map[resize_mode])
                        
                    else:
                        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
                        mask_resized_np = vector_resize_mask(mask_np, latent_w, latent_h, resize_mode)
                        if mask_resized_np.ndim == 3 and mask_resized_np.shape[2] == 3:
                            mask_resized_np = cv2.cvtColor(mask_resized_np, cv2.COLOR_BGR2GRAY)
                        elif mask_resized_np.ndim == 2:
                            pass
                        else:
                            raise ValueError(f"Unexpected mask shape: {mask_resized_np.shape}")

                        mask_resized = torch.from_numpy(mask_resized_np).unsqueeze(0).unsqueeze(0).float() / 255.0
                        mask_resized = mask_resized.to(mask.device)

                else:
                    interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                    mask_resized = F.interpolate(mask, size=(latent_h, latent_w), mode=interp_map[resize_mode])


            progressbar_update(pbar, 2, total_steps)
            # invert option
            if set_mask == "invert":
                mask = 1.0 - mask_resized
            else:
                mask = mask_resized

            progressbar_update(pbar, 3, total_steps)
            cls.show_data_preparation_progress("feathering")
            if mask.ndim == 5:
                print(f"{CYAN}{BOLD}[LatentMaskPrep]{RESET} mask shape 5D : skip feather/dilate.")
                mask_final = mask  # pass
            else:
                min_size = min(latent_h, latent_w)
                strength_level=int(strength)
                strength_map = {0:0, 1: 0.3, 2: 0.75, 3: 1.25, 4: 1.75, 5: 2.5}
                feather_str = strength_map.get(strength_level, 0)
                if mask_mode == "dilate":
                    mask_final = cls.dilate_mtensor(mask, kernel_size=5, sigma=2, iterations=1)
                elif mask_mode == "dilate_feather":
                    dilated = cls.dilate_mtensor(mask, kernel_size=5, sigma=2, iterations=1)
                    mask_final = cls._feather_core(dilated, int(min_size*0.06), feather_str)
                elif mask_mode == "default":
                    mask_final = mask
                elif mask_mode == "light_feather":
                    mask_final = cls._feather_core(mask, int(min_size*0.03), feather_str*0.5)
                elif mask_mode == "feather":
                    mask_final = cls._feather_core(mask, int(min_size*0.06), feather_str)
                elif mask_mode == "hard_feather":
                    mask_final = cls._feather_core(mask, int(min_size*0.12), feather_str*1.5)
                else:
                    mask_final = mask

            progressbar_update(pbar, 4, total_steps)
            mask = mask_final.float()

            return IO.NodeOutput({"noise_mask": mask_final})
        else:
            #base mask

            if latent_ref is not None:
                if "samples" in latent_ref:
                    latent_s = latent_ref.get("samples", None)
                    if latent_s is None:
                        raise ValueError("latent_ref provided but no 'samples' key found")
                    if latent_s.ndim == 4:  # (B,C,H,W)
                        latent_h, latent_w = latent_s.shape[-2:]
                    elif latent_s.ndim == 5:  # (B,C,D,H,W)
                        latent_h, latent_w = latent_s.shape[3], latent_s.shape[4]
                    else:
                        raise ValueError(f"Unsupported latent_ref shape: {latent.shape}")
            else:
                latent_h, latent_w = height // 8, width // 8

            progressbar_update(pbar, 1, total_steps)
            # mask resizing to latent size
            cls.show_data_preparation_progress("resizing")
            if use_vector==True:
                if (latent_w > 0 and latent_w < 256) or (latent_h > 0 and latent_h < 256):
                    print(f"[WARNING] size is too small ({latent_w}x{latent_h}). Falling back to standard interpolation to prevent vector resizing artifacts.")
                    interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                    mask_resized = F.interpolate(mask, size=(latent_h, latent_w), mode=interp_map[resize_mode])

                else:
                    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
                    mask_resized_np = vector_resize_norm_mask(mask_np, latent_w, latent_h, resize_mode)
                    if mask_resized_np.ndim == 3 and mask_resized_np.shape[2] == 3:
                        mask_resized_np = cv2.cvtColor(mask_resized_np, cv2.COLOR_BGR2GRAY)
                    elif mask_resized_np.ndim == 2:
                        pass
                    else:
                        raise ValueError(f"Unexpected mask shape: {mask_resized_np.shape}")

                    mask_resized = torch.from_numpy(mask_resized_np).unsqueeze(0).unsqueeze(0).float() / 255.0
                    mask_resized = mask_resized.to(mask.device)
            
            else:
                interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                mask_resized = F.interpolate(mask, size=(latent_h, latent_w), mode=interp_map[resize_mode])

            progressbar_update(pbar, 2, total_steps)
            # invert option
            if set_mask == "invert":
                mask = 1.0 - mask_resized
            else:
                mask = mask_resized

            progressbar_update(pbar, 3, total_steps)
            cls.show_data_preparation_progress("feathering")
            min_size = min(latent_h, latent_w)
            strength_level=int(strength)
            strength_map = {0:0, 1: 0.3, 2: 0.75, 3: 1.25, 4: 1.75, 5: 2.5}
            feather_str = strength_map.get(strength_level, 0)
            if mask_mode == "dilate":
                mask_final = cls.dilate_mtensor(mask, kernel_size=5, sigma=2, iterations=1)
            elif mask_mode == "dilate_feather":
                dilated = cls.dilate_mtensor(mask, kernel_size=5, sigma=2, iterations=1)
                mask_final = cls._feather_core(dilated, int(min_size*0.06), feather_str)
            elif mask_mode == "default":
                mask_final = mask
            elif mask_mode == "light_feather":
                mask_final = cls._feather_core(mask, int(min_size*0.03), feather_str*0.5)
            elif mask_mode == "feather":
                mask_final = cls._feather_core(mask, int(min_size*0.06), feather_str)
            elif mask_mode == "hard_feather":
                mask_final = cls._feather_core(mask, int(min_size*0.12), feather_str*1.5)
            else:
                mask_final = mask

            progressbar_update(pbar, 4, total_steps)
            mask = mask_final.float()

            return IO.NodeOutput(mask_final)

#----------------------------------------
class MaskPrepResizer(IO.ComfyNode):

    @classmethod
    def gaussian_blur(cls, tensor, kernel_size=5, sigma=2):
    
        k = kernel_size // 2
        x = torch.arange(-k, k+1, dtype=torch.float32)
        gauss = torch.exp(-(x**2)/(2*sigma**2))
        gauss = gauss / gauss.sum()
        kernel1d = gauss.unsqueeze(0)
        kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
        kernel2d = kernel2d / kernel2d.sum()
        kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)

        blurred = F.conv2d(tensor, kernel2d, padding=k)
        return blurred

    @classmethod
    def dilate_mtensor(cls, mask_tensor, kernel_size=5, sigma=2, iterations=1, tapered_corners=True):
        orig_h, orig_w = mask_tensor.shape[-2], mask_tensor.shape[-1]
        if tapered_corners:
            for _ in range(iterations):
                blurred = cls.gaussian_blur(mask_tensor, kernel_size=kernel_size, sigma=sigma)
                mask_tensor = (blurred > 0.3).float()
        else:
            stride=1
            for _ in range(iterations):
                mask_tensor = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
        mask_tensor = mask_tensor[:, :, :orig_h, :orig_w]

        return mask_tensor 

    @classmethod
    def _feather_core(cls, mask, feather_size, feather_str):

        if feather_str <= 0:
            return mask
        if feather_size <= 0:
            return mask
        kernel_size = feather_size * 2 + 1
        if kernel_size < 3:
            return mask
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.linspace(-feather_str, feather_str, kernel_size, device=mask.device)
        gauss = torch.exp(-x**2)
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)
        if mask.ndim == 2:        # [H,W]
            mask4d = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:      # [B,H,W]
            mask4d = mask.unsqueeze(1)
        elif mask.ndim == 4:      # [B,1,H,W]
            mask4d = mask
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
            
        blurred = F.conv2d(mask4d, kernel_2d, padding=kernel_size//2)
        #blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-6)
        blurred = torch.clamp(blurred, 0.0, 1.0)

        return blurred

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskPrepResizer",
            display_name="마스크 벡터 리사이징 준비기",
            category="커스텀임베딩/샘플링",
            description="벡터 리사이징과 페더링을 활용하여 일반 이미지 해상도 기준으로 마스크를 정밀 가공합니다.",
            inputs=[
                IO.Mask.Input("mask", tooltip="입력 마스크"),
                IO.Combo.Input("mask_mode", options=["dilate", "dilate_feather", "default", "light_feather", "feather", "hard_feather"], default="default", tooltip="마스크 처리 모드"),
                IO.Combo.Input("set_mask", options=["default", "invert"], default="default", tooltip="마스크 반전 옵션"),
                IO.Combo.Input("strength", options=["0", "1", "2", "3", "4", "5"], default="0", tooltip="페더링 강도"),
                IO.Int.Input("custom_width", default=512, min=64, max=4096, step=1, tooltip="가공할 마스크의 최종 픽셀 너비"),
                IO.Int.Input("custom_height", default=512, min=64, max=4096, step=1, tooltip="가공할 마스크의 최종 픽셀 높이"),
                IO.Boolean.Input("use_vector", default=True, tooltip="벡터 기반 리사이즈 사용 여부. 리사이즈시 256보다 작아지게 된다면 사용하지 않는게 낫습니다."),
                IO.Combo.Input("resize_mode", options=["nearest", "bilinear", "bicubic", "lanczos", "pixelbox"], default="pixelbox", tooltip="리사이즈 보간 모드"),
            ],
            outputs=[
                IO.Mask.Output("mask", tooltip="가공된 마스크"),
            ]
        )

    @classmethod
    def execute(cls, mask, mask_mode="default", set_mask="default", strength="0", custom_width=512, custom_height=512, use_vector=True, resize_mode="pixelbox") -> IO.NodeOutput:
        
        mask = ensure_mask_tensor(mask)
        orig_h, orig_w = mask.shape[-2], mask.shape[-1]

        target_h, target_w = custom_height, custom_width

        if use_vector==True:
            if (target_w > 0 and target_w < 256) or (target_h > 0 and target_h < 256):
                print(f"[WARNING] size is too small ({target_w}x{target_h}). Falling back to standard interpolation to prevent vector resizing artifacts.")
                interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
                mask_resized = F.interpolate(mask, size=(target_h, target_w), mode=interp_map[resize_mode])
            else:
                mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
                mask_resized_np = vector_resize_mask(mask_np, target_w, target_h, resize_mode)
                if mask_resized_np.ndim == 3 and mask_resized_np.shape[2] == 3:
                    mask_resized_np = cv2.cvtColor(mask_resized_np, cv2.COLOR_BGR2GRAY)
                elif mask_resized_np.ndim == 2:
                    pass
                else:
                    raise ValueError(f"Unexpected mask shape: {mask_resized_np.shape}")

                mask_resized = torch.from_numpy(mask_resized_np).unsqueeze(0).unsqueeze(0).float() / 255.0
                mask_resized = mask_resized.to(mask.device)

        else:
            interp_map = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "bicubic", "pixelbox": "area"}
            mask_resized = F.interpolate(mask, size=(target_h, target_w), mode=interp_map[resize_mode])

        if set_mask == "invert":
            mask_processed = 1.0 - mask_resized
        else:
            mask_processed = mask_resized

        min_size = min(target_h, target_w)
        strength_level = int(strength)
        strength_map = {0: 0, 1: 0.3, 2: 0.75, 3: 1.25, 4: 1.75, 5: 2.5}
        feather_str = strength_map.get(strength_level, 0)

        if mask_mode == "dilate":
            mask_final = cls.dilate_mtensor(mask_processed, kernel_size=5, sigma=2, iterations=1)
        elif mask_mode == "dilate_feather":
            dilated = cls.dilate_mtensor(mask_processed, kernel_size=5, sigma=2, iterations=1)
            mask_final = cls._feather_core(dilated, int(min_size * 0.06), feather_str)
        elif mask_mode == "default":
            mask_final = mask_processed
        elif mask_mode == "light_feather":
            mask_final = cls._feather_core(mask_processed, int(min_size * 0.03), feather_str * 0.5)
        elif mask_mode == "feather":
            mask_final = cls._feather_core(mask_processed, int(min_size * 0.06), feather_str)
        elif mask_mode == "hard_feather":
            mask_final = cls._feather_core(mask_processed, int(min_size * 0.12), feather_str * 1.5)
        else:
            mask_final = mask_processed

        return IO.NodeOutput(mask_final.float())

#----------------------------------------

class EasyMultyVaeEncode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyMultyVaeEncode",
            display_name="간단 Vae 멀티인코드",
            category="커스텀임베딩/샘플링",
            description="이미지를 받아 라텐트로 가공합니다.\n"
                        "추가 이미지를 받을 경우 샘플링용 라텐트로도 출력됩니다.",
            inputs=[
                IO.Image.Input("pixel", tooltip="샘플링할 베이스 이미지"),
                IO.Image.Input("sample_palette", tooltip="노이즈샘플링 추가용 샘플링 팔레트 이미지", optional=True),
                IO.Vae.Input("vae", tooltip="참고할 Vae객체"),
                IO.Combo.Input("output_set", options=["default", "Multy_latent"], default="default", tooltip="동시 출력기능을 켜고 끄는 설정입니다."),
            ],
            outputs=[
                IO.Latent.Output("samples", tooltip="베이스 라텐트"),
                IO.Latent.Output("re_sample_latent", tooltip="샘플링 팔레트용 라텐트"),
            ],
        )

    @classmethod
    def execute(cls, pixel, vae, sample_palette=None, output_set="default") -> IO.NodeOutput:
        # base image encoding
        pixel_tensor = safe_to_torch(pixel)
        base_latent_tensor = vae.encode(pixel_tensor)
        base_latent = {"samples": base_latent_tensor}

        # sample image encoding (optional)
        if sample_palette is not None:
            palette_tensor = safe_to_torch(sample_palette)
            palette_latent_tensor = vae.encode(palette_tensor)
            palette_latent = {"samples": palette_latent_tensor}
        else:
            palette_latent = None

        if output_set == "default": # base_latent
            return IO.NodeOutput(base_latent, None)
        else:  # multi_latent
            return IO.NodeOutput(base_latent, palette_latent)

#----------------------------------------
class EasyInpaintAndConditioning(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            # (B, C, H, W) -> (B, H, W, C)
            arr = arr.permute(0, 2, 3, 1)
            
        return arr.float()

    @classmethod
    def sanitize_input_mask(cls, mask):
        if mask is None:
            return None

        if isinstance(mask, dict):
            if "latent_mask" in mask:
                mask = mask["latent_mask"]
            elif "noise_mask" in mask:
                mask = mask["noise_mask"]
            elif "mask" in mask:
                mask = mask["mask"]
            elif "samples" in mask:
                mask = mask["samples"]

        if hasattr(mask, "dtype") and mask.dtype == torch.bool:
            mask = mask.float()
        if mask.ndim == 5:
            return mask
        return ensure_mask_tensor(mask)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyInpaintAndConditioning",
            display_name="간단 인페인트 및 컨디셔닝",
            category="커스텀임베딩/샘플링",
            description=(
                "이미지와 마스크를 기반으로 인페인팅용 컨디셔닝(concat)과 노이즈 마스크를 생성합니다.\n"
                "5D 타입 라텐트이미지의 충돌을 줄였습니다.\n"
                "작동시 에러가 발생하는 스케줄러들이 존재합니다. 모델에 적힌 추천 스케줄러를 사용하시기 바랍니다.\n"
                "시작점과 끝점을 조정한 경우, 조건결합 노드(combine)를 사용할 필요가 있습니다."
            ),
            inputs=[
                IO.Conditioning.Input("positive", tooltip="긍정 프롬프트 컨디셔닝"),
                IO.Conditioning.Input("negative", tooltip="부정 프롬프트 컨디셔닝"),
                IO.Image.Input("pixels", tooltip="인페인팅 대상 원본 이미지 텐서"),
                IO.Mask.Input("mask", tooltip="인페인팅 영역 마스크"),
                IO.Vae.Input("vae", tooltip="이미지 인코딩에 사용할 VAE 객체"),
                IO.Combo.Input("mask_resize_mode", options=["nearest", "bilinear", "bicubic", "lanczos", "pixelbox"], default="pixelbox", tooltip="리사이즈 보간 모드"),
                IO.Int.Input("grow_mask_by", default=6, min=0, max=64, step=1, tooltip="마스크 경계를 부드럽게 확장하여 이음새를 자연스럽게 만듭니다."),
                IO.Boolean.Input("noise_mask", default=True, tooltip="마스크 영역 내부로만 노이즈와 샘플링 범위를 제한합니다."),
                IO.Image.Input("sample_palette", tooltip="멀티 샘플링용 추가 팔레트 이미지 (선택)", optional=True),
                IO.Float.Input("pixel_concat", default=0.0, min=0.0, max=1.0, step=0.1, tooltip="마스크 영역 내의 이미지 우선치를 조정합니다. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Combo.Input("pixel_blend", options=["default", "blend", "overwrite"], default="default", tooltip="마스크 영역 내의 합성 형태를 지정합니다. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="팔레트 블렌딩시 예상이미지. 팔레트 미연결시 원본이미지가 나옵니다."),
                IO.Boolean.Input("use_inpaint_model", default=False, tooltip="샘플링 데이터 전달시 참고합니다. 스위치를 켤 경우 마스크영역이 인페인트모델 재처리영역인 회색조가 됩니다."),
                IO.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="인페인팅 컨디셔닝이 작동하기 시작하는 타임스텝 비율"),
                IO.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.05, tooltip="인페인팅 컨디셔닝이 종료되는 타임스텝 비율"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Conditioning.Output("positive", tooltip="인페인팅 속성이 바인딩된 긍정 컨디셔닝"),
                IO.Conditioning.Output("negative", tooltip="인페인팅 속성이 바인딩된 부정 컨디셔닝"),
                IO.Latent.Output("latent", tooltip="노이즈 마스크가 포함된 인페인팅 전용 베이스 라텐트"),
                IO.Latent.Output("concat_latent", tooltip="인페인트 작업이 처리된 컨디셔닝 전달 라텐트. 작업 처리가 되긴 하는지 확인하는 용도입니다."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, pixels, mask, vae, mask_resize_mode="pixelbox", grow_mask_by=6, noise_mask=True, sample_palette=None, pixel_concat=0.0, 
                pixel_blend="default", clear_cache=False, preview_mode=False, use_inpaint_model=False, start_percent=0.0, end_percent=1.0) -> IO.NodeOutput:

        if clear_cache:
            current_device = model_management.get_torch_device()
            if current_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패:", e)
            else:
                print("CPU 모드: GPU 캐시 초기화 생략")

            gc.collect()
            print("CPU 캐시 초기화 완료")

        # (B, H, W, C)
        mask_tensor = cls.sanitize_input_mask(mask)          # (B, 1, H, W)

        downscale_ratio = 8

        if pixels.dim() == 4:
            if pixels.shape[-1] == 3:  # [B, H, W, C]
                h, w = pixels.shape[-2], pixels.shape[-3]
            else:                      # [B, C, H, W]
                h, w = pixels.shape[-2], pixels.shape[-1]
        else:
            h, w = pixels.shape[-2], pixels.shape[-1]

 
        x = (w // downscale_ratio) * downscale_ratio
        y = (h // downscale_ratio) * downscale_ratio

        masknp_2d = mask_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255
        resized_mask_np = vector_resize_mask(masknp_2d, w, h, resize_mode = mask_resize_mode)
        mask_tensor = torch.from_numpy(resized_mask_np).to(pixels.device)

        orig_pixels = pixels.clone()
        if w != x or h != y:
            x_offset = (w % downscale_ratio) // 2
            y_offset = (h % downscale_ratio) // 2
            pixels = pixels[:, :, y_offset:y + y_offset, x_offset:x + x_offset]
            mask_tensor = mask_tensor[y_offset:y + y_offset, x_offset:x + x_offset]

        if grow_mask_by == 0:
            mask_erosion = ensure_mask_tensor(mask_tensor)
        else:
            mask_tensor = ensure_mask_tensor(mask_tensor) 
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by), device=mask_tensor.device)
            padding = math.ceil((grow_mask_by - 1) / 2)
            mask_erosion = torch.clamp(F.conv2d(mask_tensor.round(), kernel_tensor, padding=padding), 0, 1)
        p_h, p_w = (pixels.shape[1], pixels.shape[2]) if pixels.shape[-1] == 3 else (pixels.shape[2], pixels.shape[3])

        if mask_erosion.shape[-2] > p_h or mask_erosion.shape[-1] > p_w:
            mask_erosion = mask_erosion[:, :, :p_h, :p_w]

        m4d = mask_erosion.permute(0, 2, 3, 1)
        concat_pixels = pixels.clone()  # [B, H, W, C]
        if sample_palette is not None:
            palette_pixels = cls.ensure_image_tensor(sample_palette)  # [B, H, W, C]

            if palette_pixels.shape[-1] == 3 or palette_pixels.shape[-1] == 1:
                palette_pixels = F.interpolate(
                    palette_pixels.permute(0, 3, 1, 2), 
                    size=(p_h, p_w), 
                    mode="bilinear", 
                    align_corners=False
                ).permute(0, 2, 3, 1)  # [B, H, W, C]

            expanded_mask = m4d
            if expanded_mask.shape[-1] == 1 and concat_pixels.shape[-1] == 3:
                expanded_mask = expanded_mask.repeat(1, 1, 1, 3)

            alpha = max(0.0, min(1.0, pixel_concat))
            if alpha > 0.0:
                if pixel_blend == "overwrite":
                    raw_blended = pixels * (1.0 - expanded_mask) + (pixels * (1.0 - alpha) + palette_pixels * alpha) * expanded_mask
                    orig_pixels = raw_blended.clone()
                elif pixel_blend == "blend":
                
                    raw_blended = pixels * (1.0 - expanded_mask) + (pixels * (1.0 - alpha) + palette_pixels * alpha) * expanded_mask
            
                else:  # default
                    raw_blended = pixels + (palette_pixels - pixels) * expanded_mask * alpha

                # Ratio Clamping
                max_vals = raw_blended.max(dim=-1, keepdim=True)[0]

                scale_factor = torch.clamp(1.0 / torch.max(max_vals, torch.ones_like(max_vals)), max=1.0)
                
                # Apply Ratio Clamping only within the mask area
                blend_alpha = expanded_mask * alpha
                concat_pixels = raw_blended * scale_factor * blend_alpha + pixels * (1.0 - blend_alpha)
            else:  # default
                print(f"skip the pixel blend logics")
                pass
        else:
            pass

        if use_inpaint_model:
            m3d = m4d.squeeze(-1)
            m = (1.0 - m3d.round()) # (B, H, W)
            for i in range(min(3, concat_pixels.shape[-1])):
                concat_pixels[:,:,:,i] -= 0.5
                concat_pixels[:,:,:,i] *= m
                concat_pixels[:,:,:,i] += 0.5

        else:
            m = (1.0 - m4d.round()) # (B, H, W, 1)
            muted_pixels = concat_pixels * 0.5 + 0.5 * 0.5
            concat_pixels = concat_pixels * m + muted_pixels * (1.0 - m)

        concat_latent = vae.encode(concat_pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {
            "samples": orig_latent
        }
        blend_latent = {
            "samples": concat_latent
        }
        if noise_mask:
            lat_h, lat_w = orig_latent.shape[-2], orig_latent.shape[-1]
            target_mask = F.interpolate(mask_erosion.float(), size=(lat_h, lat_w), mode="nearest")
            out_latent["noise_mask"] = target_mask.round().float()

        conditioning_data = {"reference_latents": concat_latent, "concat_mask": mask_erosion}

        out_positive = sampler_edit.editcond_set_values_with_timestep_range(positive, conditioning_data, start_percent=start_percent, end_percent=end_percent)
        out_negative = sampler_edit.editcond_set_values_with_timestep_range(negative, conditioning_data, start_percent=start_percent, end_percent=end_percent)
        if preview_mode:
            return IO.NodeOutput(out_positive, out_negative, out_latent, blend_latent, ui=UI.PreviewImage(concat_pixels))
        return IO.NodeOutput(out_positive, out_negative, out_latent, blend_latent)

#----------------------------------------


class EasyInpaintAndConditioning_modify(IO.ComfyNode):

    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            # (B, C, H, W) -> (B, H, W, C)
            arr = arr.permute(0, 2, 3, 1)
            
        return arr.float()

    @classmethod
    def sanitize_input_mask(cls, mask):
        if mask is None:
            return None

        if isinstance(mask, dict):
            if "latent_mask" in mask:
                mask = mask["latent_mask"]
            elif "noise_mask" in mask:
                mask = mask["noise_mask"]
            elif "mask" in mask:
                mask = mask["mask"]
            elif "samples" in mask:
                mask = mask["samples"]

        if hasattr(mask, "dtype") and mask.dtype == torch.bool:
            mask = mask.float()
        if mask.ndim == 5:
            return mask
        return ensure_mask_tensor(mask)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyInpaintAndConditioning_modify",
            display_name="간단 인페인트 및 컨디셔닝(모드)",
            category="커스텀임베딩/샘플링",
            description=(
                "이미지와 마스크를 기반으로 인페인팅용 컨디셔닝(concat)과 노이즈 마스크를 생성합니다.\n"
                "5D 타입 라텐트이미지의 충돌을 줄였습니다.\n"
                "AdaIN/Spatial의 경우는 구도와 화풍이 비슷해야 이미지 붕괴를 줄일 수 있습니다.\n"
                "작동시 에러가 발생하는 스케줄러들이 존재합니다. 모델에 적힌 추천 스케줄러를 사용하시기 바랍니다.\n"
                "시작점과 끝점을 조정한 경우, 조건결합 노드(combine)를 사용할 필요가 있습니다."
            ),
            inputs=[
                IO.Conditioning.Input("positive", tooltip="긍정 프롬프트 컨디셔닝"),
                IO.Conditioning.Input("negative", tooltip="부정 프롬프트 컨디셔닝"),
                IO.Image.Input("pixels", tooltip="인페인팅 대상 원본 이미지 텐서"),
                IO.Vae.Input("vae", tooltip="이미지 인코딩에 사용할 VAE 객체"),
                IO.Mask.Input("mask", optional=True, tooltip="인페인팅 영역 마스크. 연결 안할 시 전체 기준이 됩니다."),
                IO.Combo.Input("mask_resize_mode", options=["nearest", "bilinear", "bicubic", "lanczos", "pixelbox"], default="pixelbox", tooltip="리사이즈 보간 모드. 마스크를 연결할 시에만 작동합니다."),
                IO.Int.Input("grow_mask_by", default=2, min=0, max=64, step=1, tooltip="마스크 경계를 부드럽게 확장하여 이음새를 자연스럽게 만듭니다. 마스크를 연결할 시에만 작동합니다."),
                IO.Boolean.Input("noise_mask", default=True, tooltip="마스크 영역 내부로만 노이즈와 샘플링 범위를 제한합니다. 마스크를 연결할 시에만 작동합니다."),
                IO.Image.Input("sample_palette", tooltip="멀티 샘플링용 추가 팔레트 이미지 (선택)", optional=True),
                IO.Combo.Input("transfer_str", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"], default="0", tooltip="샘플 라텐트의 가이딩 강도. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Combo.Input("style_type", options=["Latent_AdaIN", "Spatial_Attention", "Residual_Blending", "Channel_Selective"], default="Latent_AdaIN", tooltip="라텐트 가이딩 정보 유형. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Combo.Input("latent_mode", options=["linear", "ease_in", "EASE_OUT", "EASE_IN_OUT"], default="linear", tooltip="라텐트 가이딩 정보 유형. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Float.Input("Latent_activation_value", default=1.0, min=0.0, max=1.0, step=0.1, tooltip="라텐트 가이드 강도 (훅 영향력). 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="팔레트 블렌딩시 예상이미지. 팔레트 미연결시 원본이미지가 나옵니다."),
                IO.Combo.Input("inpaint_blend_mode", options=["Inpaint_Model", "Static_Edge",  "Faint_Edge", "Blank_Solid", "Sample_Guided"], default="Static_Edge", tooltip="인페인팅 전처리 블렌딩 모드 선택:\n"
                "- Inpaint_Model: 전용 인페인팅 모델용 (완전 회색조)\n"
                "- Static_Edge: 원본 엣지와 명암을 옅게 남김 (디테일 수정/유지용)\n"
                "- Blank_Solid: 마스크 안쪽을 완전한 중간 회색으로 비움 (구도/자세 변경용 추천)\n"
                "- Sample_Guided: 샘플 팔레트/레퍼런스 흐름과 결합"
                ),
                IO.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="인페인팅 컨디셔닝이 작동하기 시작하는 타임스텝 비율"),
                IO.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.05, tooltip="인페인팅 컨디셔닝이 종료되는 타임스텝 비율"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Conditioning.Output("positive", tooltip="인페인팅 속성이 바인딩된 긍정 컨디셔닝"),
                IO.Conditioning.Output("negative", tooltip="인페인팅 속성이 바인딩된 부정 컨디셔닝"),
                IO.Latent.Output("latent", tooltip="노이즈 마스크가 포함된 인페인팅 전용 베이스 라텐트"),
                IO.Latent.Output("concat_latent", tooltip="인페인트 작업이 처리된 컨디셔닝 전달 라텐트. 작업 처리가 되긴 하는지 확인하는 용도입니다."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, pixels, vae, mask=None, mask_resize_mode="pixelbox", grow_mask_by=2, noise_mask=True, sample_palette=None, 
                transfer_str="0", style_type="Latent_AdaIN", latent_mode="linear", Latent_activation_value=1.0, 
                clear_cache=False, preview_mode=False, inpaint_blend_mode="Static_Edge", start_percent=0.0, end_percent=1.0) -> IO.NodeOutput:

        if clear_cache:
            current_device = model_management.get_torch_device()
            if current_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패:", e)
            else:
                print("CPU 모드: GPU 캐시 초기화 생략")

            gc.collect()
            print("CPU 캐시 초기화 완료")

        start_percent = max(0.0, min(0.9, start_percent))
        end_percent = max(start_percent, min(1.0, end_percent))

        if pixels.dim() == 4:
            if pixels.shape[-1] == 3:  # [B, H, W, C]
                b, h, w, c = pixels.shape
            else:  # [B, C, H, W]
                b, c, h, w = pixels.shape
                pixels = pixels.permute(0, 2, 3, 1) # [B, H, W, C]
        else:
            b, h, w, c = pixels.shape

        downscale_ratio = 8
        target_w = (w // downscale_ratio) * downscale_ratio
        target_h = (h // downscale_ratio) * downscale_ratio

        mask_tensor = cls.sanitize_input_mask(mask)
        if mask_tensor is None:
            mask_tensor = torch.ones((b, 1, h, w), device=pixels.device, dtype=torch.float32)
        else:
            masknp_2d = mask_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255
            resized_mask_np = vector_resize_mask(masknp_2d, w, h, resize_mode=mask_resize_mode)
            mask_tensor = torch.from_numpy(resized_mask_np).to(pixels.device)
            mask_tensor = ensure_mask_tensor(mask_tensor)

        if w != target_w or h != target_h:
            x_offset = (w % downscale_ratio) // 2
            y_offset = (h % downscale_ratio) // 2
            pixels = pixels[:, y_offset:target_h + y_offset, x_offset:target_w + x_offset, :]
            mask_tensor = mask_tensor[:, :, y_offset:target_h + y_offset, x_offset:target_w + x_offset]

        orig_pixels = pixels.clone()

        p_h, p_w = pixels.shape[1], pixels.shape[2]

        if grow_mask_by == 0:
            mask_erosion = torch.clamp(ensure_mask_tensor(mask_tensor), 0.0, 1.0)
        else:
            mask_tensor = ensure_mask_tensor(mask_tensor) 
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by), device=mask_tensor.device)
            padding = math.ceil((grow_mask_by - 1) / 2)
            mask_erosion = torch.clamp(F.conv2d(mask_tensor.round(), kernel_tensor, padding=padding), 0, 1)

        if mask_erosion.shape[-2] > p_h or mask_erosion.shape[-1] > p_w:
            mask_erosion = mask_erosion[:, :, :p_h, :p_w]

        mask_erosion = torch.clamp(mask_erosion, 0.0, 1.0) # 
        m4d = mask_erosion.permute(0, 2, 3, 1)

        orig_latent = vae.encode(orig_pixels)

        concat_pixels = pixels.clone()
        concat_latent = None
        ref_latent_features = {}
        t_str_val = 0
        
        t_str_val = int(transfer_str)/20

        conditioning_data = {}
        if sample_palette is not None and t_str_val != 0:
            palette_pixels = cls.ensure_image_tensor(sample_palette)
            if palette_pixels.shape[-1] == 3 or palette_pixels.shape[-1] == 1:
                palette_pixels = F.interpolate(
                    palette_pixels.permute(0, 3, 1, 2), 
                    size=(p_h, p_w), 
                    mode="bilinear", 
                    align_corners=False
                ).permute(0, 2, 3, 1)

            concat_latent = vae.encode(palette_pixels)
            lat_samples = concat_latent.get("samples", concat_latent) if isinstance(concat_latent, dict) else concat_latent
            
            if lat_samples is not None:
                lat_samples = lat_samples.to(pixels.device)

                if concat_latent.dim() == 5:
                    mu = concat_latent.mean(dim=(3, 4), keepdim=True)
                    sigma = concat_latent.std(dim=(3, 4), keepdim=True) + 1e-6
                else:
                    mu = concat_latent.mean(dim=(2, 3), keepdim=True)
                    sigma = concat_latent.std(dim=(2, 3), keepdim=True) + 1e-6

                ref_latent_features = {
                    "mean": mu,
                    "std": sigma,
                    "v_feat": concat_latent,
                    "base": concat_latent
                }

                conditioning_data.update({"ref_latent_features": ref_latent_features, "ref_lat_transfer_str": t_str_val, "ref_lat_style_type": style_type, "reference_weight": Latent_activation_value, "palette_mode": latent_mode,})
        else:
            # default
            print(f"skip the pixel blend logics")
            pass

        if "Inpaint_Model" in inpaint_blend_mode:
            m3d = m4d.squeeze(-1)
            m = (1.0 - m3d.round())
            for i in range(min(3, concat_pixels.shape[-1])):
                concat_pixels[:,:,:,i] -= 0.5
                concat_pixels[:,:,:,i] *= m
                concat_pixels[:,:,:,i] += 0.5

        elif "Static_Edge" in inpaint_blend_mode:
            m = (1.0 - m4d.round())
            muted_pixels = concat_pixels * 0.5 + 0.5 * 0.5
            concat_pixels = concat_pixels * m + muted_pixels * (1.0 - m)

        elif "Faint_Edge" in inpaint_blend_mode:
            m = (1.0 - m4d.round())
            # Increase the 0.5(gray) ratio to almost eliminate the original color, leaving only a very faint outline.
            faint_pixels = concat_pixels * 0.2 + 0.5 * 0.8
            concat_pixels = concat_pixels * m + faint_pixels * (1.0 - m)

        elif "Blank_Solid" in inpaint_blend_mode:
            m = (1.0 - m4d.round())
            solid_gray = torch.full_like(concat_pixels, 0.5)
            concat_pixels = concat_pixels * m + solid_gray * (1.0 - m)

        elif "Sample_Guided" in inpaint_blend_mode:
            m = (1.0 - m4d.round())
            
            # step 1 : Blank_Solid
            solid_gray = torch.full_like(concat_pixels, 0.5)

            # step 2 : Sobel Edge Guided
            if sample_palette is not None and 'palette_pixels' in locals():
                # 1) RGB to neutral gray and fused pixel
                muted_palette = palette_pixels * 0.3 + 0.5 * 0.7

                # 2) composite edgeguide (blank solid 0.7 / edge guide 0.3)
                concat_pixels = muted_palette * (1.0 - m) + concat_pixels * m

            else:
                print("cannot find sample_palette, prepare it as a blank solid.")
                base_cleared = concat_pixels * m + solid_gray * (1.0 - m)
                concat_pixels = base_cleared

        concat_latent = vae.encode(concat_pixels)

        out_latent = {
            "samples": orig_latent
        }
        blend_latent = {
            "samples": concat_latent
        }
        if noise_mask:
            lat_h, lat_w = orig_latent.shape[-2], orig_latent.shape[-1]
            target_mask = F.interpolate(mask_erosion.float(), size=(lat_h, lat_w), mode="nearest")
            out_latent["noise_mask"] = target_mask.round().float()

        if mask is not None:
            conditioning_data.update({"reference_latents": concat_latent, "concat_mask": mask_erosion})
        else:
            conditioning_data.update({"reference_latents": concat_latent})

        out_positive = sampler_edit.editcond_set_values_with_timestep_range(positive, conditioning_data, start_percent=start_percent, end_percent=end_percent)
        out_negative = sampler_edit.editcond_set_values_with_timestep_range(negative, conditioning_data, start_percent=start_percent, end_percent=end_percent)

        if preview_mode:
            return IO.NodeOutput(out_positive, out_negative, out_latent, blend_latent, ui=UI.PreviewImage(concat_pixels))
        return IO.NodeOutput(out_positive, out_negative, out_latent, blend_latent)

#----------------------------------------


class EasySamplerHook(IO.ComfyNode):

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "data selecting": "input data checking...",
            "noise set": "create noise...",
            "waiting": "load model. Waiting for response..."
        }
        print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} {stages[stage]}")

    @classmethod
    def apply_denoise_mask(cls, latent, mask):
        # mask size match
        if mask.shape[-1] != latent.shape[-1] or mask.shape[-2] != latent.shape[-2]:
            mask = F.interpolate(mask.float(), size=latent.shape[-2:], mode="nearest")

        return mask

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasySamplerHook",
            display_name="간단 훅 샘플러",
            category="커스텀임베딩/샘플링",
            description="Noise 추가 여부, step 범위, leftover noise 처리, 간단한 간이 인페인팅 등을 제공하는 샘플러.\n"
                        "훅 로직이 있는 마스크나 로라등을 적용하기 위한 노드입니다.\n"
                        "CPU 모드에서도 충돌을 최소화하며 latent를 초기화하고 샘플링을 수행합니다.\n"
                        "라데온 계열도 돌아가긴 하지만, ROCm 처리 조건이 부족하면 CPU모드로 기동될 수 있습니다.\n"
                        "사용된 시드 값은 CMD에 프린트 로그로 남아있습니다. 시드를 고정하려면 복사해서 붙여넣으면 됩니다.\n"
                        "Quen 계열에도 사용은 가능합니다.\n"
                        "절대 여러장을 동시처리하지 마세요. 샘플러가 터집니다.\n"
                        "[참고] 베이스 온리(Base Only) 모드에서는 시드가 고정되어 있어도 연산 특성상 결과물이 미세하게 달라질 수 있습니다.",
            inputs=[
                IO.Model.Input("model", tooltip="사용할 모델"),
                IO.Boolean.Input("disable_noise", default=False, tooltip="노이즈 적용여부 확인"),
                IO.Combo.Input("noise_mode", options=["default", "base_double"], default="default",
                                tooltip="노이즈 적용 방식. 노이즈 스타일을 선택지에 따라 다르게 처리합니다. 베이스 더블 사용 시 normalize_noise를 none으로 두시면 안 됩니다."),
                IO.String.Input("seedset", default=0, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Sigmas.Input("sigmas", tooltip="사용할 시그마 정보. 입력 안할 시 디폴트로 돌아갑니다.", optional=True),
                IO.Int.Input("steps", default=20, min=1, max=10000, step=1, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만,"
                                "너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, tooltip="CFG 스케일. 높아질수록 텍스트 처리 강도가 오르지만,\n"
                                   "과하면 품질이 떨어지거나 왜곡될 수 있습니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 알고리즘"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러"),
                IO.Conditioning.Input("positive", tooltip="포지티브 컨디셔닝"),
                IO.Conditioning.Input("negative", tooltip="네거티브 컨디셔닝"),
                IO.Latent.Input("latent", tooltip="입력 latent. 빈 라텐트 이미지를 넣거나, 인코딩 또는 불러온 라텐트를 연결할 수 있습니다."),
                IO.Int.Input("height", default= 512, min=64, max=2048, step=1, tooltip="라텐트 에러날 시 대응용 높이조절.\n"
                                 " 정상적으로 라텐트가 들어올 땐 작동하지 않습니다."),
                IO.Int.Input("width", default= 512, min=64, max=2048, step=1, tooltip="라텐트 에러날 시 대응용 너비조절.\n"
                                 " 정상적으로 라텐트가 들어올 땐 작동하지 않습니다."),
                IO.Combo.Input("mode", options=["cpu","nvidia","amd"], default="cpu", tooltip="실행 장치"),
                IO.Float.Input("denoise", default=1.000, min=0.000, max=1.000, step=0.001, tooltip="디노이즈 강도"),
                IO.Combo.Input("normalize_noise", options=["none", "soft", "standard"], tooltip="xl계열에서 노이즈 번 에러 발생시 활용가능한 노이즈 안정화", default="none"),
                IO.Boolean.Input("get_shrinkmask", default=True, tooltip="훅 마스크 확인시 마스크영역을 조금 좁힙니다. 마스크가 없을 시엔 적용되지 않습니다."),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            outputs=[
                IO.Latent.Output("samples", tooltip="디노이즈 처리된 라텐트."),
            ],
        )

    @classmethod
    def execute(cls, model, disable_noise, latent, noise_mode="default", seedset=0, positive=None, negative=None, sigmas=None, steps=20, cfg=8.0, sampler_name="euler",
                scheduler="simple", height=512, width=512, mode="cpu", denoise=1.000, normalize_noise="none", get_shrinkmask=True, clear_cache=False) -> IO.NodeOutput:

        cls.show_data_preparation_progress("start")

        # set device
        if mode == "cpu":
            device = "cpu"
        elif mode == "nvidia":
            device = "cuda"
        elif mode == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "cuda"
            else:
                device = "cpu" 

        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패:", e)
            elif device == "cpu":
                print("CPU 모드: GPU 캐시 초기화 생략")

            gc.collect()
            print("CPU 캐시 초기화 완료")

        sampler = comfy.samplers.sampler_object(sampler_name)

        positive, pos_hook, pos_masks = get_safe_conditioning_hooked(positive, device, model, cfg)

        negative, neg_hook, neg_masks = get_safe_conditioning_hooked(negative, device, model, cfg)

        findkey = False
        for cond_list in (positive, negative):
            if cond_list:
                for c in cond_list:
                    if isinstance(c, tuple) and len(c) > 1 and isinstance(c[1], dict):
                        if c[1].get("masksetkey", False):
                            findkey = True
                            break
                    elif isinstance(c, dict):
                        if c.get("masksetkey", False):
                            findkey = True
                            break
                if findkey:
                    break

        seed_val = parse_seed(seedset)
        if seed_val == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**63 - 1))
        else:
            base_seed = seed_val
                
        latent_image = None
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} 사용된 시드: {YELLOW}{base_seed}{RESET}")

        if isinstance(latent, dict):
            latent_image = latent.get("samples", None)
        else:
            latent_image = latent

        cls.show_data_preparation_progress("data selecting")
        if latent_image is None:
            latent_image = torch.randn((1,4,height//8,width//8),
                                       generator=generator, device=device)

        else:
            latent_image = latent_image.to(device)
            latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        if latent_image.ndim == 5:  # (B,C,D,H,W)
            h, w = latent_image.shape[3], latent_image.shape[4]
        elif latent_image.ndim == 4:  # (B,C,H,W)
            h, w = latent_image.shape[2], latent_image.shape[3]
        elif latent_image.ndim == 3:  # (C,H,W)
            h, w = latent_image.shape[1], latent_image.shape[2]
        else:
            raise ValueError(f"Unexpected latent shape: {latent_image.shape}")

        target_size = (w, h)

        noise_mask = None
        
        valid_masks = []
        if pos_masks:
            valid_masks.extend([m for m in pos_masks if m is not None])
        if neg_masks:
            valid_masks.extend([m for m in neg_masks if m is not None])

        if len(valid_masks) > 0:
            threshold = 0.6
            noise_mask = prepare_noise_mask(valid_masks, threshold, target_size, get_shrinkmask, findkey)
            if noise_mask is not None:
                noise_mask = noise_mask.to(device)
                print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} A mask was found during conditioning. Switching to masked region-limited sampling.")
            else:
                if findkey:
                    print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} MaskBreaker is activated, lifting the mask area restriction. (Global sampling)")
                else:
                    print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} No mask was found in the conditioning. It is being processed using standard sampling.")
        else:
            if isinstance(latent, dict) and "noise_mask" in latent:
                noise_mask = latent.get("noise_mask")
                print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} latent mask was found, Switching to masked region-limited sampling.")
            else:
                print(f"{CYAN}{BOLD}[EasySamplerHook]{RESET} No connected mask. Processing with standard sampling.")

        cls.show_data_preparation_progress("noise set")

        batch_inds = latent.get("batch_index", None)

        if disable_noise:
            noise = torch.zeros_like(latent_image, device=device)
        else:
            noise_base = prepare_noise_safe(latent_image, generator, device, batch_inds)
            if noise_mode == "base_double":
                pure_noise = torch.randn_like(latent_image, generator=generator, device=device)

                pure_ratio = denoise  
                latent_ratio = 1.0 - pure_ratio

                base_noise = noise_base * latent_ratio + pure_noise * pure_ratio

                noise = get_processed_noise(base_noise, normalize_noise)

            else:
                # "default"
                noise = get_processed_noise(noise_base, normalize_noise)

        # 2. Multy batch remove
        if noise is not None and noise.shape[0] > 1:
            noise = noise[:1]
        if latent_image is not None and latent_image.shape[0] > 1:
            latent_image = latent_image[:1]
        cls.show_data_preparation_progress("waiting")

        if sigmas is not None:
            use_sigmas = sigmas.to(device)
        else:
            use_sigmas = get_sigmas(model, scheduler, steps, denoise, device)

        extra = None
        if latent_image.ndim == 5:
            extra = {"is_flow": (latent_image.ndim == 5)}
            if noise_mask is not None:
                noise_mask = cls.apply_denoise_mask(latent_image, noise_mask)
            

        dim = latent_image.ndim
        shape = latent_image.shape[2:]
        model_options = build_model_options(use_sigmas, pos_hook, neg_hook, noise_mask, extra_options=extra, latent_shape=shape, latent_dim=dim)
        
        comfy.samplers.cast_to_load_options(model_options, device=device, dtype=latent_image.dtype)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = sampler_edit.sample_custom_edit(model, noise, cfg, sampler, use_sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=base_seed)


        l_prev = latent.copy()
        l_prev.pop("downscale_ratio_spacial", None)
        out = latent.copy()
        out["samples"] = samples

        del noise
        del latent_image
        gc.collect()
            
        return IO.NodeOutput(out, )


#----------------------------------------
class EasySamplerADV(IO.ComfyNode):

    @classmethod
    def apply_denoise_mask(cls, latent_image, mask):
        # mask size match
        if mask.shape[-1] != latent_image.shape[-1] or mask.shape[-2] != latent_image.shape[-2]:
            mask = F.interpolate(mask.float(), size=latent_image.shape[-2:], mode="nearest")

        return mask

    @classmethod
    def show_data_preparation_progress(cls, stage):
        stages = {
            "start": "get data...",
            "data selecting": "input data checking...",
            "noise set": "create noise...",
            "waiting": "load model. Waiting for response..."
        }
        print(f"{CYAN}{BOLD}[EasySamplerADV]{RESET} {stages[stage]}")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasySamplerADV",
            display_name="간단 샘플러 어드밴스",
            category="커스텀임베딩/샘플링",
            description="Noise 추가 여부, step 범위, leftover noise 처리, 간단한 간이 인페인팅 등을 제공하는 샘플러.\n"
                        "마스크를 사용할 시 마스크는 라텐트 크기에 맞게 리사이즈를 해야 합니다.\n"
                        "CPU 모드에서도 충돌을 최소화하며 latent를 초기화하고 샘플링을 수행합니다.\n"
                        "라데온 계열도 돌아가긴 하지만, ROCm 처리 조건이 부족하면 CPU모드로 기동될 수 있습니다.\n"
                        "사용된 시드 값은 CMD에 프린트 로그로 남아있습니다. 시드를 고정하려면 복사해서 붙여넣으면 됩니다.\n"
                        "Quen 계열에도 사용은 가능합니다.\n"
                        "절대 여러장을 동시처리하지 마세요. 샘플러가 터집니다.\n"
                        "훅 사용 로직을 추가했지만, 시험기능이라 마스크와 같이 쓰면 오버플로우 위험이 있습니다.\n"
                        "[참고] 베이스 온리(Base Only) 모드에서는 시드가 고정되어 있어도 연산 특성상 결과물이 미세하게 달라질 수 있습니다.",
            inputs=[
                IO.Model.Input("model", tooltip="사용할 모델"),
                IO.Boolean.Input("disable_noise", default=False, tooltip="노이즈 적용여부 확인"),
                IO.Mask.Input("mask", tooltip="영역 지정 마스크. 샘플링 참고 자료가 됩니다.", optional=True),
                IO.Combo.Input("noise_mode", options=["default", "base_double"], default="default",
                                tooltip="노이즈 적용방식. 노이즈 스타일을 선택지에 따라 다르게 처리합니다. 베이스 더블 사용 시 normalize_noise를 none으로 두시면 안 됩니다."),
                IO.String.Input("seedset", default=0, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Sigmas.Input("sigmas", tooltip="사용할 시그마 정보. 입력 안할 시 디폴트로 돌아갑니다.", optional=True),
                IO.Int.Input("steps", default=20, min=1, max=10000, step=1, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만,"
                                "너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, tooltip="CFG 스케일. 높아질수록 텍스트 처리 강도가 오르지만,\n"
                                   "과하면 품질이 떨어지거나 왜곡될 수 있습니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 알고리즘"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러"),
                IO.Conditioning.Input("positive", tooltip="포지티브 컨디셔닝"),
                IO.Conditioning.Input("negative", tooltip="네거티브 컨디셔닝"),
                IO.Latent.Input("latent", tooltip="입력 latent. 빈 라텐트 이미지를 넣거나, 인코딩 또는 불러온 라텐트를 연결할 수 있습니다."),
                IO.Int.Input("height", default= 512, min=64, max=2048, step=1, tooltip="라텐트 에러날 시 대응용 높이조절.\n"
                                 " 정상적으로 라텐트가 들어올 땐 작동하지 않습니다."),
                IO.Int.Input("width", default= 512, min=64, max=2048, step=1, tooltip="라텐트 에러날 시 대응용 너비조절.\n"
                                 " 정상적으로 라텐트가 들어올 땐 작동하지 않습니다."),
                IO.Combo.Input("mode", options=["cpu","nvidia","amd"], default="cpu", tooltip="실행 장치"),
                IO.Float.Input("denoise", default=1.000, min=0.000, max=1.000, step=0.001, tooltip="디노이즈 강도"),
                IO.Combo.Input("normalize_noise", options=["none", "soft", "standard"], tooltip="xl계열에서 노이즈 번 에러 발생시 활용가능한 노이즈 안정화", default="none"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            outputs=[
                IO.Latent.Output("samples", tooltip="디노이즈 처리된 라텐트."),
            ],
        )

    @classmethod
    def execute(cls, model, disable_noise, latent, mask=None, noise_mode="default", palette_inject="2",
                seedset=0, positive=None, negative=None, sigmas=None, steps=20, cfg=8.0, sampler_name="euler",
                scheduler="simple", height=512, width=512, mode="cpu", denoise=1.000, normalize_noise="none", clear_cache=False) -> IO.NodeOutput:

        cls.show_data_preparation_progress("start")

        # set device
        if mode == "cpu":
            device = "cpu"
        elif mode == "nvidia":
            device = "cuda"
        elif mode == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "cuda"
            else:
                device = "cpu" 

        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패:", e)
            elif device == "cpu":
                print("CPU 모드: GPU 캐시 초기화 생략")
            gc.collect()
            print("CPU 캐시 초기화 완료")

        sampler = comfy.samplers.sampler_object(sampler_name)

        positive = get_safe_conditioning(positive, device, model, cfg)

        negative = get_safe_conditioning(negative, device, model, cfg)

        seed_val = parse_seed(seedset)
        if seed_val == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**63 - 1))
        else:
            base_seed = seed_val
                
        latent_image = None
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print(f"{CYAN}{BOLD}[EasySamplerADV]{RESET} 사용된 시드: {YELLOW}{base_seed}{RESET}")

        if isinstance(latent, dict):
            latent_image = latent.get("samples", None)
        else:
            latent_image = latent

        cls.show_data_preparation_progress("data selecting")
        if latent_image is None:
            latent_image = torch.randn((1,4,height//8,width//8),
                                       generator=generator, device=device)

        else:
            latent_image = latent_image.to(device)
            latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        if latent_image.ndim == 5:  # (B,C,D,H,W)
            h, w = latent_image.shape[3], latent_image.shape[4]
        elif latent_image.ndim == 4:  # (B,C,H,W)
            h, w = latent_image.shape[2], latent_image.shape[3]
        elif latent_image.ndim == 3:  # (C,H,W)
            h, w = latent_image.shape[1], latent_image.shape[2]
        else:
            raise ValueError(f"Unexpected latent shape: {latent_image.shape}")

        target_size = (w, h)

        noise_mask = None
        if mask is not None:
            if isinstance(mask, dict):
                if "latent_mask" in mask:
                    mask = mask["latent_mask"]
                elif "noise_mask" in mask:
                    mask = mask["noise_mask"]
                elif "mask" in mask:
                    mask = mask["mask"]
            if mask.dtype == torch.bool:
                mask = mask.float()
            denoise_mask = mask
            noise_mask = cls.apply_denoise_mask(latent_image, denoise_mask)
            noise_mask = noise_mask.to(device)

        elif isinstance(latent, dict) and "noise_mask" in latent:
            denoise_mask = latent.get("noise_mask")
            noise_mask = cls.apply_denoise_mask(latent_image, denoise_mask)
            noise_mask = noise_mask.to(device)
            print(f"{CYAN}{BOLD}[EasySamplerADV]{RESET} Latent mask find")

        cls.show_data_preparation_progress("noise set")

        batch_inds = latent.get("batch_index", None)

        inject_level = int(palette_inject) * 0.1
        inject_level = max(0.0, min(inject_level, 0.5))

        if disable_noise:
            noise = torch.zeros_like(latent_image, device=device)
        else:
            noise_base = prepare_noise_safe(latent_image, generator, device, batch_inds)

            if noise_mode == "base_double":
                pure_noise = torch.randn_like(latent_image, generator=generator, device=device)

                pure_ratio = denoise  
                latent_ratio = 1.0 - pure_ratio

                base_noise = noise_base * latent_ratio + pure_noise * pure_ratio

                noise = get_processed_noise(base_noise, normalize_noise)

            else:
                # "default"
                noise = get_processed_noise(noise_base, normalize_noise)


        # 2. Multy batch remove
        if noise is not None and noise.shape[0] > 1:
            noise = noise[:1]
        if latent_image is not None and latent_image.shape[0] > 1:
            latent_image = latent_image[:1]
        cls.show_data_preparation_progress("waiting")

        if sigmas is not None:
            use_sigmas = sigmas.to(device)
        else:
            use_sigmas = get_sigmas(model, scheduler, steps, denoise, device)

        extra = None
        if latent_image.ndim == 5:
            extra = {"is_flow": (latent_image.ndim == 5)}

        model_options = {"transformer_options": extra or {}}
        comfy.samplers.cast_to_load_options(model_options, device=device, dtype=latent_image.dtype)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, use_sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=base_seed)


        l_prev = latent.copy()
        l_prev.pop("downscale_ratio_spacial", None)
        out = latent.copy()
        out["samples"] = samples

        del noise
        del latent_image
        gc.collect()
            
        return IO.NodeOutput(out, )

#----------------------------------------

class EasyLoraVersionChecker(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        files = []
        base_dirs = folder_paths.get_folder_paths("loras")
        for d in base_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(rel_path)

        return IO.Schema(
            node_id="EasyLoraVersionChecker",
            display_name="LoRA 버전 체커",
            category="커스텀임베딩/특수",
            description="LoRA/LyCORIS safetensors 파일의 타입, 차원, 추천 clip skip, 해시, 호환 모델을 확인합니다.\n"
                        "탐색 기준은 kohya 방식을 기준으로 돌립니다. \n"
                        "일부 학습에서는 LoRA를 LyCORIS처럼 병렬 세팅하거나 메타데이터를 증발시키는 경우가 있어 메타데이터만으로는 감지가 힘들 수 있습니다.",
            inputs=[
                IO.Combo.Input("lora_file", options=files if files else ["no lora"], default=files[0] if files else "no lora", tooltip="확인할 LoRA 파일"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="폰트 크기"),
                IO.Combo.Input("find_keys", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
                               default="20", tooltip="확인할 키워드 수"),
                IO.Combo.Input("save_text", options=["off", "on"],
                               default="off", tooltip="저장 여부"),
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True,
            outputs=[],
        )


    @classmethod
    def execute(cls, lora_file, font_size=14, find_keys="20", save_text="off") -> IO.NodeOutput:
        path = None
        for d in folder_paths.get_folder_paths("loras"):
            candidate = os.path.join(d, lora_file)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f"{lora_file} not found")

        # safetensors data loading
        data = load_file(path)
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}

        # network type
        network_module = metadata.get("ss_network_module", "lora")
        network_args = metadata.get("ss_network_args", "")
        if "lycoris" in str(network_module).lower():
            network_type = "LyCORIS"
        elif "conv_dim" in str(network_args).lower() or "conv_alpha" in str(network_args).lower():
            network_type = "LoRA (LyCORIS-style args detected)"
        else:
            network_type = "LoRA"

        # model name check
        base_model = metadata.get("ss_base_model_version") or metadata.get("modelspec.architecture")
        sd_model_name = metadata.get("ss_sd_model_name")
        if base_model:
            model_group = base_model
        elif sd_model_name:
            if sd_model_name.lower() in ["model.safetensors", "model.ckpt", "sd_model.safetensors"]:
                model_group = f"{sd_model_name} (metadata incomplete)"
            else:
                model_group = sd_model_name
        else:
            model_group = "Unknown"

        # Dim check
        shape_counts = {}
        if "ss_network_dim" in metadata:
            recommended_dim = f"dim[{metadata['ss_network_dim']}] (metadata confirmed)"
        else:
            for k, v in data.items():
                shape = list(v.shape)
                shape_str = str(shape)
                shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
            if shape_counts:
                top_shape = max(shape_counts, key=shape_counts.get)
                recommended_dim = f"dim[{top_shape}] (train dim)"
            else:
                recommended_dim = "dim[unknown]"

        # Shape Info
        shape_info = "unknown"
        if shape_counts:
            top_shape = max(shape_counts, key=shape_counts.get)
            shape_info = f"{top_shape} (x{shape_counts[top_shape]})"

        # Recommended Shape
        if "ss_bucket_info" in metadata or "ss_resolution" in metadata:
            if "ss_resolution" in metadata:
                recommended_shape = f"{metadata['ss_resolution']} (kohya bucket)"
            else:
                bucket_info = metadata.get("ss_bucket_info", {}).get("buckets", {})
                if bucket_info:
                    max_bucket = max(bucket_info.items(), key=lambda x: x[1].get("count", 0))
                    recommended_shape = f"{max_bucket[1]['resolution']} (kohya bucket)"
                else:
                    recommended_shape = "unknown"
        else:
            if shape_counts:
                top_shape = max(shape_counts, key=shape_counts.get)
                recommended_shape = f"{top_shape} (train shape)"
            else:
                recommended_shape = "unknown"

        # Prompt Keywords (kohya + NAI)

        tag_freq = metadata.get("ss_tag_frequency", {})
        prompt_keywords = "keyword find failed, no metadata recording information"
        if isinstance(tag_freq, str):
            try:
                tag_freq = json.loads(tag_freq)
            except Exception:
                tag_freq = {}
        prompt_keywords = "keyword find failed, no metadata recording information"

        if tag_freq and isinstance(tag_freq, dict):
            if "dataset" in tag_freq:
                # kohya style

                all_tags = tag_freq.get("dataset", {})
                    
            else:
                all_tags = {}
                for ver, vdict in tag_freq.items():
                    if isinstance(vdict, dict):
                        for tag, freq in vdict.items():
                            if isinstance(freq, (int, float)):
                                all_tags[tag] = freq

            if all_tags:
                sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
                N = int(find_keys)
                prompt_keywords = ", ".join([tag for tag, _ in sorted_tags[:N]])

        # extra metadata fields
        clip_skip = metadata.get("ss_clip_skip", "not recorded")
        epoch = metadata.get("ss_epoch", "unknown")
        num_epochs = metadata.get("ss_num_epochs", "unknown")
        learning_rate = metadata.get("ss_learning_rate", "unknown")
        optimizer = metadata.get("ss_optimizer", "not recorded")

        # training_style check
        if "lion" in optimizer.lower():
            training_style = "Diffusers-style training detected"
        elif "adamw8bit" in optimizer.lower():
            training_style = "kohya-ss style"
        elif "dadapt" in optimizer.lower():
            training_style = "metadata incomplete (DAdapt style)"
        elif optimizer == "not recorded":
            training_style = "metadata incomplete"
        else:
            training_style = "none"

        # file_size
        file_size = os.path.getsize(path)
        file_size_mb = round(file_size / (1024*1024), 2)
        file_size_gb = round(file_size / (1024*1024*1024), 2)

        info_lines = [
            f"Network Type: {network_type}",
            f"Model Group: {model_group}",
            f"File Size: {file_size_mb} MB ({file_size_gb} GB)",
            f"Recommended Dim: {recommended_dim}",
            f"Recommended Shape: {recommended_shape}",
            f"Shape Info: {shape_info}",
            f"Clip Skip: {clip_skip}",
            f"Epoch: {epoch}/{num_epochs}",
            f"Learning Rate: {learning_rate}",
            f"Training Style: {training_style}",
            f"Optimizer: {optimizer}",
            "Prompt Keywords:",
            prompt_keywords
        ]

        if save_text == "on":
            txt_path = path.replace(".safetensors", "_metadata.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(info_lines))
                print(f"Metadata saved to {txt_path}")
            except Exception as e:
                print(f"Failed to save metadata: {e}")

        final_text = "\n".join(info_lines)
        return IO.NodeOutput(final_text, ui={"stats":[final_text], "font_size":[font_size]})


#----------------------------------------
#Extra Node Settings
#----------------------------------------

class EasyTextslot_Loader(IO.ComfyNode):

    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(b_layout_dir):
        for f in os.listdir(b_layout_dir):
            full_path = os.path.join(b_layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyTextslot_Loader",
            display_name="텍스트 슬롯 로더",
            category="커스텀임베딩/특수",
            description="이미지를 선택한 컷 레이아웃에 맞게 배치합니다.",
            inputs=[
                IO.Combo.Input("b_layout", options=cls.files, default=cls.files[0] if cls.files else "no ref_layout", tooltip="텍스트박스 레퍼런스 이미지"),
                IO.Combo.Input("textbox_Lines", options=["thin", "bold", "outline"], default="thin", tooltip="텍스트박스 형태 지정"),
                IO.String.Input("fill_color", default="#ffffff", tooltip="텍스트박스 색상 (HEX 코드)"),
                IO.String.Input("outline_color", default="#000000", tooltip="텍스트박스 외곽 색상 (HEX 코드)"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True,
            outputs=[bubblelayout.Output("b_layout", tooltip="로드한 텍스트 슬롯 이미지")],
        )

    @classmethod
    def execute(cls, b_layout, textbox_Lines, fill_color, outline_color, show_preview=False) -> IO.NodeOutput:

        if b_layout == "no ref_layout":
            raise ValueError("Layout cannot found.")

        ref_path = os.path.join(b_layout_dir, b_layout)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Layout not found: {ref_path}")
        
        img = Image.open(ref_path).convert("RGB")
        arr = np.array(img)
        tolerance = 30

        arr_i32 = arr.astype(np.int32)

        # B (Green)
        target_b = np.array(COLOR_MAP["B"], dtype=np.int32)
        lower_b = np.clip(target_b - tolerance, 0, 255).astype(np.uint8)
        upper_b = np.clip(target_b + tolerance, 0, 255).astype(np.uint8)
        mask_b = cv2.inRange(arr, lower_b, upper_b)

        # C (Blue)
        target_c = np.array(COLOR_MAP["C"], dtype=np.int32)
        lower_c = np.clip(target_c - tolerance, 0, 255).astype(np.uint8)
        upper_c = np.clip(target_c + tolerance, 0, 255).astype(np.uint8)
        mask_c = cv2.inRange(arr, lower_c, upper_c)

        # 3. textbox_Lines
        if textbox_Lines == "bold":
            kernel = np.ones((3, 3), np.uint8)
            mask_b = cv2.dilate(mask_b, kernel, iterations=1)
        elif textbox_Lines == "outline":
            kernel = np.ones((5, 5), np.uint8)
            mask_b = cv2.dilate(mask_b, kernel, iterations=1)
        else:
            pass

        # (HEX to RGB)
        fill_rgb = hex_to_rgb(fill_color)
        outline_rgb = hex_to_rgb(outline_color)

        result_arr = arr.copy()
        result_arr[mask_c > 0] = fill_rgb
        result_arr[mask_b > 0] = outline_rgb

        result_tensor = torch.from_numpy(result_arr.astype(np.float32) / 255.0).unsqueeze(0)

        if show_preview:

            result_rgb = to_tensor_output(result_tensor)
            return IO.NodeOutput(result_tensor, ui=UI.PreviewImage(result_rgb))

        return IO.NodeOutput(result_tensor)

#----------------------------------------

class EasyTextslot(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = []
        for root, dirs, filenames in os.walk(font_dir):
            for f in filenames:
                name, ext = os.path.splitext(f)
                if ext.lower() in [".ttf", ".ttc"]:   
                    rel_path = os.path.relpath(os.path.join(root, f), font_dir)
                    files.append(name)


        return IO.Schema(
            node_id="EasyTextslot",
            display_name="텍스트 이미지 기입",
            category="커스텀임베딩/특수",
            description="이미지에 텍스트 키를 넣습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                bubblelayout.Input("b_layout", tooltip="텍스트박스 참고 이미지", optional=True),
                IO.Combo.Input("font", options=files, default=files[0] if files else "", tooltip="적용할 폰트 파일"),
                IO.Int.Input("font_size", default=32, min=6, max=48, tooltip="폰트 크기"),
                IO.Combo.Input("bubble_shape", options=["ellipse", "rounded_rect", "shout", "long_shout", "thought"], default="ellipse", tooltip="말풍선 형태 선택"),
                IO.String.Input("fill_color", default="#ffffff", tooltip="텍스트박스 색상 (HEX 코드)"),
                IO.String.Input("outline_color", default="#000000", tooltip="텍스트박스 외곽 색상 (HEX 코드)"),
                IO.String.Input("text_color", default="#000000", tooltip="글자 색상 (HEX 코드)"),
                IO.String.Input("text", multiline=True, default="empty text", tooltip="텍스트 기입란"),
                IO.Int.Input("slot_size_x", default=240, min=8, max=2048, tooltip="텍스트박스 크기(X축)"),
                IO.Int.Input("slot_size_y", default=100, min=8, max=2048, tooltip="텍스트박스 크기(Y축)"),
                IO.Int.Input("slot_x", default=8, min=8, max=2048, tooltip="텍스트박스 위치(X축)"),
                IO.Int.Input("slot_y", default=10, min=8, max=2048, tooltip="텍스트박스 위치(Y축)"),
                IO.Float.Input("lotate_slot", default=0.0, min=-180.0, max=180.0, tooltip="텍스트박스 회전"),
                IO.Combo.Input("textbox_Lines", options=["thin", "bold", "outline"], default="thin", tooltip="텍스트박스 형태 지정"),
                IO.Combo.Input("textbox_blur", options=["basic", "spread"], default="basic", tooltip="텍스트박스 블러처리 여부"),
                IO.Combo.Input("text_anchor", options=["basic", "left-top", "left-center", "left-bottom", "center", "right-top", "right-center", "right-bottom"], default="basic", tooltip="텍스트 배치"),
                IO.Combo.Input("text_style", options=["thin", "bold", "outline"], default="thin", tooltip="텍스트 형태 지정"),
                IO.Combo.Input("text_shadow", options=["basic", "shadow"], default="basic", tooltip="텍스트 그림자 추가 여부"),
                IO.Int.Input("text_box_opa", default=128, min=0, max=255, tooltip="텍스트박스 투명도 설정. 0(투명)>255(불투명)"),
                IO.Int.Input("text_Line_opa", default=128, min=0, max=255, tooltip="텍스트박스 라인 투명도 설정. 0(투명)>255(불투명)"),
                IO.Int.Input("text_opa", default=255, min=0, max=255, tooltip="텍스트 투명도 설정. 0(투명)>255(불투명)"),
            ],
            outputs=[
                IO.Image.Output("output_image", tooltip="텍스트가 삽입된 이미지")
            ]
        )

    @classmethod
    def execute(self, image, font, font_size, bubble_shape, fill_color, outline_color, text_color, text, slot_size_x, slot_size_y,
                slot_x, slot_y, lotate_slot=0.0, textbox_Lines="thin", textbox_blur="basic", text_anchor="basic", text_style="thin", 
                text_shadow="basic", text_box_opa=128, text_Line_opa=128, text_opa=128, b_layout=None) -> IO.NodeOutput:



        torch_img = to_torch_image(image)

        arr = (torch_img.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # OpenCV → PIL
        pil_img = Image.fromarray(arr).convert("RGBA")
        
        # Max data
        img_w, img_h = pil_img.size
        
        slot_size_x = max(8, min(slot_size_x, img_w))
        slot_size_y = max(8, min(slot_size_y, img_h))
        
        slot_x = max(0, min(slot_x, img_w - slot_size_x))
        slot_y = max(0, min(slot_y, img_h - slot_size_y))
        
        font_size = max(6, min(font_size, 48))
        text_box_opa = max(0, min(text_box_opa, 255))
        text_Line_opa = max(0, min(text_Line_opa, 255))
        text_opa = max(0, min(text_opa, 255))

        # text slot layer
        bubble_layer = Image.new("RGBA", pil_img.size, (0,0,0,0))
        draw = ImageDraw.Draw(bubble_layer)

        # load fonts
        font_path = None
        for root, dirs, filenames in os.walk(font_dir):
            for f in filenames:
                name, ext = os.path.splitext(f)
                if ext.lower() in [".ttf", ".ttc"] and name.lower() == font.lower():
                    font_path = os.path.join(root, f)
                    break

        if font_path is None:
            raise FileNotFoundError(f"Font '{font}' not found in models/fonts")

        pil_font = ImageFont.truetype(font_path, font_size)

        # text slot coordinates
        coords = (slot_x, slot_y, slot_x + slot_size_x, slot_y + slot_size_y)

        # text slot
        fill_rgba = hex_to_rgb(fill_color) + (text_box_opa,)
        outline_rgba = hex_to_rgb(outline_color) + (text_Line_opa,)
        text_rgba = hex_to_rgb(text_color) + (text_opa,)

        if b_layout is not None:
            # b_layout → PIL
            torch_bubble = to_torch_image(b_layout)
            arr_bubble = (torch_bubble.squeeze().cpu().numpy() * 255).astype(np.uint8)
            bubble_img = Image.fromarray(arr_bubble).convert("RGBA")

            bubble_np = np.array(bubble_img)
            
            tolerance = 30
            bubble_i32 = bubble_np[..., :3].astype(np.int32)
            target_a = np.array(COLOR_MAP["A"], dtype=np.int32)
            
            lower_a = np.clip(target_a - tolerance, 0, 255).astype(np.uint8)
            upper_a = np.clip(target_a + tolerance, 0, 255).astype(np.uint8)
            
            mask_a = cv2.inRange(bubble_np[..., :3], lower_a, upper_a)

            bubble_np[mask_a > 0, 3] = 0
            
            processed_bubble = Image.fromarray(bubble_np, mode="RGBA")

            # slot size setting
            processed_bubble = processed_bubble.resize((slot_size_x, slot_size_y), Image.Resampling.LANCZOS)
            final_np = np.array(processed_bubble)
            final_np[final_np[..., 3] > 128, 3] = 255
            final_np[final_np[..., 3] <= 128, 3] = 0
            processed_bubble = Image.fromarray(final_np, mode="RGBA")

            # bubble_layer (alpha setting)
            bubble_layer.paste(processed_bubble, (slot_x, slot_y), processed_bubble)

        else:
            if bubble_shape == "ellipse":
                draw.ellipse(coords, fill=fill_rgba, outline=outline_rgba, width=3 if textbox_Lines=="bold" else 1)
            elif bubble_shape == "rounded_rect":
                draw.rounded_rectangle(coords, radius=20, fill=fill_rgba, outline=outline_rgba, width=3 if textbox_Lines=="bold" else 1)

            elif bubble_shape == "shout":
                # shout style
                cx = slot_x + slot_size_x // 2
                cy = slot_y + slot_size_y // 2
                r = min(slot_size_x, slot_size_y) // 2

                spikes = 12

                points = []

                for i in range(spikes * 2):
                    angle = math.pi * i / spikes
                    if i % 2 == 0:  # inner
                        radius = r * 0.8
                    else:           # outer
                        radius = r * 1.2
                    x = cx + int(radius * math.cos(angle))
                    y = cy + int(radius * math.sin(angle))
                    points.append((x, y))


                draw.polygon(points, fill=fill_rgba, outline=outline_rgba, width=3 if textbox_Lines=="bold" else 1)

            elif bubble_shape == "long_shout":
                # long_shout style
                cx = slot_x + slot_size_x // 2
                cy = slot_y + slot_size_y // 2
                rx = slot_size_x // 2
                ry = slot_size_y // 3


                spikes = 12

                points = []

                for i in range(spikes * 2):
                    angle = math.pi * i / spikes
                    if i % 2 == 0:  # inner
                        radius_x = rx * 0.8
                        radius_y = ry * 0.8

                    else:           # outer
                        radius_x = rx * 1.2
                        radius_y = ry * 1.2

                    x = cx + int(radius_x * math.cos(angle))
                    y = cy + int(radius_y * math.sin(angle))
                    points.append((x, y))

                draw.polygon(points, fill=fill_rgba, outline=outline_rgba, width=3 if textbox_Lines=="bold" else 1)

            elif bubble_shape == "thought":
                # multy bubble style
                draw.ellipse((slot_x+slot_size_x-10, slot_y+slot_size_y * 2//3-10,
                              slot_x+slot_size_x+10, slot_y+slot_size_y * 2//3+10),
                             fill=fill_rgba, outline=outline_rgba, width=3 if textbox_Lines=="bold" else 1)
                draw.ellipse(coords, fill=fill_rgba, outline=outline_rgba, width=3 if textbox_Lines=="bold" else 1)
            
            outline_layer = Image.new("RGBA", pil_img.size, (0,0,0,0))
            outline_draw = ImageDraw.Draw(outline_layer)
            outline_draw.ellipse(coords, outline=outline_rgba, width=2)

        if textbox_blur == "spread":
            bubble_layer = bubble_layer.filter(ImageFilter.GaussianBlur(radius=2))

        # text anchor
        bbox = pil_font.getbbox(text) # 또는 draw.textbbox((0, 0), text, font=pil_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x_offset = bbox[0]
        text_y_offset = bbox[1]

        if text_anchor == "center":
            tx = slot_x + (slot_size_x - text_w) // 2 - text_x_offset
            ty = slot_y + (slot_size_y - text_h) // 2 - text_y_offset
        elif text_anchor == "left-center":
            tx = slot_x + 10 - text_x_offset
            ty = slot_y + (slot_size_y - text_h) // 2 - text_y_offset
        elif text_anchor == "right-center":
            tx = slot_x + slot_size_x - text_w - 10 - text_x_offset
            ty = slot_y + (slot_size_y - text_h) // 2 - text_y_offset
        elif text_anchor == "left-top":
            tx = slot_x + 10 - text_x_offset
            ty = slot_y + 10 - text_y_offset
        elif text_anchor == "right-top":
            tx = slot_x + slot_size_x - text_w - 10 - text_x_offset
            ty = slot_y + 10 - text_y_offset
        elif text_anchor == "left-bottom":
            tx = slot_x + 10 - text_x_offset
            ty = slot_y + slot_size_y - text_h - 10 - text_y_offset
        elif text_anchor == "right-bottom":
            tx = slot_x + slot_size_x - text_w - 10 - text_x_offset
            ty = slot_y + slot_size_y - text_h - 10 - text_y_offset
        else:  # basic (Default center-aligned or top-left)
            tx = slot_x + (slot_size_x - text_w) // 2 - text_x_offset
            ty = slot_y + (slot_size_y - text_h) // 2 - text_y_offset

        # Making text layer
        text_layer = Image.new("RGBA", bubble_layer.size, (0,0,0,0))
        text_draw = ImageDraw.Draw(text_layer)
        
        text_draw.text((tx, ty), text, font=pil_font, fill=text_rgba)

        # text style
        if text_style == "outline":
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx or dy:
                        text_draw.text((tx+dx, ty+dy), text, font=pil_font, fill=text_rgba)

        if text_style == "bold":
            for dx in [0,1]:
                for dy in [0,1]:
                    text_draw.text((tx+dx, ty+dy), text, font=pil_font, fill=text_rgba)
        else:
            pass

        # text shadow
        if text_shadow == "shadow":
            text_draw.text((tx+2, ty+2), text, font=pil_font, fill="#000000")
        else:
            pass

        # lotate textbox
        lotate_slot = max(-180.0, min(lotate_slot, 180.0))
        if lotate_slot != 0.0:
            text_layer = text_layer.rotate(lotate_slot, expand=False)
            bubble_layer = bubble_layer.rotate(lotate_slot, expand=False)

        # alpha blending
        bubble_layer = Image.alpha_composite(bubble_layer, text_layer)
        blended = Image.alpha_composite(pil_img, bubble_layer)

        # PIL → torch tensor
        arr_out = np.array(blended.convert("RGB")).astype(np.float32) / 255.0

        tensor_out = torch.from_numpy(arr_out).permute(2,0,1).unsqueeze(0)

        result = to_tensor_image_output(blended.convert("RGB"))
        
        return IO.NodeOutput(result)

#----------------------------------------

class LogTranslate(IO.ComfyNode):

    @staticmethod
    def google_translate(text, target_lang="en"):
        try:
            result = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return result
        except Exception as e:
            print(f"[Translate] failed translation: {e}")
            return text

    def show_progress(stage, translated=None):
        stages = {
            "start": "Start replace Line...",
            "translate": "Start translate...",
            "waiting": "Waiting for response...",
            "complete": "Translation complete.",
            "print": f"Translated text: {translated}" if translated else "No text"
        }
        print(f"{CYAN}{BOLD}[LogTranslate]{RESET} {stages[stage]}")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LogTranslate",
            display_name="로그 번역기",
            category="커스텀임베딩/특수",
            description="ComfyUI 에러 로그를 줄바꿈/필터링 후 번역하여 에러 지점을 빠르게 확인",
            inputs=[
                IO.String.Input("log_text", multiline=True, tooltip="에러 로그 붙여넣기"),
                IO.Combo.Input("set_log", options=["clean_log", "translate", "highlight"], default="clean_log", tooltip="처리방식"),
                IO.Combo.Input("target_language", options = ["skip", "en", "ja", "ko", "zh-CN", "zh-TW", "es", "pt", "ru", "fr", "la"], default="skip", tooltip="번역될 언어 선택"),
                IO.Combo.Input("set_mode", options=["string", "text"], default="string", tooltip="저장 여부"),
                IO.String.Input("file_name", default="errorlog_%number", tooltip="텍스트로 저장할 에러 로그명"),
                IO.Boolean.Input("use_widget", default=False, tooltip="노드 위젯을 켜거나 끕니다."),
                IO.Int.Input("widgetfont_size", default=14, min=8, max=40, tooltip="노드 위젯이 켜져있을 때 텍스트의 글자 크기를 조정합니다."),
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True,
            outputs=[
                IO.String.Output("clean_log", tooltip="정리된 원문 로그")
            ],
        )

    @classmethod
    def execute(cls, log_text=None, set_log="clean_log", target_language="skip", set_mode="string", file_name="errorlog_%number", use_widget=False, widgetfont_size=14) -> IO.NodeOutput:

        request_text = log_text or ""
        cls.show_progress("start")

        parts = request_text.split("## Logs")
        report_part = parts[0].strip()
        logs_part_raw = parts[1] if len(parts) > 1 else ""

        workflow_cut = logs_part_raw.split("## Attached Workflow")[0]

        # 1. Line break criteria: date pattern
        # ex: 2026-05-28T18:02:31.093702
        logs_lines = re.split(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", workflow_cut)
        logs_part = "\n".join([l.strip() for l in logs_lines if l.strip()])

        # 2. dir type skip
        clean_lines = []
        for line in logs_lines:
            if re.search(r"[A-Z]:\\", line):  # win Dir type
                continue
            if line.strip():
                clean_lines.append(line.strip())

        clean_log = "\n".join(clean_lines)

        # 3. request
        request_text = report_part
            
        cls.show_progress("translate")
        cls.show_progress("waiting")

        # 4. translation
        if target_language == "skip" or set_log == "clean_log":
            translated = request_text
        else:
            translated = cls.google_translate(request_text, target_language)
        cls.show_progress("complete")
        cls.show_progress("print", translated)

        # 5. error keyword
        error_keywords = ["AttributeError", "RuntimeError", "CUDA out of memory", "InternalTorchDynamoError", "ValueError", "TypeError", 
                          "Boolean value of Tensor", "Exception during processing", "Prompt executed in"]
        highlight_lines = [
            line for line in clean_log.splitlines()
            if any(k.lower() in line.lower() for k in error_keywords)
        ]
        highlight = "\n\n=== Highlighted Errors ===\n" + "\n".join(highlight_lines)

        if set_log == "clean_log":
            output = translated + "\n\n===## Logs ===\n" + clean_log
        elif set_log == "translate":
            output = translated + "\n\n===## Logs ===\n" + clean_log
        elif set_log == "highlight":
            output = translated + "\n\n" + highlight
        else:
            output = translated + clean_log

        # 6. save mode
        if set_mode == "text":
            save_dir = os.path.join(folder_paths.base_path, "output", "translations")
            os.makedirs(save_dir, exist_ok=True)
            filename = resolve_filename(file_name, save_dir)
            filepath = os.path.join(save_dir, filename + ".txt")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("=== Original Log ===\n")
                    f.write(log_text + "\n\n")
                    f.write("=== Translated Log ===\n")
                    f.write(output + "\n")
                print(f"{CYAN}{BOLD}[LogTranslate]{RESET} Translation log saved: {filepath}")
            except Exception as e:
                print(f"{CYAN}{BOLD}[LogTranslate]{RESET} Failed to save translation logs: {e}")

        if use_widget:
            return IO.NodeOutput(output, ui={"stats":[output], "font_size":[widgetfont_size]})
        else:
            return IO.NodeOutput(output)

#----------------------------------------

EMB_NODE_CLASS_MAPPINGS = {
    "EasyProjLayerLoad": EasyProjLayerLoad,
    "EasyProjectionLayer": EasyProjectionLayer,
    "EasyEmbeddingLoader": EasyEmbeddingLoader,
    "EasyClipTextEncodeSimple": EasyClipTextEncodeSimple,
    "EasyClipTextEncodeTokenInfo": EasyClipTextEncodeTokenInfo,
    "EasyTranslate": EasyTranslate,
    "EasyClipTextEncodeADV": EasyClipTextEncodeADV,
    "EasyClipTextMask": EasyClipTextMask,
    "EasyClipMultiTextMask": EasyClipMultiTextMask,
    "EasyClipTextMaskNoArea": EasyClipTextMaskNoArea,
    "EasyClipTextMask_And_Image_Using_Vision": EasyClipTextMask_And_Image_Using_Vision,
    "SimpleMaskAreaPrep": SimpleMaskAreaPrep,
    "AutoMaskPrep": AutoMaskPrep,
    "AutoMaskPrep_ADV": AutoMaskPrep_ADV,
    "AutobboxMaskPrep": AutobboxMaskPrep,
    "EasyEmbedLoader": EasyEmbedLoader,
    "EasyEmbedTransform": EasyEmbedTransform,
    "EasyencoderChecker": EasyencoderChecker,
    "EasyEmbeddingChecker": EasyEmbeddingChecker,
    "EasyProjLayerChecker": EasyProjLayerChecker,
    "EasySampler": EasySampler,
    "EasySigmaCalculator": EasySigmaCalculator,
    "LatentMaskPrep": LatentMaskPrep,
    "MaskPrepResizer": MaskPrepResizer,
    "EasyMultyVaeEncode": EasyMultyVaeEncode,
    "EasyInpaintAndConditioning": EasyInpaintAndConditioning,
    "EasyInpaintAndConditioning_modify": EasyInpaintAndConditioning_modify,
    "EasySamplerHook": EasySamplerHook,
    "EasySamplerADV": EasySamplerADV,
    "EasyLoraVersionChecker": EasyLoraVersionChecker,
    "EasyTextslot_Loader": EasyTextslot_Loader,
    "EasyTextslot": EasyTextslot,
    "LogTranslate": LogTranslate,
}


EMB_NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyProjLayerLoad": "프로젝션 레이어 로더",
    "EasyProjectionLayer": "프로젝션 레이어",
    "ProjectionLayer": "임베딩 로더",
    "EasyEmbeddingLoader": "임베딩 로더",
    "EasyClipTextEncodeSimple": "CLIP 텍스트 인코더 (단순형)",
    "EasyClipTextEncodeTokenInfo": "CLIP 텍스트 인코더 (토큰 확인)",
    "EasyTranslate": "번역지원노드",
    "EasyClipTextEncodeADV": "CLIP 텍스트 인코더 어드밴스",
    "EasyClipTextMask": "CLIP 텍스트 마스크",
    "EasyClipMultiTextMask": "CLIP 멀티 텍스트 마스크",
    "EasyClipTextMaskNoArea": "CLIP 텍스트 마스크(하드마스크)",
    "EasyClipTextMask_And_Image_Using_Vision": "CLIP 텍스트 마스크(참고이미지 적용, Clip Vision 내장 필수)",
    "SimpleMaskAreaPrep": "간이 마스크영역 준비기",
    "AutoMaskPrep": "오토 마스크 준비기",
    "AutoMaskPrep_ADV": "오토 마스크 준비기(고급)",
    "AutobboxMaskPrep": "오토 bbox마스크 준비기",
    "EasyEmbedLoader": "BIN PT 임베딩 로더",
    "EasyEmbedTransform": "BIN → EMBEDDING 변환",
    "EasyencoderChecker": "CLIP 인코더 체크(시각화)",
    "EasyEmbeddingChecker": "임베딩 체크(시각화)",
    "EasyProjLayerChecker": "프로젝션 레이어 체크(시각화)",
    "EasySampler": "간단 샘플러",
    "EasySigmaCalculator": "시그마 준비기",
    "LatentMaskPrep": "라텐트 마스크 준비기",
    "MaskPrepResizer": "마스크 벡터 리사이징 준비기",
    "EasyMultyVaeEncode": "간단 Vae 멀티인코드",
    "EasyInpaintAndConditioning": "간단 인페인트 및 컨디셔닝",
    "EasyInpaintAndConditioning_modify": "간단 인페인트 및 컨디셔닝(모드)",
    "EasySamplerHook": "간단 훅 샘플러",
    "EasySamplerADV": "간단 샘플러 어드밴스",
    "EasyLoraVersionChecker": "LoRA 버전 체커",
    "EasyTextslot_Loader": "텍스트 슬롯 로더",
    "EasyTextslot": "텍스트 이미지 기입",
    "LogTranslate": "로그 번역기",
}
#----------------------------------------