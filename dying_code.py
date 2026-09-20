# This file contains a list of functions that were moved here after being deprecated from the previous logic.
# Note: Some functions in this file depend on `match_minstd` and `get_processed_noise` from `easyembedding.py`.
# If reusing this code, please include those functions or replace them with your own implementations.

def apply_mask_with_hooks(cond, mask, strength, hooks=None):
    if hooks:
        cond = set_hooks_for_conditioning(cond, hooks)
    _, _, h, w = mask.shape

    cond = node_helpers.conditioning_set_values(cond, {
        "mask": mask,
        "area": (h, w, 0, 0),
        "mask_strength": strength,
        "set_area_to_bounds": False 
    })
    return cond

def safe_mask(mask):
    if mask is None:
        return None

    if isinstance(mask, torch.Tensor):
        m = mask
    else:
        m = to_torch_image(mask)

    if m.dim() == 4:
        m = m.squeeze(0).squeeze(0)
    elif m.dim() == 3:
        m = m.squeeze(0)
    elif m.dim() == 2:
        pass
    else:
        m = m.view(m.shape[-2], m.shape[-1])

    m = m.float()
    return m

def ensure_maskprompt(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, dict):
        if "noise_mask" in t:
            t = t["noise_mask"]
        elif "mask" in t:
            t = t["mask"]
        else:
            raise ValueError("Unsupported dict format for mask")
            
    if isinstance(t, (list, np.ndarray)):
        t = torch.as_tensor(np.asarray(t, dtype=np.float32))
    
    if t.dim() == 2:        # [H, W] -> [B, H, W]
        t = t.unsqueeze(0)
    elif t.dim() == 3:      # [B, H, W] -> pass
        pass
    elif t.dim() == 4:      # [B, 1, H, W] -> [B, H, W]
        t = t[0]
    elif t.dim() == 5:      # [B, 1, D, H, W] -> [B, H, W]
        t = t[0]
    else:
        raise ValueError(f"Unsupported mask type or shape: {t.shape}")

    return t.float()

def resize_mask_to_latentsize(mask):
    if mask is None or mask.numel() == 0:
        return None
    
    # [B, H, W] -> [B, 1, H, W]
    if mask.ndim == 2: mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3: mask = mask.unsqueeze(1)
    
    target_h, target_w = mask.shape[-2] // 8, mask.shape[-1] // 8
    
    mask_resized = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

    mask_resized = (mask_resized > 0.5).float()
    
    return mask_resized

def mask_equalize(mask, target_mask=None):
    m = mask.clone()
    m = ensure_mask_tensor(m)

    if target_mask is not None:
        target_mask = ensure_mask_tensor(target_mask)
        target_shape = target_mask.shape[-2:] # (H, W)
        if m.shape[-2:] != target_shape:
            m = F.interpolate(m, size=target_shape, mode="nearest")
            
    return m

def resize_mask_to_latent(mask, latent_shape=(64,64)):
    if mask is None or mask.numel() == 0:
        return None
    if mask.ndim == 3:  # [B, H, W]
        mask = mask.unsqueeze(1)  # -> [B, 1, H, W]
    mask_resized = F.interpolate(mask, size=latent_shape, mode="nearest")
    return mask_resized.squeeze(1)  # -> [H, W]

def normalize_mask_PIL(mask):
    """
    - RGB → Grayscale
    - RGBA → remove alpha, change RGB Grayscale
    - NCHW / NHWC / HW / HWC
    """

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask.ndim == 4:
        mask = mask[0]

    if mask.ndim == 3 and mask.shape[0] in (1, 3, 4):
        mask = np.transpose(mask, (1, 2, 0))

    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = mask[:, :, :3]
        mask = (0.299 * mask[:, :, 0] +
                0.587 * mask[:, :, 1] +
                0.114 * mask[:, :, 2])

    elif mask.ndim == 3 and mask.shape[2] == 3:
        mask = (0.299 * mask[:, :, 0] +
                0.587 * mask[:, :, 1] +
                0.114 * mask[:, :, 2])

    mask = np.asarray(mask)
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = np.clip(mask / 255.0, 0.0, 1.0)
    else:
        mask = np.clip(mask, 0.0, 1.0)

    if mask.ndim != 2:
        raise ValueError(f"The mask dimension is not 2D: {mask.shape}")

    return mask

def apply_mask(image, mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    return image * mask

def crop_to_mask(pil_img, mask_np):
    """
    mask_np: 0~1 float numpy mask
    """
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return pil_img 

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return pil_img.crop((x1, y1, x2, y2))

def _extract_tensor_any(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = _extract_tensor_any(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _extract_tensor_any(v)
            if t is not None:
                return t
    return None

def apply_weight(cond_tensor, weight):
    return cond_tensor * weight

def deep_type_check(obj, path="root"):
    """Debug code"""
    if isinstance(obj, list):
        print(f"[DEBUG] {path} -> list (len={len(obj)})")
        for i, item in enumerate(obj):
            deep_type_check(item, f"{path}[{i}]")
    elif isinstance(obj, tuple):
        print(f"[DEBUG] {path} -> tuple (len={len(obj)})")
        for i, item in enumerate(obj):
            deep_type_check(item, f"{path}({i})")
    elif isinstance(obj, dict):
        print(f"[DEBUG] {path} -> dict (len={len(obj)})")
        for k, v in obj.items():
            deep_type_check(k, f"{path}.key({repr(k)})")
            deep_type_check(v, f"{path}[{repr(k)}]")
    else:
        print(f"[DEBUG] {path} -> {type(obj).__name__} : {repr(obj)}")

def normalize_keywords(keywords: dict[int, str] | list) -> list[str]:
    normalized: list[str] = []

    def flatten_and_normalize(item):
        if isinstance(item, list):
            for sub in item:
                flatten_and_normalize(sub)
        else:
            kw = str(item).lower().strip()
            if "-" in kw:
                kw = kw.replace("-", " ")
            normalized.append(kw)

    if isinstance(keywords, dict):
        iterable = keywords.values()
    else:
        iterable = keywords

    for kw in iterable:
        flatten_and_normalize(kw)

    print("[DEBUG] normalize_keywords:", normalized, type(normalized))
    return normalized


def collect_keywords(keywords: dict[int, str] | list[str]) -> dict[str, list[str]]:

    if isinstance(keywords, dict):
        iterable = list(keywords.values())
    elif isinstance(keywords, list):
        iterable = keywords
    else:
        raise TypeError(f"[ERROR] keywords는 dict 또는 list여야 합니다. 현재 타입: {type(keywords)}")

    grouped: dict[str, list[str]] = {}
    for kw in normalize_keywords(iterable):
        for group, group_tags in TAG_GROUPS.items():
            if kw in group_tags:
                grouped.setdefault(group, []).append(kw)

    print("[DEBUG] collect_keywords:", grouped, type(grouped))
    return grouped

def progressbar_to_prompt(prompt):

    total_steps = len(prompt)
    pbar = comfy.utils.ProgressBar(int(total_steps))
    for i, ch in enumerate(prompt):
        # prompt node check logic
        pbar.update(i+1)
    return total_steps

def get_valid_fields(input_names):
        valid = []
        for name in input_names:
            val = data_dict.get(name)
            if val and isinstance(val, str) and val.strip():
                valid.append(val.strip())
        return valid

def inject_palette_noise(latent_image, noise_base, palette_image,
                         palette_mode, inject_level, target_size, denoise, device, normalize_noise=False):
    with torch.no_grad():
        if palette_image is None or inject_level <= 0.0 or palette_mode == "default":
            return noise_base

        pal_tensor = palette_image.get("samples", None) if isinstance(palette_image, dict) else palette_image
        if pal_tensor is None:
            return noise_base

    pal_tensor = pal_tensor.to(noise_base.device)
    if pal_tensor.ndim == 4:
        if pal_tensor.shape[-2:] != latent_image.shape[-2:]:
            pal_tensor = F.interpolate(pal_tensor, size=latent_image.shape[-2:], mode="bilinear", align_corners=False)
    elif pal_tensor.ndim == 5:
        if pal_tensor.shape[-3:] != latent_image.shape[-3:]:
            pal_tensor = F.interpolate(pal_tensor, size=latent_image.shape[-3:], mode="trilinear", align_corners=False)
    else:
        raise ValueError(f"Unsupported latent dimension: {pal_tensor.ndim}")

    pal_norm = match_minstd(pal_tensor, noise_base)
    pal_norm = get_processed_noise(pal_norm, normalize_noise)

    if palette_mode == "blend":
        return noise_base + (pal_norm * inject_level * 0.1)
    elif palette_mode == "color_palette":
        return noise_base + (pal_norm * denoise * inject_level)
    elif palette_mode == "overlay":
        overlay = pal_norm * denoise
        return noise_base * (1.0 - inject_level) + overlay * inject_level
    else:
        return noise_base