import os
import folder_paths
import torch
import numpy as np
import cv2
import easyocr

import comfy.utils
import comfy.sd
from comfy_api.latest import IO, UI

models_dir = folder_paths.models_dir # easyocr (ComfyUI/models/easyocr)
easyocr_dir = os.path.join(models_dir, "easyocr")
os.makedirs(easyocr_dir, exist_ok=True) # Auto model downloading

LANG_MAP = {
    "ko": ["ko"],
    "en": ["en"],
    "ja": ["ja", "en"],
    "zh-CN": ["ch_sim", "en"],
    "zh-TW": ["ch_tra", "en"]
}

def normalize_mask_tensor(mask):

    if not isinstance(mask, torch.Tensor):
        mask_tensor = torch.from_numpy(np.array(mask)).float()
    else:
        mask_tensor = mask.float()

    if mask_tensor.ndim == 2:
        return mask_tensor
    elif mask_tensor.ndim == 3:
        if mask_tensor.shape[0] == 1:
            return mask_tensor.squeeze(0)
        elif mask_tensor.shape[-1] == 1:
            return mask_tensor.squeeze(-1)
        else:
            return mask_tensor[0]
    elif mask_tensor.ndim == 4:
        return mask_tensor[0, 0, :, :]
    elif mask_tensor.ndim == 5:
        return mask_tensor
    else:
        raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")

def ensure_mask_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {t.shape}")
    return t.float()


def ensure_image_nhwc(arr):
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

def imagetocv(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = ensure_image_nhwc(image)
    else:
        arr = np.array(image)

    # 4dim [B, H, W, C] or [B, C, H, W] -> [H, W, C]
    if arr.ndim == 4:
        arr = arr[0]
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        
    # 3dim -> check to [C, H, W] or [H, W, C]
    elif arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0)) #[H, W, C]
        
    # 2dim [H, W]) -> [H, W, 1]
    elif arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    # C = 1 -> C = 3
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]  # del alpha

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr

class EasyOCRText(IO.ComfyNode):
    models_dir = folder_paths.models_dir
    easyocr_dir = os.path.join(models_dir, "easyocr")
    os.makedirs(easyocr_dir, exist_ok=True)
    
    readers = {}

    @classmethod
    def get_reader(cls, text_model):
        LANG_MAP = {
            "ko": ["ko"],
            "en": ["en"],
            "ja": ["ja", "en"],
            "zh-CN": ["ch_sim", "en"],
            "zh-TW": ["ch_tra", "en"],
        }
        if text_model not in cls.readers:
            cls.readers[text_model] = easyocr.Reader(
                LANG_MAP[text_model],
                gpu=torch.cuda.is_available(),
                model_storage_directory=cls.easyocr_dir,
                download_enabled=True
            )
        return cls.readers[text_model]

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyOCRText",
            display_name="OCR 텍스트 파서",
            description="EasyOCR을 사용하여 이미지(또는 마스크 영역)로부터 텍스트를 추출합니다.",
            inputs=[
                IO.Combo.Input("text_model", options=['ko', 'en', 'ja', 'zh-CN', 'zh-TW'], default="ko", tooltip="사용할 언어 모델"),
                IO.Image.Input("image", tooltip="텍스트를 추출할 입력 이미지"),
                IO.Mask.Input("mask", optional=True, tooltip="특정 텍스트 영역만 지정할 마스크 (선택사항)"),
                IO.Int.Input("font_size", default=14, min=8, max=40, step=2, tooltip="폰트 크기"),
                IO.Boolean.Input("show_preview", default=True, tooltip="노드 화면에 위젯 프리뷰를 띄울지 여부"),
            ],
            outputs=[
                IO.String.Output("text", tooltip="추출된 텍스트 결과")
            ],
            category="커스텀임베딩/특수"
        )

    @classmethod
    def execute(cls, image, text_model="ko", mask=None, font_size=14, show_preview=True) -> IO.NodeOutput:
        # 1. IMAGE -> imagetoOpenCV
        img = imagetocv(image)
        
        # 2. Mask processing
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask[0].cpu().numpy() if mask.ndim == 3 else mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            if mask_np.shape[:2] != img.shape[:2]:
                mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

            if mask_np.max() > 1.0:
                mask_np = mask_np / 255.0
            mask_3ch = np.stack([mask_np] * 3, axis=-1)
            img = np.where(mask_3ch > 0.5, img, 255).astype(np.uint8)

        # 3. Execute OCR
        reader = cls.get_reader(text_model)
		
		# 3. create string files
        results = reader.readtext(img)

        text = "\n".join(result[1] for result in results)
        
        if show_preview:

            return IO.NodeOutput(text, ui={"stats":[text], "font_size":[font_size]})

        return IO.NodeOutput(text)

# ComfyUI 노드 등록을 위한 매핑 딕셔너리 예시
OCR_NODE_CLASS_MAPPINGS = {
    "EasyOCRText": EasyOCRText
}

OCR_NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyOCRText": "OCR 텍스트 파서"
}