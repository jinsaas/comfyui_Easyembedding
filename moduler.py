import os
import torch
import folder_paths
import textwrap

from safetensors import safe_open
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import comfy.utils
import comfy.sd
from comfy_api.latest import IO, UI

#----------------------------------------
# Header Utillity
#----------------------------------------

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _optimized_walk(base_dir):

    files = []
    if not os.path.exists(base_dir):
        return files
    for root, dirs, filenames in os.walk(base_dir, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_')) and d not in ('__pycache__', 'node_modules')]
        for f in filenames:
            if f.endswith((".pt", ".safetensors", ".bin")):
                rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                rel_path = rel_path.replace("\\", "/")
                files.append(rel_path)
    return files


def check_model_type(file_path: str):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    file_name = os.path.basename(file_path)
    total_elements = 0
    dtypes = set()
    tensor_names = []
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            for name in tensor_names:
                tensor_slice = f.get_slice(name)
                shape = tensor_slice.get_shape()
                dt = tensor_slice.get_dtype()
                dtypes.add(str(dt))
                
                elements = 1
                for dim in shape:
                    elements *= dim
                total_elements += elements
    except Exception as e:
        print(f"[Error] Failed to parse safetensors: {e}")
        return

    if not tensor_names:
        print(f"File: {file_name} -> Category: 7. No Metadata / Unknown")
        return

    names_lower = [name.lower() for name in tensor_names]
    is_quantized = any(any(q in n for q in ["qweight", "scales", "zeros", "quant", "ggml"]) for n in names_lower)
    is_cosmos = any("cosmos" in n or "actor" in n or "net.blocks" in n for n in names_lower)
    is_dit = any("diffusion_model" in n or "transformer_blocks" in n or "x_embedder" in n for n in names_lower)
    is_unet = any("model.diffusion_model.input_blocks" in n or "down_blocks" in n for n in names_lower)

    if is_cosmos:
        category = "6. Quantized Cosmos" if is_quantized else "3. Standard Cosmos"
    elif is_dit or is_unet:
        if is_unet:
            category = "4. Quantized SD" if is_quantized else "1. Standard SD"
        else:
            category = "5. Quantized DiT" if is_quantized else "2. Standard DiT"
    else:
        category = "5. Quantized DiT" if is_quantized else "7. No Metadata / Unknown"

    bytes_per_elem = 0.5 if is_quantized else 2.0
    estimated_vram = (total_elements * bytes_per_elem) / (1024 ** 3) + 1.5

    print("=" * 50)
    print(f"File Name      : {file_name}")
    print(f"Parameters     : ~{total_elements / 1e9:.2f}B")
    print(f"Detected Dtype : {list(dtypes)}")
    print(f"Estimated VRAM : ~{estimated_vram:.1f} GB")
    print(f"Model Category : {category}")
    print("=" * 50)

_clip_cache = None  # Cache variable
_cliplast_mtime = 0  # For detecting folder state changes

_unet_cache = None
_unetlast_mtime = 0  # For detecting folder state changes

_folder_cache = None
_vaelast_mtime = 0  # For detecting folder state changes
#----------------------------------------
# node base
#----------------------------------------

class EasyencoderLoader(IO.ComfyNode):

    @classmethod
    def show_status(cls, stage):
        stages = {
            "loading": "Loading CLIP model into memory...",
            "complete": "CLIP model load complete!"
        }
        print(f"{GREEN}{BOLD}[EasyencoderLoader]{RESET} {stages.get(stage, stage)}")

    @classmethod
    def _get_clip_files(cls, force_clear=False):
        global _clip_cache, _cliplast_mtime  # call global code
        if force_clear: # Reset Cache
            _clip_cache = None
            _cliplast_mtime = 0
            print(f"{GREEN}{BOLD}[EasyencoderLoader] Cache cleared manually by user!{RESET}")
            
        clip_dirs = folder_paths.get_folder_paths("clip")
        current_mtime = sum(os.path.getmtime(d) for d in clip_dirs if os.path.exists(d))
        if _clip_cache is not None and _cliplast_mtime == current_mtime:
            return _clip_cache

        files = []
        for d in clip_dirs:
            files.extend(_optimized_walk(d))
        files = files if files else folder_paths.get_filename_list("clip")

        _clip_cache = files  # Caching
        _cliplast_mtime = current_mtime
        return files

    @classmethod
    def define_schema(cls):
        files = cls._get_clip_files()

        return IO.Schema(
            node_id="EasyencoderLoader",
            display_name="이지 인코더 로더",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 CLIP 텍스트 인코더를 로드합니다.",
            inputs=[
                IO.Combo.Input("clip_name", options=files if files else [""], default=files[0] if files else "", tooltip="로드할 텍스트 인코더 파일"),
                IO.Combo.Input("cliptype", options=["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2", "joyimage", "mage", "minimax"], default="stable_diffusion", tooltip="클립 모델 타입"),
                IO.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True, tooltip="실행 디바이스 설정"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="캐시 초기화.노드에 준비한 모델을 언캐싱합니다. 재탐색 시간이 소요됩니다.")
            ],
            outputs=[
                IO.Clip.Output("CLIP", tooltip="로드된 클립 모델")
            ],
        )

    @classmethod
    def execute(cls, clip_name, cliptype="stable_diffusion", device="default", clear_cache=False) -> IO.NodeOutput:
        if clear_cache:
            cls._get_clip_files(force_clear=True)

        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        if not clip_path or not os.path.exists(clip_path):
            raise FileNotFoundError(f"[EasyencoderLoader] Cannot find the CLIP file: {clip_name}")

        cls.show_status("loading")
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        upper_type = cliptype.upper()
        if not hasattr(comfy.sd.CLIPType, upper_type):
            print(f"{YELLOW}{BOLD}[EasyencoderLoader] Warning: Unknown CLIP type '{cliptype}'. Falling back to STABLE_DIFFUSION.{RESET}")
            upper_type = "STABLE_DIFFUSION"
        clip_type = getattr(comfy.sd.CLIPType, upper_type)

        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path], 
                embedding_directory=folder_paths.get_folder_paths("embeddings"), 
                clip_type=clip_type, 
                model_options=model_options
            )
        except Exception as e:
            raise RuntimeError(f"[EasyencoderLoader] Failed to load CLIP model '{clip_name}': {e}")

        cls.show_status("complete")
        return IO.NodeOutput(clip)

#----------------------------------------

class EasyUnetLoader(IO.ComfyNode):

    @classmethod
    def show_status(cls, stage):
        stages = {
            "loading": "Loading UNet model into memory...",
            "complete": "UNet model load complete!"
        }
        print(f"{GREEN}{BOLD}[EasyUnetLoader]{RESET} {stages.get(stage, stage)}")

    @classmethod
    def _get_unet_files(cls, force_clear=False):
        global _unet_cache, _unetlast_mtime  # call global code
        if force_clear: # Reset Cache
            _unet_cache = None
            _unetlast_mtime = 0
            print(f"{GREEN}{BOLD}[EasyUnetLoader] Cache cleared manually by user!{RESET}")

        unet_dirs = folder_paths.get_folder_paths("diffusion_models")
        current_mtime = sum(os.path.getmtime(d) for d in unet_dirs if os.path.exists(d))
        if _unet_cache is not None and _unetlast_mtime == current_mtime:
            return _unet_cache

        files = []
        for d in unet_dirs:
            files.extend(_optimized_walk(d))
        files = files if files else folder_paths.get_filename_list("diffusion_models")

        _unet_cache = files  # Caching
        _unetlast_mtime = current_mtime
        return files

    @classmethod
    def define_schema(cls):
        files = cls._get_unet_files()

        return IO.Schema(
            node_id="EasyUnetLoader",
            display_name="이지 UNet 로더",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 UNet/DiT/Cosmos 모델을 로드합니다.",
            inputs=[
                IO.Combo.Input("unet_name", options=files if files else [""], default=files[0] if files else "", tooltip="로드할 UNET/DiT/cosmos 모델 파일"),
                IO.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default", advanced=True, tooltip="가중치 데이터타입 설정"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="캐시 초기화.노드에 준비한 모델을 언캐싱합니다. 재탐색 시간이 소요됩니다.")
            ],
            outputs=[
                IO.Model.Output("MODEL", tooltip="로드된 디퓨전 모델")
            ],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype="default", clear_cache=False) -> IO.NodeOutput:
        if clear_cache:
            cls._get_unet_files(force_clear=True)

        model_options = {}
        
        if weight_dtype.startswith("fp8"):
            target_dtype = torch.float8_e4m3fn if "e4m3fn" in weight_dtype else torch.float8_e5m2
            if not hasattr(torch, "float8_e4m3fn"):
                print(f"{YELLOW}{BOLD}[EasyUnetLoader] Warning: Current PyTorch does not support FP8. Falling back to default.{RESET}")
                weight_dtype = "default"
            else:
                model_options["dtype"] = target_dtype
                if weight_dtype == "fp8_e4m3fn_fast":
                    model_options["fp8_optimizations"] = True

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        if not unet_path or not os.path.exists(unet_path):
            raise FileNotFoundError(f"[EasyUnetLoader] Cannot find the UNet file: {unet_name}")

        cls.show_status("loading")
        try:
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        except Exception as e:
            raise RuntimeError(f"[EasyUnetLoader] Failed to load UNet model '{unet_name}': {e}")
            
        cls.show_status("complete")
        return IO.NodeOutput(model)


#----------------------------------------

class EasyVAELoader(IO.ComfyNode):
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2"]

    @classmethod
    def show_status(cls, stage):
        stages = {
            "loading": "Loading VAE model into memory...",
            "complete": "VAE model load complete!"
        }
        print(f"{GREEN}{BOLD}[EasyVAELoader]{RESET} {stages.get(stage, stage)}")

    @classmethod
    def _get_recursive_files(cls, folder_name, force_clear=False):
        global _folder_cache, _vaelast_mtime  # call global code
        if force_clear: # Reset Cache
            _folder_cache = None
            _vaelast_mtime = 0
            print(f"{GREEN}{BOLD}[EasyVAELoader] Cache cleared manually by user!{RESET}")

        dirs = folder_paths.get_folder_paths(folder_name)
        current_mtime = sum(os.path.getmtime(d) for d in dirs if os.path.exists(d))
        
        if _folder_cache is not None and _vaelast_mtime == current_mtime:
            return _folder_cache

        files = []
        for d in dirs:
            files.extend(_optimized_walk(d))
        _folder_cache = files  # Caching
        _vaelast_mtime = current_mtime
        return files

    @classmethod
    def vae_list(cls):
        vaes = cls._get_recursive_files("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        
        have_img_encoder, have_img_decoder = set(), set()
        for v in approx_vaes:
            parts = v.split("_", 1)
            if len(parts) != 2 or parts[0] not in cls.image_taes:
                for tae in cls.video_taes:
                    if v.startswith(tae):
                        vaes.append(v)
                        break
                continue

            if parts[1].startswith("encoder."):
                have_img_encoder.add(parts[0])
            elif parts[1].startswith("decoder."):
                have_img_decoder.add(parts[0])

        vaes += [k for k in have_img_decoder if k in have_img_encoder]
        vaes.append("pixel_space")
        return vaes

    @classmethod
    def _get_taesd_path(cls, name, suffix):
        return next(
            (os.path.join(d, f) for d in folder_paths.get_folder_paths("vae_approx") 
             for root, _, files in os.walk(d, followlinks=True) 
             for f in files if f.startswith(f"{name}_{suffix}.")),
            None
        )

    @classmethod
    def load_taesd(cls, name):
        sd = {}
        encoder_path = cls._get_taesd_path(name, "encoder")
        decoder_path = cls._get_taesd_path(name, "decoder")

        if not encoder_path or not decoder_path:
            raise FileNotFoundError(f"[EasyVAELoader] Cannot find TAESD files for {name}")

        enc = comfy.utils.load_torch_file(encoder_path)
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(decoder_path)
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        scales = {
            "taesd": (0.18215, 0.0),
            "taesdxl": (0.13025, 0.0),
            "taesd3": (1.5305, 0.0609),
            "taef1": (0.3611, 0.1159)
        }
        if name in scales:
            scale, shift = scales[name]
            sd["vae_scale"] = torch.tensor(scale)
            sd["vae_shift"] = torch.tensor(shift)
        return sd

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyVAELoader",
            display_name="이지 VAE 로더",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 VAE 및 TAESD 모델을 로드합니다.",
            inputs=[
                IO.Combo.Input("vae_name", options=cls.vae_list(), tooltip="로드할 VAE 파일"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="캐시 초기화.노드에 준비한 모델을 언캐싱합니다. 재탐색 시간이 소요됩니다.")
            ],
            outputs=[
                IO.Vae.Output("VAE", tooltip="로드된 VAE 모델")
            ],
        )

    @classmethod
    def execute(cls, vae_name, clear_cache=False) -> IO.NodeOutput:
        if clear_cache:
            cls._get_recursive_files("vae", force_clear=True)

        metadata = None
        vae_path = None
        
        cls.show_status("loading")
        if vae_name == "pixel_space":
            sd = {}
            sd["pixel_space_vae"] = torch.tensor(1.0)
        elif vae_name in cls.image_taes:
            sd = cls.load_taesd(vae_name)
        else:
            for d in folder_paths.get_folder_paths("vae"):
                p = os.path.join(d, vae_name)
                if os.path.exists(p):
                    vae_path = p
                    break

            if not vae_path:
                for d in folder_paths.get_folder_paths("vae_approx"):
                    p = os.path.join(d, vae_name)
                    if os.path.exists(p):
                        vae_path = p
                        break
            
            if not vae_path:
                raise FileNotFoundError(f"[EasyVAELoader] Cannot find the VAE file: {vae_name}")
                
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)

        if vae_name == "taef2":
            if metadata is None:
                metadata = {"tae_latent_channels": 128}
            else:
                metadata["tae_latent_channels"] = 128

        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        
        if vae_path is not None:
            vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (vae_path, metadata, None))
            
        cls.show_status("complete")
        return IO.NodeOutput(vae)


#----------------------------------------

class EasyUnetClipLoader(IO.ComfyNode):

    @classmethod
    def show_status(cls, stage):
        stages = {
            "unetloading": "Loading UNet models into memory...",
            "unetcomplete": "UNet model load complete!",
            "cliploading": "Loading CLIP models into memory...",
            "complete": "CLIP model load complete!"
        }
        print(f"{GREEN}{BOLD}[EasyUnetClipLoader]{RESET} {stages.get(stage, stage)}")

    @classmethod
    def _get_unet_files(cls, force_clear=False):
        global _unet_cache, _unetlast_mtime  # call global code
        if force_clear:
            _unet_cache = None
            _unetlast_mtime = 0
            print("[EasyUnetClipLoader] UNet cache cleared manually!")

        unet_dirs = folder_paths.get_folder_paths("diffusion_models")
        current_unetmtime = sum(os.path.getmtime(d) for d in unet_dirs if os.path.exists(d))

        if _unet_cache is not None and _unetlast_mtime == current_unetmtime:
            return _unet_cache

        files = []
        for d in unet_dirs:
            files.extend(_optimized_walk(d))
        files = files if files else folder_paths.get_filename_list("diffusion_models")

        _unet_cache = files
        _unetlast_mtime = current_unetmtime
        return files

    @classmethod
    def _get_clip_files(cls, force_clear=False):
        global _clip_cache, _cliplast_mtime  # call global code
        if force_clear:
            _clip_cache = None
            _cliplast_mtime = 0
            print("[EasyUnetClipLoader] CLIP cache cleared manually!")

        clip_dirs = folder_paths.get_folder_paths("clip")
        current_clipmtime = sum(os.path.getmtime(d) for d in clip_dirs if os.path.exists(d))

        if _clip_cache is not None and _cliplast_mtime == current_clipmtime:
            return _clip_cache

        files = []
        for d in clip_dirs:
            files.extend(_optimized_walk(d))
        files = files if files else folder_paths.get_filename_list("clip")

        _clip_cache = files
        _cliplast_mtime = current_clipmtime
        return files

    @classmethod
    def define_schema(cls):
        unet_files = cls._get_unet_files()
        clip_files = cls._get_clip_files()

        return IO.Schema(
            node_id="EasyUnetClipLoader",
            display_name="이지 UNet & CLIP 통합 로더",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 디퓨전 모델(UNet/DiT)과 텍스트 인코더(CLIP)를 동시에 로드합니다.",
            inputs=[
                IO.Combo.Input("unet_name", options=unet_files if unet_files else [""], default=unet_files[0] if unet_files else "", tooltip="로드할 디퓨전 모델 파일"),
                IO.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default", advanced=True, tooltip="UNet 가중치 데이터타입 설정"),
                IO.Combo.Input("clip_name", options=clip_files if clip_files else [""], default=clip_files[0] if clip_files else "", tooltip="로드할 텍스트 인코더 파일"),
                IO.Combo.Input("cliptype", options=["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2", "joyimage", "mage", "minimax"], default="stable_diffusion", tooltip="클립 모델 타입"),
                IO.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True, tooltip="CLIP 실행 디바이스 설정"),
                IO.Boolean.Input("unet_clear_cache", default=False, tooltip="캐시 초기화.노드에 준비한 모델을 언캐싱합니다. 재탐색 시간이 소요됩니다."),
                IO.Boolean.Input("clip_clear_cache", default=False, tooltip="캐시 초기화.노드에 준비한 모델을 언캐싱합니다. 재탐색 시간이 소요됩니다.")
            ],
            outputs=[
                IO.Model.Output("MODEL", tooltip="로드된 디퓨전 모델"),
                IO.Clip.Output("CLIP", tooltip="로드된 클립 모델")
            ],
        )

    @classmethod
    def execute(cls, unet_name, clip_name, weight_dtype="default", cliptype="stable_diffusion", device="default", unet_clear_cache=False, clip_clear_cache=False) -> IO.NodeOutput:
        if unet_clear_cache:
            cls._get_unet_files(force_clear=True)
        if clip_clear_cache:
            cls._get_clip_files(force_clear=True)

        cls.show_status("loading")

        # 1. UNet loading
        cls.show_status("unetloading")
        model_options = {}
        if weight_dtype.startswith("fp8"):
            target_dtype = torch.float8_e4m3fn if "e4m3fn" in weight_dtype else torch.float8_e5m2
            if not hasattr(torch, "float8_e4m3fn"):
                print(f"{YELLOW}{BOLD}[EasyUnetClipLoader] Warning: Current PyTorch does not support FP8. Falling back to default.{RESET}")
                weight_dtype = "default"
            else:
                model_options["dtype"] = target_dtype
                if weight_dtype == "fp8_e4m3fn_fast":
                    model_options["fp8_optimizations"] = True

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        if not unet_path or not os.path.exists(unet_path):
            raise FileNotFoundError(f"[EasyUnetClipLoader] Cannot find the UNet file: {unet_name}")

        try:
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        except Exception as e:
            raise RuntimeError(f"[EasyUnetClipLoader] Failed to load UNet model '{unet_name}': {e}")
            
        cls.show_status("unetcomplete")

        # 2. Clip loading
        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        if not clip_path or not os.path.exists(clip_path):
            raise FileNotFoundError(f"[EasyUnetClipLoader] Cannot find the CLIP file: {clip_name}")

        cls.show_status("cliploading")
        clip_model_options = {}
        if device == "cpu":
            clip_model_options["load_device"] = clip_model_options["offload_device"] = torch.device("cpu")

        upper_type = cliptype.upper()
        if not hasattr(comfy.sd.CLIPType, upper_type):
            print(f"{YELLOW}{BOLD}[EasyUnetClipLoader] Warning: Unknown CLIP type '{cliptype}'. Falling back to STABLE_DIFFUSION.{RESET}")
            upper_type = "STABLE_DIFFUSION"
        clip_type = getattr(comfy.sd.CLIPType, upper_type)

        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path], 
                embedding_directory=folder_paths.get_folder_paths("embeddings"), 
                clip_type=clip_type, 
                model_options=clip_model_options
            )
        except Exception as e:
            raise RuntimeError(f"[EasyUnetClipLoader] Failed to load CLIP model '{clip_name}': {e}")

        cls.show_status("complete")
        return IO.NodeOutput(model, clip)

#----------------------------------------

class EasyUnetChecker(IO.ComfyNode):
    
    @classmethod
    def show_status(cls, stage):
        stages = {
            "checking": "Checking model info & path...",
            "loading": "Loading model into memory...",
            "complete": "Model load complete!"
        }
        color = GREEN if stage == "complete" else CYAN
        print(f"{color}{BOLD}[EasyUnetChecker]{RESET} {stages.get(stage, stage)}")

    @classmethod
    def _get_unet_files(cls):
        unet_dirs = folder_paths.get_folder_paths("diffusion_models")
        files = []
        for d in unet_dirs:
            files.extend(_optimized_walk(d))
        return files if files else folder_paths.get_filename_list("diffusion_models")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyUnetChecker",
            display_name="UNet/DiT 모델 분석기",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 선택된 UNet 또는 DiT 모델 파일을 분석하여 파라미터 수, VRAM 추정치, 데이터타입 등을 시각화합니다.",
            inputs=[
                IO.Combo.Input("unet_name", options=cls._get_unet_files(),
                              tooltip="확인할 UNET/DiT 모델 파일"),
                IO.Int.Input("font_size", default=14, min=8, max=40, step=2, tooltip="출력 이미지 폰트 크기"),
                IO.Boolean.Input("show_preview", default=True, tooltip="노드 화면에 위젯 프리뷰를 띄울지 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.String.Output("text", tooltip="분석 결과 텍스트")]
        )

    @classmethod
    def execute(cls, unet_name, font_size=14, show_preview=True) -> IO.NodeOutput:
        cls.show_status("checking")
        try:
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        except Exception:
            unet_path = None

        file_name = os.path.basename(unet_path) if unet_path else unet_name

        cls.show_status("loading")
        total_elements = 0
        dtypes = set()
        tensor_names = []

        if unet_path and os.path.exists(unet_path):
            try:
                with safe_open(unet_path, framework="pt", device="cpu") as f:
                    tensor_names = list(f.keys())
                    for name in tensor_names:
                        tensor_slice = f.get_slice(name)
                        shape = tensor_slice.get_shape()
                        dt = tensor_slice.get_dtype()
                        dtypes.add(str(dt))
                        elements = np.prod(shape)
                        total_elements += elements
            except Exception as e:
                print(f"[UnetChecker] Failed to read tensors: {e}")
                tensor_names = []

        category = "Unknown"
        if tensor_names:
            names_lower = [name.lower() for name in tensor_names]
            is_quantized = any(q in n for n in names_lower for q in ["qweight", "scales", "zeros", "quant", "ggml"])
            is_cosmos = any("cosmos" in n or "actor" in n or "net.blocks" in n for n in names_lower)
            is_dit = any("diffusion_model" in n or "transformer_blocks" in n or "x_embedder" in n for n in names_lower)
            is_unet = any("model.diffusion_model.input_blocks" in n or "down_blocks" in n for n in names_lower)

            if is_cosmos:
                category = "Cosmos - Quantized" if is_quantized else "Cosmos - Standard"
            elif is_dit:
                category = "DiT - Quantized" if is_quantized else "DiT - Standard"
            elif is_unet:
                category = "SD - Quantized" if is_quantized else "SD - Standard"
            else:
                category = "Unknown"

        bytes_per_elem = 0.5 if ("Quantized" in category) else 2.0
        estimated_vram = (total_elements * bytes_per_elem) / (1024 ** 3) + 1.5 if total_elements > 0 else 0.0

        found_attn_keys = {}
        
        search_patterns = [
            "qkv", "query_key_value",
            "to_q", "to_k", "to_v",
            "query", "key", "value",
            "q", "k", "v"
        ]

        for pattern in search_patterns:
            if pattern in found_attn_keys:
                continue
            
            for name in tensor_names:
                name_lower = name.lower()
                if len(pattern) == 1:
                    if f".{pattern}." in name_lower or name_lower.endswith(f".{pattern}") or f"_{pattern}." in name_lower or name_lower.startswith(f"{pattern}."):
                        found_attn_keys[pattern] = name
                        break
                else:
                    if pattern in name_lower:
                        found_attn_keys[pattern] = name
                        break


        # textwrap
        wrapped_file = textwrap.fill(f"File: {file_name}", width=35)
        raw_lines = [
            wrapped_file,
            f"Parameters: ~{total_elements / 1e9:.2f}B",
            f"Est. VRAM: ~{estimated_vram:.1f} GB",
            f"Dtype: {list(dtypes)[:2] if dtypes else ['N/A']}",
            "Category:",
            f"> {category}"
        ]

        if found_attn_keys:
            raw_lines.append("Attn Keys (Found):")
            for pat, real_name in found_attn_keys.items():
                raw_lines.append(f"• [{pat}] {os.path.basename(real_name) if len(real_name) > 30 else real_name}")

        text_output_str = "\n".join(raw_lines)

        if show_preview:

            cls.show_status("complete")
            return IO.NodeOutput(text_output_str, ui={"stats":[text_output_str], "font_size":[font_size]})

        cls.show_status("complete")
        return IO.NodeOutput(text_output_str)

#----------------------------------------

class EasyVAEChecker(IO.ComfyNode):
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2"]

    @classmethod
    def define_schema(cls, **kwargs):
        vae_dirs = folder_paths.get_folder_paths("vae")
        files = []
        for d in vae_dirs:
            files.extend(_optimized_walk(d))

        return IO.Schema(
            node_id="EasyVAEChecker",
            display_name="VAE 체크 및 인코딩 테스트",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 선택한 VAE의 구조 분석, VRAM 추정 및 실제 Encode/Decode 테스트 결과를 시각화합니다.",
            inputs=[
                IO.Combo.Input("vae_name", options=files, default=files[0] if files else "", tooltip="확인할 VAE 파일"),
                IO.Int.Input("font_size", default=14, min=8, max=40, step=2, tooltip="폰트 크기"),
                IO.Boolean.Input("show_preview", default=True, tooltip="노드 화면에 위젯 프리뷰를 띄울지 여부"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.String.Output("text", tooltip="VAE 정보 및 테스트 결과")
            ],
        )

    @classmethod
    def load_taesd_for_checker(cls, name):
        sd = {}
        def _get_path(n, suffix):
            return next(
                (os.path.join(d, f) for d in folder_paths.get_folder_paths("vae_approx") 
                 for root, _, files in os.walk(d, followlinks=True) 
                 for f in files if f.startswith(f"{n}_{suffix}.")),
                None
            )
        enc_path = _get_path(name, "encoder")
        dec_path = _get_path(name, "decoder")
        if enc_path and os.path.exists(enc_path):
            enc = comfy.utils.load_torch_file(enc_path)
            for k in enc: sd[f"taesd_encoder.{k}"] = enc[k]
        if dec_path and os.path.exists(dec_path):
            dec = comfy.utils.load_torch_file(dec_path)
            for k in dec: sd[f"taesd_decoder.{k}"] = dec[k]
        return sd

    @classmethod
    def execute(cls, vae_name, font_size=14, show_preview=True) -> IO.NodeOutput:
        vae_type = "Unknown"
        vae_style = "Unknown"
        unet_layer_info = "Not Detected (DiT / Transformer or Custom)"
        latent_channels = "Unknown"
        latent_ndim_str = "Unknown"
        encode_status = "Skipped"
        total_elements = 0

        sd = {}
        if vae_name == "pixel_space":
            vae_type = "Pixel Space"
            vae_style = "No VAE (Direct)"
        elif vae_name in cls.image_taes:
            vae_type = "TAESD (Approx)"
            vae_style = "Lightweight VAE"
            sd = cls.load_taesd_for_checker(vae_name)
        else:
            vae_dirs = folder_paths.get_folder_paths("vae")
            vae_path = None
            for d in vae_dirs:
                path = os.path.join(d, vae_name)
                if os.path.exists(path):
                    vae_path = path
                    break
            
            if not vae_path:
                for d in folder_paths.get_folder_paths("vae_approx"):
                    path = os.path.join(d, vae_name)
                    if os.path.exists(path):
                        vae_path = path
                        break

            if vae_path:
                try:
                    if vae_path.endswith(".pt"):
                        sd = torch.load(vae_path, map_location="cpu")
                        vae_type = "PT"
                    elif vae_path.endswith(".safetensors"):
                        with safe_open(vae_path, framework="pt", device="cpu") as f:
                            sd = {k: f.get_tensor(k) for k in f.keys()}
                        vae_type = "Safetensors"
                except Exception:
                    sd = {}

        unet_channels_found = None
        has_unet_keys = False
        unet_layer_count = 0

        for k, v in sd.items():
            if hasattr(v, "numel"):
                total_elements += v.numel()

            # UNet core block key pattern detection
            if "encoder.conv_out.weight" in k or "decoder.conv_in.weight" in k:
                has_unet_keys = True
            if "encoder" in k or "decoder" in k:
                if "conv" in k or "block" in k:
                    unet_layer_count += 1

        if has_unet_keys:
            vae_style = "SD based (UNet-based VAE)"
            unet_info = f"Detected (UNet Layers: {unet_layer_count})"
        else:
            # If the UNet key cannot be found, it is considered to be of the DiT/Transformer class or a custom architecture
            vae_style = "DiT / Transformer / Custom VAE"
            unet_info = "None (Non-UNet Architecture)"

        estimated_vram = (total_elements * 2.0) / (1024 ** 3) + 0.8 if total_elements > 0 else 0.2

        try:
            metadata = None
            if vae_name == "taef2":
                metadata = {"tae_latent_channels": 128}
            
            vae_obj = comfy.sd.VAE(sd=sd, metadata=metadata)

            generator = torch.Generator(device="cpu")
            generator.manual_seed(42)
            test_image = torch.rand((1, 512, 512, 3), dtype=torch.float32, generator=generator)
            
            encoded_latent = vae_obj.encode(test_image[:, :, :, :3])
            
            if isinstance(encoded_latent, dict) and "samples" in encoded_latent:
                lat_tensor = encoded_latent["samples"]
            else:
                lat_tensor = encoded_latent

            # Step2: Parsing the exact number of channels from the shape of the actual encoded latent tensor
            if hasattr(lat_tensor, "ndim") and lat_tensor.ndim >= 4:
                # [B, C, H, W]
                real_lat_channels = lat_tensor.shape[1]
                latent_channels = str(real_lat_channels)
            else:
                latent_channels = "Unknown"

            latent_ndim_str = f"ndim: {lat_tensor.ndim} | shape: {list(lat_tensor.shape)}"

            _ = vae_obj.decode(lat_tensor)
            encode_status = "Success (Encode/Decode OK)"
        except Exception as e:
            encode_status = f"Failed: {str(e)[:25]}..."
            latent_ndim_str = "ndim: N/A (Error)"

        raw_lines = [
            f"Format: {vae_type}",
            f"Style: {vae_style}",
            f"UNet Layers: {unet_info}",
            f"Latent Channels: {latent_channels}",
            f"Est. VRAM: ~{estimated_vram:.2f} GB",
            f"Latent Tensor ({latent_ndim_str})",
            f"Test Result: {encode_status}",
            f"Canvas Tested: {512}x{512}"
        ]

        text_output_str = "\n".join(raw_lines)

        if show_preview:

            return IO.NodeOutput(text_output_str, ui={"stats":[text_output_str], "font_size":[font_size]})

        return IO.NodeOutput(text_output_str)


#----------------------------------------
class EasyControlNetChecker(IO.ComfyNode):
    @classmethod
    def show_status(cls, stage):
        stages = {
            "checking": "Checking controlnet info & path...",
            "loading": "Loading controlnet into memory...",
            "complete": "ControlNet load & analysis complete!"
        }
        print(f"[EasyControlNetChecker] {stages.get(stage, stage)}")
    @classmethod
    def _get_controlnet_files(cls, **kwargs):
        controlnet_dirs = folder_paths.get_folder_paths("controlnet")
        files = []
        for d in controlnet_dirs:
            files.extend(_optimized_walk(d))
        return files if files else folder_paths.get_filename_list("controlnet")

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EasyControlNetChecker",
            display_name="ControlNet 모델 분석기",
            category="커스텀임베딩/특수",
            description="하위 폴더도 탐색하며 선택된 ControlNet 모델 파일을 분석하여 파라미터, VRAM 추정치, 데이터타입 등을 시각화합니다.\n"
                        "위젯은 한번만 생성되며, 위젯 호출방식이 맘에 안 들면 껐다가 다시 키신 뒤 위젯옵션을 건드리고 사용하세요",
            inputs=[
                IO.Combo.Input("control_net_name", options=cls._get_controlnet_files(), tooltip="확인할 컨트롤넷 모델 파일"),
                IO.Int.Input("font_size", default=12, min=8, max=40, tooltip="노드 위젯에만 적용됩니다. 폰트 크기를 적용할 수 있습니다.(2.0 텍스트에어리어 위젯에선 적용되지 않습니다.)"),
                IO.Boolean.Input("use_widget_2_0", default=False, tooltip="켜면 2.0 텍스트에어리어 위젯으로 표시합니다. 꺼져 있으면 1.0 위젯으로 표시합니다.\n"
                                 "위젯 2.0 모드의 경우 실제로 해당 위젯 지원상태여야 합니다."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.String.Output("text", tooltip="ControlNet 정보 및 테스트 결과")
            ],
        )

    @classmethod
    def execute(self, control_net_name, font_size=12, use_widget_2_0=False) -> IO.NodeOutput:
        self.show_status("checking")
        path = None
        for d in folder_paths.get_folder_paths("controlnet"):
            candidate = os.path.join(d, control_net_name)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f"{control_net_name} not found in controlnet directories.")

        self.show_status("loading")
        
        total_layers = 0
        qkv_matched_count = 0
        total_params = 0
        param_bytes = 0
        load_mode = "Safetensors Header & Key-Pattern Parser"
        
        first_layer = None
        last_layer = None
        metadata = {}
        keys_list = []

        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                try:
                    metadata = f.metadata() or {}
                except Exception:
                    metadata = {}
                    
                keys_list = list(f.keys())
                total_layers = len(keys_list)
                
                if total_layers > 0:
                    first_layer = keys_list[0]
                    last_layer = keys_list[-1]
                    
                for key in keys_list:
                    tensor_slice = f.get_slice(key)
                    shape = tensor_slice.get_shape()
                    dtype = tensor_slice.get_dtype()
                    
                    numel = 1
                    for s in shape:
                        numel *= s
                    total_params += numel
                    
                    element_size = 2 if "16" in str(dtype) else 4
                    param_bytes += numel * element_size
                    
                    # QKV / Attention Injection find
                    lower_key = key.lower()
                    if any(k in lower_key for k in ["q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v", "attention", "attn", "control_model"]):
                        qkv_matched_count += 1
        else:
            load_mode = "PyTorch Checkpoint Parser"
            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            keys_list = list(state_dict.keys())
            total_layers = len(keys_list)
            
            if total_layers > 0:
                first_layer = keys_list[0]
                last_layer = keys_list[-1]
                
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    total_params += tensor.numel()
                    param_bytes += tensor.numel() * tensor.element_size()
                    
                    lower_key = str(key).lower()
                    if any(k in lower_key for k in ["q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v", "attention", "attn", "control_model"]):
                        qkv_matched_count += 1

        self.show_status("complete")

        # 1. Identification of the ControlNet Basic Learning Model (Base Model)
        base_model_name = metadata.get("ss_base_model_version", None)
        if not base_model_name:
            if 300_000_000 <= total_params <= 450_000_000:
                base_model_name = "Stable Diffusion 1.5"
            elif total_params > 700_000_000:
                base_model_name = "Stable Diffusion XL (SDXL)"
            else:
                base_model_name = "Custom / Unknown Base Model"

        # 2. Recommended: determine whether resolution is present (specify only pixel-based SD series, otherwise skip)
        recommended_resolution = None
        if "1.5" in base_model_name:
            recommended_resolution = "512x512"
        elif "XL" in base_model_name:
            recommended_resolution = "1024x1024"

        # 3. Style logic method (specify QKV / attention processing method)
        if qkv_matched_count > 150:
            style_logic = f"Standard U-Net Cross-Attention & QKV Injection ({qkv_matched_count} layers linked)"
        elif qkv_matched_count > 0:
            style_logic = f"Partial/Custom Attention Injection ({qkv_matched_count} layers)"
        else:
            style_logic = "Non-Attention / Pure Convolutional Structure"

        model_vram_gb = param_bytes / (1024 ** 3)
        estimated_total_vram_gb = model_vram_gb * 1.3

        # Layer-structure formatting
        layer_structure_str = f"{first_layer} ... {last_layer} (find {total_layers}unit)" if total_layers > 0 else "N/A"

        info_lines = [
            f"=== ControlNet Diagnostic Report ===",
            f"Model File: {str(control_net_name)}",
            f"Parser Mode: {str(load_mode)}",
            f"Layer structure (start ~ end / total number of layers): {layer_structure_str}",
            f"ControlNet Basic Learning Model: {base_model_name}",
        ]
        
        if recommended_resolution:
            info_lines.append(f"Recommended resolution: available ({recommended_resolution})")
        else:
            info_lines.append(f"Recommended resolution: none (skip entering resolution)")
            
        info_lines.extend([
            f"Style logic method: {style_logic}",
            f"Total Parameters: {int(total_params):,}",
            f"Estimated Model VRAM: ~{model_vram_gb:.2f} GB (Runtime: ~{estimated_total_vram_gb:.2f} GB)",
            f"Status: Successfully Analyzed."
        ])
        
        final_text = "\n".join(info_lines)

        # output settings
        ui_data = {"text": [final_text], "font_size": [font_size], "widget": [use_widget_2_0]}
        return IO.NodeOutput(final_text,ui=ui_data)

#----------------------------------------

MODULE_NODE_CLASS_MAPPINGS = {
    "EasyencoderLoader": EasyencoderLoader,
    "EasyUnetLoader": EasyUnetLoader,
    "EasyVAELoader": EasyVAELoader,
    "EasyUnetClipLoader": EasyUnetClipLoader,
    "EasyUnetChecker": EasyUnetChecker,
    "EasyVAEChecker": EasyVAEChecker,
    "EasyControlNetChecker": EasyControlNetChecker,

}


MODULE_NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyencoderLoader": "이지 인코더 로더",
    "EasyUnetLoader": "이지 UNet 로더",
    "EasyVAELoader": "이지 VAE 로더",
    "EasyUnetClipLoader": "이지 UNet & CLIP 통합 로더",
    "EasyUnetChecker": "UNet/DiT 모델 분석기",
    "EasyVAEChecker": "VAE 체크 및 인코딩 테스트", 
    "EasyControlNetChecker": "ControlNet 모델 분석기"
}
#----------------------------------------
