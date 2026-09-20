import os
import sys
import re
import math
import random
import hashlib
from typing import List, Dict

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file, safe_open

import folder_paths
import comfy
import comfy.clip_model
import comfy.sd
import comfy.utils
import nodes
import node_helpers
from node_helpers import conditioning_set_values

from comfy_api.latest import IO, UI
from comfy_api.latest._io_public import ComfyTypeIO, comfytype, Custom

#----------------------------------------
# Header Utils
#----------------------------------------

proj_dir = os.path.join(folder_paths.base_path, "models", "proj_embeddings")
os.makedirs(proj_dir, exist_ok=True)
folder_paths.folder_names_and_paths["proj_embeddings"] = ([proj_dir], folder_paths.supported_pt_extensions)

proj_dir = folder_paths.get_folder_paths("proj_embeddings")
embeddings_dir = folder_paths.get_folder_paths("embeddings")

embed_text_dir = os.path.join(folder_paths.base_path, "models", "embeddings", "embed_text")
os.makedirs(embed_text_dir, exist_ok=True)

folder_paths.folder_names_and_paths["embed_text"] = ([embed_text_dir], [".txt"])

embed_text_dir = folder_paths.get_folder_paths("embed_text")

embedload = Custom("EMBEDS")
easytext = Custom("ESLogged")

@comfytype(io_type="EMBEDS")
class Embeds(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="ESLogged")
class Logged(ComfyTypeIO):
    Type = str

def find_tensor(obj):
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

def read_metadata(file_path: str) -> dict:
    """safetensors 파일에서 메타데이터를 읽어 반환"""
    try:
        with safe_open(file_path, framework="torch") as f:
            return f.metadata()
    except Exception as e:
        print(f"메타데이터 읽기 실패: {e}")
        return {}

_counter = 0  

def resolve_filename_txt(prefix: str, save_dir: str, ext: str=".txt") -> str:
    base = prefix.replace("%number", "")
    existing = [f for f in os.listdir(save_dir) if f.startswith(base) and f.endswith(ext)]
    max_num = 0
    pattern = r"_(\d+)" + re.escape(ext) + "$"
    for f in existing:
        m = re.search(pattern, f)
        if m:
            num = int(m.group(1))
            max_num = max(max_num, num)
    next_num = max_num + 1
    if "%number" in prefix:
        filename = prefix.replace("%number", str(next_num)) + ext
    else:
        filename = f"{prefix}_{next_num}{ext}"
    return filename


def save_text_embedding_hidden_prompt(text: str, file_name: str, save_dir=None):
    embed_text_dir = os.path.join(folder_paths.base_path, "models", "embeddings", "embed_text")

    if not os.path.exists(embed_text_dir):
        print("저장될 폴더가 없습니다. 폴더를 생성합니다.")
        os.makedirs(embed_text_dir, exist_ok=True)

    save_dir = save_dir or embed_text_dir
    save_path = os.path.join(save_dir, f"{file_name}.txt")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        size_kb = os.path.getsize(save_path) / 1024.0
        print(f"저장 완료: {save_path}, 키워드='{text}', 용량={size_kb:.2f}KB")
        return save_path
    except Exception as e:
        print(f"저장 실패: {e}")
        return None


#----------------------------------------
#ex_Embed Node Settings
#----------------------------------------
class easyEmbeddingToTextNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        files = []
        for d in embedding_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(".safetensors"):
                        files.append(f)

        return IO.Schema(
            node_id="easyEmbeddingToTextNode",
            display_name="임베딩 → 텍스트 변환",
            category="커스텀임베딩/특수",
            description="저장된 safetensors 임베딩 파일의 메타데이터에서 원본 텍스트를 복원합니다.\n"
                        "임베딩 파일의 메타데이터에 텍스트 데이터가 없다면 출력이 실패할 수 있습니다.",
            inputs=[
                IO.Combo.Input("embedding_file", options=files, default=files[0] if files else "",
                               tooltip="불러올 임베딩 파일 선택"),
                IO.Combo.Input("save_mode", options=["none","save"], default="none",
                               tooltip="복원된 텍스트를 파일로 저장할지 여부")
            ],
            hidden=[IO.Hidden.prompt],
            is_output_node=True,
            outputs=[
                easytext.Output("ESLogged", tooltip="복원된 텍스트 정보")
            ],
        )

    @classmethod
    def execute(cls, embedding_file, save_mode="none") -> IO.NodeOutput:
        result_text = "unknown_token"

        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        file_path = None
        for d in embedding_dirs:
            candidate = os.path.join(d, embedding_file)
            if os.path.exists(candidate):
                file_path = candidate
                break

        if file_path:
            try:
                with safe_open(file_path, framework="torch") as f:
                    metadata = f.metadata()
                    result_text = metadata.get("original_text", embedding_file)
            except Exception as e:
                print(f"[easyEmbeddingToTextNode] Failed to read metadata: {e}")

        if save_mode == "save":
            save_dir = os.path.join(folder_paths.base_path, "models", "embeddings", "embed_text")
            os.makedirs(save_dir, exist_ok=True)

            filename = resolve_filename_txt("embed_text_%number", save_dir)
            save_path = os.path.join(save_dir, filename)

            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(result_text)
                size_kb = os.path.getsize(save_path) / 1024.0
                print(f"Save complete: {save_path}, keyword='{result_text}', volume={size_kb:.2f}KB")
            except Exception as e:
                print(f"Save failure: {e}")

        return IO.NodeOutput(result_text)

#----------------------------------------

class easyTextLoader(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        # Get a list of .txt files from the embed_text folder
        text_dirs = folder_paths.get_folder_paths("embed_text")
        files = []
        for d in text_dirs:
            for f in os.listdir(d):
                if f.endswith(".txt"):
                    files.append(f)

        return IO.Schema(
            node_id="easyTextLoader",
            display_name="텍스트 로더",
            category="커스텀임베딩/특수",
            description="embed_text 폴더에서 저장된 텍스트(.txt)를 불러와 문자열 출력으로 프롬프트에 연결합니다.",
            inputs=[
                IO.Combo.Input("text_file", options=files, default=files[0] if files else "",
                               tooltip="불러올 텍스트 파일을 선택하세요")
            ],
            outputs=[
                IO.String.Output("load_text", tooltip="불러온 텍스트 문자열")
            ],
        )

    @classmethod
    def execute(cls, text_file) -> IO.NodeOutput:
        text_dirs = folder_paths.get_folder_paths("embed_text")
        file_path = None

        for d in text_dirs:
            candidate = os.path.join(d, text_file)
            if os.path.exists(candidate):
                file_path = candidate
                break

        if file_path is None:
            print(f"[easyTextLoader] File {text_file} could not be found.")
            return IO.NodeOutput("")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            size_kb = os.path.getsize(file_path) / 1024.0
            print(f"[easyTextLoader] Load complete: {file_path}, length={len(content)}, volume={size_kb:.2f}KB")

            return IO.NodeOutput(content)
        except Exception as e:
            print(f"[easyTextLoader] Load failed: {e}")
            return IO.NodeOutput("")

#---------------------------------------------------------
class EasyLoraCheckerToText(IO.ComfyNode):

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
            node_id="EasyLoraCheckerToText",
            display_name="LoRA 텍스트 체커",
            category="커스텀임베딩/특수",
            description="LoRA/LyCORIS safetensors 파일의 타입, 차원, 추천 clip skip, 해시, 호환 모델을 확인합니다.\n"
                        "탐색 기준은 kohya 방식을 기준으로 돌립니다. \n"
                        "일부 학습에서는 LoRA를 LyCORIS처럼 병렬 세팅하거나 메타데이터를 증발시키는 경우가 있어 메타데이터만으로는 감지가 힘들 수 있습니다.\n"
                        "주의사항: 이 노드는 Nodes 2.0에 호환되어 있습니다. 사용하시려면 모드를 변경할 필요가 있습니다.",
            inputs=[
                IO.Combo.Input("lora_file", options=files if files else ["no lora"], default=files[0] if files else "no lora", tooltip="확인할 LoRA 파일"),
                IO.Combo.Input("find_keys", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
                               default="20", tooltip="확인할 키워드 수"),
                IO.Combo.Input("save_text", options=["off", "on"],
                               default="off", tooltip="저장 여부"),
                IO.Boolean.Input("log_to_widget", default=False, tooltip="텍스트 위젯 기록 여부")
            ],
            hidden=[IO.Hidden.unique_id, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.String.Output("text_out", tooltip="LoRA 메타데이터 시각화")
            ],
        )


    @classmethod
    def execute(cls, lora_file, find_keys="20", save_text="off", log_to_widget=False) -> IO.NodeOutput:
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

        # output text
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
            f"Prompt Keywords: {prompt_keywords}"
        ]
        
        final_text = "\n".join(info_lines) # Combine into one using line-breaking characters

        if save_text == "on":
            txt_path = path.replace(".safetensors", "_metadata.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(info_lines))
                print(f"Metadata saved to {txt_path}")
            except Exception as e:
                print(f"Failed to save metadata: {e}")

        if log_to_widget: 
            ui_data = {"text": [final_text]} 
            return IO.NodeOutput(final_text,ui=ui_data)
            
        else:
            return IO.NodeOutput(final_text,)

#----------------------------------------
#Embed Node Settings
#----------------------------------------
TXT_NODE_CLASS_MAPPINGS = {
    "easyEmbeddingToTextNode": easyEmbeddingToTextNode,
    "easyTextLoader": easyTextLoader,
    "EasyLoraCheckerToText": EasyLoraCheckerToText
}


TXT_NODE_DISPLAY_NAME_MAPPINGS = {
    "easyEmbeddingToTextNode": "임베딩 → 텍스트 변환",
    "easyTextLoader": "텍스트 로더",
    "EasyLoraCheckerToText": "LoRA 텍스트 체커",
}

#----------------------------------------