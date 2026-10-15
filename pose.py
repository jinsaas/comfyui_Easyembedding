
#----------------------------------------
# pose code
#----------------------------------------


import os
import torch
import numpy as np
import cv2
import re
from PIL import Image
import base64
from io import BytesIO

import comfy.utils
import comfy.sd
from comfy_api.latest import IO, UI

# cnt folder path
CNT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cnt")
os.makedirs(CNT_DIR, exist_ok=True)

class EasyOpenPoseEditor(IO.ComfyNode):

    JOINT_COLORS = {
        "Nose": (0, 0, 255),
        "Neck": (128, 0, 255),
        "REye": (255, 0, 170),
        "REar": (170, 0, 255),
        "LEye": (255, 0, 255),
        "LEar": (85, 0, 255),
        "Chest": (255, 128, 0),
        "MidHip": (128, 128, 128),
        "LShoulder": (0, 255, 85),
        "RShoulder": (0, 170, 255),
        "LElbow": (0, 255, 0),
        "RElbow": (0, 255, 255),
        "LHand": (85, 255, 0),
        "RHand": (0, 255, 170),
        "LHip": (255, 85, 0),
        "RHip": (170, 255, 0),
        "LKnee": (0, 85, 255),
        "RKnee": (255, 255, 0),
        "LFeet": (255, 0, 85),
        "RFeet": (255, 170, 0),
    }
    
    BONE_COLORS = {
        "center": (200, 200, 200),
        "special": (255, 128, 128),
        "left": (0, 128, 255),
        "right": (255, 0, 128),
    }

    # Define the edge connection structure (start point, end point, matching color key)
    BONE_CONNECTIONS = [
        ("Nose", "REye", "right"), 
        ("REye", "REar", "right"),
        ("Nose", "LEye", "left"), 
        ("LEye", "LEar", "left"),
        ("Nose", "Neck", "center"),
        ("Neck", "Chest", "center"),
        ("Chest", "MidHip", "center"),
        ("Chest", "LShoulder", "special"),
        ("Chest", "RShoulder", "special"),
        ("Chest", "LHip", "special"),
        ("Chest", "RHip", "special"),
        ("MidHip", "LHip", "special"),
        ("MidHip", "RHip", "special"),
        ("LShoulder", "LElbow", "left"),
        ("LElbow", "LHand", "left"),
        ("LHip", "LKnee", "left"),
        ("LKnee", "LFeet", "left"),
        ("RShoulder", "RElbow", "right"),
        ("RElbow", "RHand", "right"),
        ("RHip", "RKnee", "right"),
        ("RKnee", "RFeet", "right"),
    ]

    @classmethod
    def make_preview_basebg(cls, arr:Image):
        _, buf = cv2.imencode(".png", arr)
        return base64.b64encode(buf).decode("utf-8")

    @classmethod
    def define_schema(cls):
        default_pose_str = "Nose{256,50}, Neck{256,100}, REye{246,40}, LEye{266,40}, REar{236,45}, LEar{276,45}, Chest{256,150}, MidHip{256,250}, LShoulder{312,110}, RShoulder{200,110}, LElbow{330,170}, RElbow{182,170}, LHand{340,230}, RHand{172,230}, LHip{280,250}, RHip{232,250}, LKnee{285,350}, RKnee{227,350}, LFeet{290,450}, RFeet{222,450}"

        # Organize file lists
        os.makedirs(CNT_DIR, exist_ok=True)
        files = []
        try:
            files = sorted([f for f in os.listdir(CNT_DIR) if f.endswith(".txt")])
        except Exception as e:
            print(f"[EasyOpenPoseEditor] Failed to list cnt dir: {e}")

        pose_file_list = ["none"] + files

        # 최신 openpose_N.txt 로드
        try:
            numbered = []
            for f in files:
                m = re.match(r"openpose_(\d+)\.txt", f)
                if m:
                    numbered.append((int(m.group(1)), f))

            if numbered:
                numbered.sort(key=lambda x: x[0], reverse=True)
                latest_file = os.path.join(CNT_DIR, numbered[0][1])
                with open(latest_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        default_pose_str = content
            else:
                # If not found, create the file openpose_default.txt
                default_path = os.path.join(CNT_DIR, "openpose_default.txt")
                if not os.path.exists(default_path):
                    with open(default_path, "w", encoding="utf-8") as f:
                        f.write(default_pose_str)
                    print(f"[EasyOpenPoseEditor] Created initial pose file: cnt/openpose_default.txt")
                    # Add to list
                    if "openpose_default.txt" not in pose_file_list:
                        pose_file_list.append("openpose_default.txt")
        except Exception as e:
            print(f"[EasyOpenPoseEditor] Failed to handle pose files: {e}")

        return IO.Schema(
            node_id="EasyOpenPoseEditor",
            display_name="오픈포즈 에디터 및 좌표 추출기",
            description=(
                "원본 이미지를 배경으로 삼아 오픈포즈 골격을 드래그/조작하고, "
                "생성된 뼈대 이미지와 키포인트 좌표 문자열을 함께 출력합니다.\n"
                "1. 원본 보고 편집 기능, 원본이미지를 살짝 늘려 포즈준비(결과물은 원본기준 크롭)\n"
                "2. 기존 포즈 로드 기능\n"
                "3. 키포인트 좌표 문자열 출력 기능"
            ),
            inputs=[
                IO.Image.Input("image", tooltip="포즈 편집 기준이 되는 원본 배경 이미지"),
                IO.Int.Input("expand_ratio", min=0, max=512, step=1, default=0, tooltip="캔버스 임시 확장 픽셀 수. 결과물은 크롭됩니다."),
                IO.Image.Input("pose_image", optional=True, tooltip="포즈 편집 기준이 되는 샘플 골격 이미지"),
                IO.Combo.Input("load_pose_file", options=pose_file_list, default="none", tooltip="저장된 포즈 파일 로드. none이면 현재 pose_points 유지"),
                IO.Boolean.Input("load_lock", default=False, tooltip="켜면 선택한 파일을 로드합니다. 로드 후 다시 끄세요."),
                IO.String.Input("pose_points", multiline=False, default="Nose{256,50}, Neck{256,100}, REye{246,40}, LEye{266,40}, REar{236,45}, LEar{276,45}, Chest{256,150}, MidHip{256,250}, LShoulder{312,110}, RShoulder{200,110}, LElbow{330,170}, RElbow{182,170}, LHand{340,230}, RHand{172,230}, LHip{280,250}, RHip{232,250}, LKnee{285,350}, RKnee{227,350}, LFeet{290,450}, RFeet{222,450}",
                                tooltip="모든 관절이 포함된 기본 키포인트 좌표값. 드래그나 수정 시 실시간 갱신됩니다."),
                IO.Boolean.Input("remap_pose", default=False, tooltip="체크 시, 현재 배경 이미지나 샘플 이미지의 크기/비율에 맞춰 pose_points 좌표를 일괄 스케일링(리매치)합니다."),
                IO.Boolean.Input("ref_switch", default=False, tooltip="위젯이 바뀌지 않는 경우 선택/선택해제하시면 적용됩니다."),
                IO.Boolean.Input("guide_preview", default=True, tooltip="좌표 라벨 및 번호 표시 여부"),
                IO.Combo.Input("guide_color", options=["green", "red", "blue", "yellow"], default="green", tooltip="좌표 라벨 컬러.가이드 프리뷰가 켜져있을 때만 작동합니다."),
                IO.Boolean.Input("preview_basebg", default=True, tooltip="프리뷰 이미지를 배경에 넣을지 선택합니다."),
                IO.Int.Input("canvas_size", min=300, max=2048, step=4, default=400, tooltip="에디터 위젯 캔버스의 화면 표시 크기 (픽셀)"),
                IO.Float.Input("point_radius", min=1.0, max=20.0, step=0.5, default=4.0, tooltip="관절 점의 반경 크기"),
                IO.Int.Input("line_thickness", min=1, max=10, step=1, default=3, tooltip="뼈대 선의 두께"),
                IO.Boolean.Input("save_pose", default=False, tooltip="현재 pose_points를 cnt 폴더에 저장"),
            ],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="컨트롤넷용 오픈포즈 뼈대 이미지"),
                IO.String.Output("pose_coords_str", tooltip="다음 노드로 전달할 관절 좌표 문자열"),
            ],
            category="커스텀임베딩/샘플링"
        )

    @classmethod
    def execute(cls, image, expand_ratio=0, pose_image=None, load_pose_file="none", load_lock=False, pose_points="Nose{256,50}, Neck{256,100}, REye{246,40}, LEye{266,40}, REar{236,45}, LEar{276,45}, Chest{256,150}, MidHip{256,250}, LShoulder{312,110}, RShoulder{200,110}, LElbow{330,170}, RElbow{182,170}, LHand{340,230}, RHand{172,230}, LHip{280,250}, RHip{232,250}, LKnee{285,350}, RKnee{227,350}, LFeet{290,450}, RFeet{222,450}", 
                remap_pose=False, ref_switch=False, guide_preview=True, guide_color="green", preview_basebg=True, canvas_size=400, point_radius=4.0, line_thickness=3, save_pose=False) -> IO.NodeOutput:
        
        # 1. Basic Background Image Processing

        if isinstance(image, torch.Tensor):
            arr = image.cpu().numpy()[0]
        else:
            arr = np.array(image)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)

        h, w = arr.shape[:2]

        if expand_ratio > 1:
            margin = min(int(expand_ratio), 512)
            pad_top = margin
            pad_bottom = margin
            pad_left = margin
            pad_right = margin
            
            exp_h = h + pad_top + pad_bottom
            exp_w = w + pad_left + pad_right
            
            pad_y = pad_top
            pad_x = pad_left
            
            canvas = np.zeros((exp_h, exp_w, 3), dtype=np.uint8)
        else:
            pad_y, pad_x = 0, 0
            exp_h, exp_w = h, w
            canvas = np.zeros((exp_h, exp_w, 3), dtype=np.uint8)

        if preview_basebg:
            if expand_ratio > 0:
                exp_bg = np.zeros((exp_h, exp_w, 3), dtype=np.uint8)
                exp_bg[pad_y:pad_y + h, pad_x:pad_x + w] = arr
                final_bg = exp_bg
            else:
                final_bg = arr
            b64_img = cls.make_preview_basebg(final_bg)
        else:
            b64_img = ""

        # 2. Preparing pose string parsing and coordinate scaling (with load lock applied)
        parsed_joints = {}
        DEFAULT_POSE = "Nose{256,50}, Neck{256,100}, REye{246,40}, LEye{266,40}, REar{236,45}, LEar{276,45}, Chest{256,150}, MidHip{256,250}, LShoulder{312,110}, RShoulder{200,110}, LElbow{330,170}, RElbow{182,170}, LHand{340,230}, RHand{172,230}, LHip{280,250}, RHip{232,250}, LKnee{285,350}, RKnee{227,350}, LFeet{290,450}, RFeet{222,450}"

        loaded_pose_str = None
        if load_lock and load_pose_file != "none":
            load_path = os.path.join(CNT_DIR, load_pose_file)
            if os.path.exists(load_path):
                with open(load_path, "r", encoding="utf-8") as f:
                    file_content = f.read().strip()
                    if file_content:
                        loaded_pose_str = file_content
                        print(f"[EasyOpenPoseEditor] SUCCESS: Loaded pose from cnt/{load_pose_file}")

        # 2. [If an open pose image (pose_image) is entered]
        if pose_image is not None:
            pose_arr = pose_image.cpu().numpy()[0] if isinstance(pose_image, torch.Tensor) else np.array(pose_image)
            if pose_arr.max() <= 1.0:
                pose_arr = (pose_arr * 255).astype(np.uint8)
            pose_h, pose_w = pose_arr.shape[:2]

            for joint_name, bgr_color in cls.JOINT_COLORS.items():
                mask = np.all(pose_arr == bgr_color, axis=-1)
                ys, xs = np.where(mask)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    if (pose_h, pose_w) != (h, w):
                        new_x = int(cx * (w / pose_w))
                        new_y = int(cy * (h / pose_h))
                        parsed_joints[joint_name] = (new_x, new_y)
                    else:
                        parsed_joints[joint_name] = (cx, cy)

        # 3. [Text/Drag Coordinate String Processing] (When file loading is successful or existing pose points are present)
        elif loaded_pose_str or (pose_points and pose_points.strip() != ""):
            target_str = loaded_pose_str if loaded_pose_str else pose_points
            active_pose_str = re.sub(r'\s+', ' ', target_str).strip()
            
            matches = re.findall(r"([A-Za-z0-9_]+)\s*\{\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*\}", active_pose_str)
            for name, x_str, y_str in matches:
                key = name if name in cls.JOINT_COLORS else name.capitalize()
                parsed_joints[key] = (float(x_str), float(y_str))

            if remap_pose:
                base_w, base_h = 512, 512
                for name, x_str, y_str in matches:
                    old_x, old_y = float(x_str), float(y_str)
                    new_x = int(old_x * (w / base_w))
                    new_y = int(old_y * (h / base_h))
                    key = name if name in cls.JOINT_COLORS else name.capitalize()
                    parsed_joints[key] = (new_x, new_y)

        # 4. [Exception handling: when no coordinates are entered] default generation
        else:
            print(f"[EasyOpenPoseEditor] Warning: 입력된 포즈 데이터가 없어 기본 포즈를 생성합니다.")
            matches = re.findall(r"([A-Za-z0-9_]+)\s*\{\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*\}", DEFAULT_POSE)
            for name, x_str, y_str in matches:
                key = name if name in cls.JOINT_COLORS else name.capitalize()
                parsed_joints[key] = (float(x_str), float(y_str))

        # Neck auto calculation
        if "Neck" not in parsed_joints and "Nose" in parsed_joints and "Chest" in parsed_joints:
            nx = (parsed_joints["Nose"][0] + parsed_joints["Chest"][0]) / 2
            ny = (parsed_joints["Nose"][1] + parsed_joints["Chest"][1]) / 2
            parsed_joints["Neck"] = (nx, ny)

        # MidHip auto calculation
        if "MidHip" not in parsed_joints and "LHip" in parsed_joints and "RHip" in parsed_joints:
            mx = (parsed_joints["LHip"][0] + parsed_joints["RHip"][0]) / 2
            my = (parsed_joints["LHip"][1] + parsed_joints["RHip"][1]) / 2
            parsed_joints["MidHip"] = (mx, my)

        # Reshape the final coordinate string
        active_pose_str = ",".join([f"{name}{{{int(x)},{int(y)}}}" for name,(x,y) in parsed_joints.items()])

        output_arr = canvas.copy()

        # 3. Bones (Edges) Drawing Loop
        for p1, p2, color_key in cls.BONE_CONNECTIONS:
            if p1 in parsed_joints and p2 in parsed_joints:
                pt1 = (int(parsed_joints[p1][0] + pad_x), int(parsed_joints[p1][1] + pad_y))
                pt2 = (int(parsed_joints[p2][0] + pad_x), int(parsed_joints[p2][1] + pad_y))
                bgr_color = cls.BONE_COLORS.get(color_key, (0, 255, 0))
                cv2.line(output_arr, pt1, pt2, bgr_color, line_thickness, cv2.LINE_AA)

        # 4. Joint vertex drawing loop
        for name, (px, py) in parsed_joints.items():
            bgr_x, bgr_y = int(px + pad_x), int(py + pad_y)
            bgr_color = cls.JOINT_COLORS.get(name, (0, 255, 0))
            cv2.circle(output_arr, (bgr_x, bgr_y), int(point_radius), bgr_color, -1)
            if guide_preview:
                cv2.putText(output_arr, name, (bgr_x + int(point_radius), bgr_y - int(point_radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        if expand_ratio > 0:
            y1 = max(0, pad_y)
            x1 = max(0, pad_x)
            y2 = min(exp_h, pad_y + h)
            x2 = min(exp_w, pad_x + w)
            
            cropped = output_arr[y1:y2, x1:x2]
            
            if cropped.shape[0] != h or cropped.shape[1] != w:
                final_output_arr = cv2.resize(cropped[pad_y:pad_y+h, pad_x:pad_x+w], (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                final_output_arr = cropped
        else:
            final_output_arr = output_arr

        if save_pose:
            try:
                existing = [f for f in os.listdir(CNT_DIR)
                           if re.match(r"openpose_(\d+)\.txt", f)]
                nums = [int(re.match(r"openpose_(\d+)\.txt", f).group(1))
                       for f in existing]
                next_num = max(nums) + 1 if nums else 1
                fname = f"openpose_{next_num}.txt"
                save_path = os.path.join(CNT_DIR, fname)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(active_pose_str)
                print(f"[EasyOpenPoseEditor] Saved: cnt/{fname}")
            except Exception as e:
                print(f"[EasyOpenPoseEditor] Failed to save pose: {e}")

        # 5. Tensor Output after conversion
        final_tensor = torch.from_numpy(final_output_arr.astype(np.float32) / 255.0).unsqueeze(0)
        ui_data = {"b64_img": [b64_img]}
        if load_lock and load_pose_file != "none":
            # Match the coordinates read from the file to the pose_points
            ui_data["pose_points_updated"] = [active_pose_str]
        else:
            # Apply input coordinates and provide automatically calculated coordinates each time a coordinate changes
            ui_data["pose_points_updated"] = [active_pose_str]

        return IO.NodeOutput(final_tensor, active_pose_str, ui=ui_data)

#----------------------------------------

POSE_NODE_CLASS_MAPPINGS = {
    "EasyOpenPoseEditor": EasyOpenPoseEditor,
}

POSE_NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyOpenPoseEditor": "오픈포즈 에디터 및 좌표 추출기"
}
#----------------------------------------
