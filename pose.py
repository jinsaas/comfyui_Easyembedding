
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
import math

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
                IO.Int.Input("canvas_size", min=300, max=800, step=4, default=400, tooltip="에디터 위젯 캔버스의 화면 표시 크기 (픽셀)"),
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

# ─────────────────────────────────────────
# Depth converter logic & Constants
# ─────────────────────────────────────────

Z_PRESET = {"Front": 0.85, "sub_Front": 0.70, "Center": 0.50, "sub_Back": 0.30, "Back": 0.15}

BONE_CONNECTIONS = [
    ("Nose",      "Neck",       ),
    ("Neck",      "Chest",      ),
    ("Chest",     "MidHip",     ),
    ("Chest",     "LShoulder",  ),
    ("Chest",     "RShoulder",  ),
    ("Chest",     "LHip",       ),
    ("Chest",     "RHip",       ),
    ("MidHip",    "LHip",       ),
    ("MidHip",    "RHip",       ),
    ("LShoulder", "LElbow",     ),
    ("LElbow",    "LHand",      ),
    ("RShoulder", "RElbow",     ),
    ("RElbow",    "RHand",      ),
    ("LHip",      "LKnee",      ),
    ("LKnee",     "LFeet",      ),
    ("RHip",      "RKnee",      ),
    ("RKnee",     "RFeet",      ),
]

BONE_CODE = [
    {"LHumerus", "Lulna","RHumerus", "Rulna","LFemur", "LTibia", "RFemur", "RTibia"},
]

BODY_CENTER = {"Nose", "Neck", "Chest", "MidHip", "REye", "LEye", "REar", "LEar"}
SHOULDER    = {"LShoulder", "RShoulder"}
HIP         = {"LHip", "RHip"}

LEFT_ARM    = {"LHumerus", "Lulna", "LElbow", "LHand"}
RIGHT_ARM   = {"RHumerus", "Rulna","RElbow", "RHand"}
LEFT_LEG    = {"LFemur", "LTibia", "LKnee", "LFeet"}
RIGHT_LEG   = {"RFemur", "RTibia", "RKnee", "RFeet"}

LIMB_SEGMENTS = [
        ("LShoulder", "LElbow", "L_Humerus_seg"),
        ("LElbow",    "LHand",  "L_Ulna_seg"),
        ("RShoulder", "RElbow", "R_Humerus_seg"),
        ("RElbow",    "RHand",  "R_Ulna_seg"),
        
        ("LHip",      "LKnee",  "L_Femur_seg"),
        ("LKnee",     "LFeet",  "L_Tibia_seg"),
        ("RHip",      "RKnee",  "R_Femur_seg"),
        ("RKnee",     "RFeet",  "R_Tibia_seg"),
        ]


def parse_pose_str(pose_str: str) -> dict:
    joints = {}
    if not pose_str or not isinstance(pose_str, str):
        return joints
    matches = re.findall(r"([A-Za-z0-9_]+)\s*\{\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*\}", pose_str)
    for name, x_str, y_str in matches:
        joints[name] = (float(x_str), float(y_str))
    return joints

def build_z_map(l_arm1, l_arm2, r_arm1, r_arm2, l_leg1, l_leg2, r_leg1, r_leg2) -> dict:
    z = {}
    for name in BODY_CENTER:
        z[name] = 0.5
    for name in SHOULDER:
        z[name] = 0.5
    for name in HIP:
        z[name] = 0.5
        
    # Directly map UI presets by segment name of each bone segment
    # Upper arm (Humerus section: shoulder ~ elbow)
    z["L_Humerus_seg"] = Z_PRESET.get(l_arm1, 0.5)
    z["R_Humerus_seg"] = Z_PRESET.get(r_arm1, 0.5)
    
    # Haman (Ulna section: elbow ~ hand)
    z["L_Ulna_seg"] = Z_PRESET.get(l_arm2, 0.5)
    z["R_Ulna_seg"] = Z_PRESET.get(r_arm2, 0.5)
    
    # Thigh (Femur segment: pelvis ~ knee)
    z["L_Femur_seg"] = Z_PRESET.get(l_leg1, 0.5)
    z["R_Femur_seg"] = Z_PRESET.get(r_leg1, 0.5)
    
    # Calf (Tibia segment: knee ~ foot)
    z["L_Tibia_seg"] = Z_PRESET.get(l_leg2, 0.5)
    z["R_Tibia_seg"] = Z_PRESET.get(r_leg2, 0.5)
    
    return z


def create_centered_cuboid_mask(h, w, center, spine_dir, width, length, factor=1):
    mask = np.zeros((h, w), dtype=np.uint8)
    if spine_dir is None or np.linalg.norm(spine_dir) == 0:
        return mask
    
    u = spine_dir / np.linalg.norm(spine_dir)
    v = np.array([-u[1], u[0]], dtype=np.float32)
    
    hw = width / 2.0 + int(factor)
    hl = length / 2.0
    
    p1 = center + u * hl + v * hw
    p2 = center + u * hl - v * hw
    p3 = center - u * hl - v * hw
    p4 = center - u * hl + v * hw
    
    pts = np.array([p1, p2, p3, p4], dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)  # x
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)  # y
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask

def draw_true_3d_cuboid(canvas, h, w, center, spine_dir, axis_vec, length, half_width, z_center, depth_strength):

    if spine_dir is None or np.linalg.norm(spine_dir) == 0:
        return
        
    u = spine_dir / np.linalg.norm(spine_dir) # Vertical direction
    v = axis_vec / (np.linalg.norm(axis_vec) + 1e-6) # Horizontal direction
    
    hl = length / 2.0
    hw = half_width
    
    # 1. Calculating the four corner vertices of a rectangular prism (based on the center)
    p_tl = center - u * hl - v * hw  # Top-Left
    p_tr = center - u * hl + v * hw  # Top-Right
    p_br = center + u * hl + v * hw  # Bottom-Right
    p_bl = center + u * hl - v * hw  # Bottom-Left
    
    # 2. Create a front face mask
    front_mask = np.zeros((h, w), dtype=np.uint8)
    front_pts = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.int32)
    front_pts[:, 0] = np.clip(front_pts[:, 0], 0, w - 1)
    front_pts[:, 1] = np.clip(front_pts[:, 1], 0, h - 1)
    cv2.fillPoly(front_mask, [front_pts.reshape(-1, 1, 2)], 255)
    
    # Apply Z-value to the front face (nearest)
    z_front = np.clip(z_center * depth_strength, 0.0, 1.0)
    y_idx, x_idx = np.where(front_mask > 0)
    if len(y_idx) > 0:
        update_mask = z_front >= canvas[y_idx, x_idx]
        canvas[y_idx[update_mask], x_idx[update_mask]] = z_front
        
    # 3. Side Faces - Adds dimensional edge thickness
    # Create an extruded border with a Z-value slightly lower than the front face (at the back)
    z_side = np.clip((z_center - 0.1) * depth_strength, 0.0, 1.0)
    
    # Draw thick outlines to express the rectangle's "side profile (thickness)"
    thick_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(thick_mask, [front_pts.reshape(-1, 1, 2)], isClosed=True, color=255, thickness=int(hw * 0.4))
    
    y_idx_s, x_idx_s = np.where(thick_mask > 0)
    if len(y_idx_s) > 0:
        update_mask_s = z_side >= canvas[y_idx_s, x_idx_s]
        canvas[y_idx_s[update_mask_s], x_idx_s[update_mask_s]] = z_side

def estimate_axis_dir(center, left_pt, right_pt):

    if left_pt is not None and right_pt is not None:
        vec = left_pt - right_pt
        # [Key point] If you use the diagonal distance as is, it shortens when the line is tilted,
        # Calibrate to fully reflect the horizontal width (X-axis scale) and the projected size
        vec_norm = np.linalg.norm(vec)
        axis_unit = vec / (vec_norm + 1e-6)
        
        # Consider the actual horizontal distance between the two shoulders (the absolute difference in X-values) together,
        # Corrects losses due to diagonal tilt.
        horizontal_span = abs(left_pt[0] - right_pt[0])
        half_w = max(vec_norm * 0.5, horizontal_span * 0.5)
        return axis_unit, half_w

    elif left_pt is not None:
        # Case 2-1: Only the left side is present → Estimate center-based symmetry
        vec = left_pt - center
        half_w = np.linalg.norm(vec)
        return vec / (np.linalg.norm(vec) + 1e-6), half_w

    elif right_pt is not None:
        # Case 2-2: Only the right one is there
        vec = center - right_pt
        half_w = np.linalg.norm(vec)
        return vec / (np.linalg.norm(vec) + 1e-6), half_w

    else:
        # Case 3: Neither → Assume it is level
        return np.array([1.0, 0.0], dtype=np.float32), shoulder_dist * 0.5

def draw_optimized_depth_map(joints: dict, z_map: dict, h: int, w: int, depth_strength: float = 1.0, neck_bias: int = 15, spine_bias: float = 0.5, bend_strength=1, thickness_scale=1.0, b_kernel=0) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.float32)
    base_radius = max(10, int(min(h, w) * 0.035))
    
    # Shoulder spacing proportional scale calculation
    shoulder_dist = base_radius * 4.0
    if "LShoulder" in joints and "RShoulder" in joints:
        dx = joints["LShoulder"][0] - joints["RShoulder"][0]
        dy = joints["LShoulder"][1] - joints["RShoulder"][1]
        shoulder_dist = max(10.0, np.sqrt(dx**2 + dy**2))

    render_items = []

    # Essential key point verification
    if not all(k in joints for k in ["Neck", "MidHip"]):
        return canvas

    neck_pt = np.array(joints["Neck"], dtype=np.float32)
    mid_hip = np.array(joints["MidHip"], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 1. Calculate the Spine-Middle (center) based on the neck and midhip
    # Spine center tilt/position can be corrected via spine_bias (default 0.5 = center of gravity)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    s_bias_delta  = spine_bias - 0.5
    bend_factor = s_bias_delta * bend_strength  
    spine_middle = (neck_pt + mid_hip)/2

    spine_vec = mid_hip - neck_pt
    spine_vec = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)
    side_vec = np.array([-spine_vec[1], spine_vec[0]])
    spine_middle = spine_middle + side_vec * bend_factor
    # 2. Spine Top (Neck ~ Spine-Middle 사이) 및 Spine Bottom (Spine-Middle ~ MidHip 사이) 정의
    spine_top = (neck_pt * 0.5 + spine_middle * 0.5)
    spine_bottom = (spine_middle * 0.5 + mid_hip * 0.5)

    min_thickness = max(8, int(min(h, w) * 0.02))
    adj = max(min_thickness, int(round(shoulder_dist * 0.25)))
    
    # 흉곽/상체 블록의 기준이 되는 'Chest' (복장뼈 자루 위치 반영)
    chest_pt = np.array(joints["Chest"], dtype=np.float32) if "Chest" in joints else spine_top
    chest_offset_len = np.linalg.norm(chest_pt - spine_top)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 3. Upper thorax, rectangular shape (center of the spine)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    top_dir = spine_top - neck_pt
    top_length = max(30, int(np.linalg.norm(top_dir) * 0.9 + chest_offset_len))
    
    axis_vec, half_w = estimate_axis_dir(chest_pt, np.array(joints.get("LShoulder")) if "LShoulder" in joints else None,
                                         np.array(joints.get("RShoulder")) if "RShoulder" in joints else None)

    draw_true_3d_cuboid(canvas, h, w, center=spine_top, spine_dir=top_dir, axis_vec=axis_vec, length=top_length, 
                        half_width=int(half_w * 1.12), z_center=(z_map.get("Chest", 0.5) + z_map.get("Neck", 0.5)) / 2.0, 
                        depth_strength=depth_strength)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 4. Lower abdomen/pelvic rectangle (based on the spine bottom and mid-hip center)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    bottom_dir = mid_hip - spine_bottom
    bottom_length = max(30, int(np.linalg.norm(bottom_dir)))
    
    hip_width = int(shoulder_dist * 0.8)
    if "LHip" in joints and "RHip" in joints:
        l_hip = np.array(joints["LHip"], dtype=np.float32)
        r_hip = np.array(joints["RHip"], dtype=np.float32)
        hip_width = max(hip_width, int(np.linalg.norm(l_hip - r_hip) * 1.15))

    axis_vec, half_hw = estimate_axis_dir(mid_hip, np.array(joints.get("LHip")) if "LHip" in joints else None,
                                         np.array(joints.get("RHip")) if "RHip" in joints else None)

    draw_true_3d_cuboid(canvas, h, w, center=mid_hip, spine_dir=bottom_dir, axis_vec=axis_vec, length=bottom_length, 
                        half_width=half_hw, z_center=z_map.get("MidHip", 0.5), depth_strength=depth_strength)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # 5. Neck cylinder and head block
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    neck_vec  = neck_pt - spine_top
    neck_len  = np.linalg.norm(neck_vec)
    neck_unit = neck_vec / (neck_len + 1e-6)

    n_bias_delta = (neck_bias - 15.0) / 15.0 # default = 15
    bias_ratio = neck_bias / 30.0
    
    # 1. Set the rotation angle for the neck joint (neck_pt) pivot (bend_strength is the maximum angle. ex: 0–30 degrees)
    # if neck_bias 15< turn Left, neck_bias 15> turn Right
    angle_deg = n_bias_delta * bend_strength   
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    head_mask = np.zeros((h, w), dtype=np.uint8)
    neck_mask = np.zeros((h, w), dtype=np.uint8)

    if "LEye" in joints and "REye" in joints and "Nose" in joints:
        # When the image is front-facing
        leye       = np.array(joints["LEye"], dtype=np.float32)
        reye       = np.array(joints["REye"], dtype=np.float32)
        nose       = np.array(joints["Nose"], dtype=np.float32)
        eye_center = (leye + reye) * 0.5
        base_x     = nose[0] * 0.4 + eye_center[0] * 0.6
        base_y     = nose[1] * 0.4 + eye_center[1] * 0.6
        
        # Base head-centered vector (based on the neck)
        head_base = np.array([base_x, base_y], dtype=np.float32)
        neck_to_head = head_base - neck_pt
        
        # Apply a 2D rotation matrix (neck joint center rotation + lift it sufficiently toward the crown of the head)
        rx = cos_a * neck_to_head[0] - sin_a * neck_to_head[1]
        ry = sin_a * neck_to_head[0] + cos_a * neck_to_head[1]
        
        new_head = neck_pt + np.array([rx, ry], dtype=np.float32)
        hx = int(new_head[0])
        # Secure a fixation offset upward (toward the crown) to avoid attachment to the vocal cords
        hy = int(new_head[1] - shoulder_dist * 0.15)

    elif "Nose" in joints and (("LEye" in joints) ^ ("REye" in joints)):
        # If the image is on the side
        nose = np.array(joints["Nose"], dtype=np.float32)
        if "REye" in joints:
            reye       = np.array(joints["REye"], dtype=np.float32)
            rear       = np.array(joints["REar"], dtype=np.float32) if "REar" in joints else reye
            eye_center = (rear + reye) * 0.5
        else:
            leye       = np.array(joints["LEye"], dtype=np.float32)
            lear       = np.array(joints["LEar"], dtype=np.float32) if "LEar" in joints else leye
            eye_center = (leye + lear) * 0.5
            
        base_x = nose[0] * 0.4 + eye_center[0] * 0.6
        base_y = nose[1] * 0.4 + eye_center[1] * 0.6
        
        head_base = np.array([base_x, base_y], dtype=np.float32)
        neck_to_head = head_base - neck_pt
        
        rx = cos_a * neck_to_head[0] - sin_a * neck_to_head[1]
        ry = sin_a * neck_to_head[0] + cos_a * neck_to_head[1]
        
        new_head = neck_pt + np.array([rx, ry], dtype=np.float32)
        hx = int(new_head[0])
        hy = int(new_head[1] - shoulder_dist * 0.15)

    else:
        # No eyes/nose — estimated above the neck
        eye_center = neck_pt + neck_unit * neck_len * 1.2
        hx = int(eye_center[0])
        hy = int(eye_center[1] - shoulder_dist * 0.15)

    # The neck cylinder is connected from the upper center of the spine to the neck point
    pt1 = tuple(spine_top.astype(int))
    pt2 = tuple(map(int, neck_pt))
    neck_thick = max(14, int(shoulder_dist * thickness_scale * 0.3))
    cv2.line(neck_mask, pt1, pt2, 255, neck_thick)
    z_neck = z_map.get("Neck", 0.5)
    render_items.append({"type": "box", "z": z_neck, "mask": neck_mask})

    # The neck B cylinder is connected from the eye_center to the neck point
    neck_mask_b = np.zeros((h, w), dtype=np.uint8)
    neck_thick1 = max(14, int(shoulder_dist * thickness_scale * 0.18))
    cv2.line(neck_mask_b, tuple(eye_center.astype(int)), tuple(neck_pt.astype(int)), 255, neck_thick1)
    z_neck_b = z_map.get("Neck", 0.5)
    render_items.append({"type": "box", "z": z_neck_b, "mask": neck_mask_b})

    # Upper spine - center of the spine
    spine_mask_a = np.zeros((h, w), dtype=np.uint8)
    spine_thick1 = max(14, int(shoulder_dist * thickness_scale * 0.7))
    cv2.line(spine_mask_a, tuple(spine_top.astype(int)), tuple(spine_middle.astype(int)), 255, spine_thick1)
    z_spine1 = (z_map.get("Neck", 0.5) + z_map.get("MidHip", 0.5)) / 2.0
    render_items.append({"type": "box", "z": z_spine1, "mask": spine_mask_a})

    # Spine center point - lower spine
    spine_mask_b = np.zeros((h, w), dtype=np.uint8)
    spine_thick2 = max(14, int(shoulder_dist * thickness_scale * 0.7))
    cv2.line(spine_mask_b, tuple(spine_middle.astype(int)), tuple(spine_bottom.astype(int)), 255, spine_thick2)
    z_spine2 = (z_map.get("Neck", 0.5) + z_map.get("MidHip", 0.5)) / 2.0
    render_items.append({"type": "box", "z": z_spine2, "mask": spine_mask_b})

    # Lower spine - pelvic center
    spine_mask_c = np.zeros((h, w), dtype=np.uint8)
    spine_thick3 = max(14, int(shoulder_dist * thickness_scale * 0.65))
    cv2.line(spine_mask_c, tuple(spine_bottom.astype(int)), tuple(mid_hip.astype(int)), 255, spine_thick3)
    z_spine3 = (z_map.get("Neck", 0.5) + z_map.get("MidHip", 0.5)) / 2.0
    render_items.append({"type": "box", "z": z_spine3, "mask": spine_mask_c})

    # The neck coordinate in open pose essentially holds the vocal cords, so it is relatively far from the head.
    # if you place it too close, it will become misaligned.
    head_radius = int(shoulder_dist * 0.5)
    cv2.circle(head_mask, (hx, hy), head_radius, 255, -1)
    box_offset = int(head_radius * 0.7)
    cv2.rectangle(head_mask, (hx - box_offset, hy - box_offset), (hx + box_offset, hy + box_offset), 255, -1)
    
    head_z = z_map.get("Nose", z_map.get("Neck", 0.5))
    render_items.append({"type": "box", "z": head_z, "mask": head_mask})
    
    # Lower part of the head → Neck connection cylinder (start point adjustment with neck bias)
    t_blend = 0.5 - (n_bias_delta * 0.5)
    head_bottom = np.array([hx, hy], dtype=np.float32) * t_blend + neck_pt * (1.0 - t_blend)
    neck_conn_mask  = np.zeros((h, w), dtype=np.uint8)
    neck_conn_thick = max(10, int(shoulder_dist * thickness_scale * 0.22))
    cv2.line(neck_conn_mask, (int(head_bottom[0]), int(head_bottom[1])), tuple(neck_pt.astype(int)), 255, neck_conn_thick)
             
    render_items.append({"type": "gradient_line","z":  (head_z + z_neck) / 2.0, "mask": neck_conn_mask, "p1": (int(head_bottom[0]),
                         int(head_bottom[1])), "p2": tuple(neck_pt.astype(int)), "z1": head_z, "z2": z_neck})


    # ─────────────────────────────────────────────────────────
    # 6. Skeletal frame (thicker capsule and joints)
    # ─────────────────────────────────────────────────────────
    base_limb_thick = shoulder_dist * thickness_scale

    for p1_name, p2_name, seg_key in LIMB_SEGMENTS:
        if p1_name not in joints or p2_name not in joints:
            continue

        pt1 = tuple(map(int, joints[p1_name]))
        pt2 = tuple(map(int, joints[p2_name]))
        seg_z = z_map.get(seg_key, 0.5)

        # If 'femur' or 'tibia' is in the seg_key, it is classified as the leg
        is_leg = "Hip" in seg_key.lower() or "femur"  in seg_key.lower() or "Knee"  in seg_key.lower() or "tibia"  in seg_key.lower() or "Feet" in seg_key.lower()
        
        # Make the legs thicker than the arms
        mult_factor = 0.55 if is_leg else 0.42
        limb_thickness = max(14, int(base_limb_thick * mult_factor))

        limb_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(limb_mask, pt1, pt2, 255, limb_thickness)
        cv2.circle(limb_mask, pt1, int(limb_thickness * 0.55), 255, -1)
        cv2.circle(limb_mask, pt2, int(limb_thickness * 0.55), 255, -1)

        render_items.append({
            "type": "gradient_line",
            "z": seg_z,
            "mask": limb_mask,
            "p1": pt1,
            "p2": pt2,
            "z1": seg_z,
            "z2": seg_z,
        })

    # ─────────────────────────────────────────────────────────
    # 7. Z-Buffer Lining and Linear Gradient Application (Painter's Algorithm)
    # ─────────────────────────────────────────────────────────
    render_items.sort(key=lambda x: x["z"])

    for item in render_items:
        mask = item["mask"] > 0
        z_val = np.clip(item["z"] * depth_strength, 0.0, 1.0)
        
        if item["type"] == "box":
            update_cond = mask & (z_val >= canvas)
            canvas[update_cond] = z_val
            
        elif item["type"] == "gradient_line":
            p1, p2 = item["p1"], item["p2"]
            z1 = np.clip(item["z1"] * depth_strength, 0.0, 1.0)
            z2 = np.clip(item["z2"] * depth_strength, 0.0, 1.0)
            
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                x1, y1 = p1
                x2, y2 = p2
                denom = max(1.0, (x2 - x1)**2 + (y2 - y1)**2)
                
                t = np.clip(((x_indices - x1)*(x2 - x1) + (y_indices - y1)*(y2 - y1)) / denom, 0.0, 1.0)
                pixel_z = z1 + (z2 - z1) * t
                
                current_vals = canvas[y_indices, x_indices]
                update_mask = pixel_z >= current_vals
                canvas[y_indices[update_mask], x_indices[update_mask]] = pixel_z[update_mask]

    # ─────────────────────────────────────────────────────────
    # 8. Apply a soft edge filter (remove jagged edges)
    # ─────────────────────────────────────────────────────────
    if b_kernel == 0:
        np.clip(canvas, 0.0, 1.0)
    else:
        blur_kernel = int(b_kernel) + 1
        canvas = cv2.GaussianBlur(canvas, (blur_kernel, blur_kernel), 0)
        np.clip(canvas, 0.0, 1.0)

    return canvas
    
class EasyPoseDepthConverter(IO.ComfyNode):

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
    def define_schema(cls):
        os.makedirs(CNT_DIR, exist_ok=True)
        files = []
        try:
            files = sorted([f for f in os.listdir(CNT_DIR) if f.endswith(".txt")])
        except Exception as e:
            print(f"[EasyPoseDepthConverter] cnt Folder scan failed: {e}")

        file_options = files if files else ["No pose file"]

        return IO.Schema(
            node_id="EasyPoseDepthConverter",
            display_name="포즈 → 뎁스준비이미지 변환기",
            description="cnt 폴더의 포즈 텍스트 파일을 직접 읽어 뎁스 가이드 준비이미지를 생성합니다.\n"
                        "그대로 넣지 마시고 Depth Anything v2 Preprocessor에 연결하세요.\n"
                        "(Depth Anything v1은 해석 실패할 가능성이 있음)",
            inputs=[
                IO.Combo.Input("pose_file", options=file_options, default=file_options[0], tooltip="cnt 폴더에 저장된 포즈 텍스트 파일 선택"),
                IO.Image.Input("image", tooltip="원본 이미지 (크기 참조용)"),
                IO.Combo.Input("left_arm1", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="왼팔 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("left_arm2", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="왼팔 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("right_arm1", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="오른팔 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("right_arm2", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="오른팔 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("left_leg1", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="왼다리 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("left_leg2", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="왼다리 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("right_leg1", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="오른다리 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Combo.Input("right_leg2", options=["Front", "sub_Front", "Center", "sub_Back", "Back"], default="Center", tooltip="오른다리 깊이 방향 (Center: 중간, Front: 앞, Back: 뒤)"),
                IO.Float.Input("neck_bias", min=0, max=30, step=1, default=15, tooltip="목 기울이기 (15=곧은 목)"),
                IO.Float.Input("spine_bias", min=0.1, max=1.0, step=0.1, default=0.5, tooltip="허리 기울이기 (0.5=곧은 허리)"),
                IO.Int.Input("bend_strength", min=1, max=50, step=1, default=1, tooltip="허리/목 이동 기준치"),
                IO.Float.Input("depth_strength", min=0.1, max=2.0, step=0.05, default=1.0, tooltip="깊이 대비 강도 (높을수록 앞뒤 차이가 강조됨)"),
                IO.Float.Input("thickness_scale", min=0.1, max=2.0, step=0.05, default=1.0, tooltip="골격 강화. 수치가 높을수록 골격이 넓어집니다."),
                IO.Combo.Input("edge_blur", options=["default", "1", "2", "3", "4", "5", "6", "7"], default="default", tooltip="블러 강화. 높을수록 경계면이 흐려집니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("show_basebg", default=False, tooltip="프리뷰의 뒤에 원본을 표시할지 여부. show_preview가 켜져있어야 합니다."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("depth_map",
                    tooltip="생성된 깊이맵 (Depth 가이드)"),
            ],
            category="커스텀임베딩/샘플링"
        )

    @classmethod
    def execute(cls, pose_file="No pose file", image=None, left_arm1="Center", left_arm2="Center", right_arm1="Center", right_arm2="Center", left_leg1="Center", left_leg2="Center", 
                right_leg1="Center", right_leg2="Center", neck_bias=15, spine_bias=0.5, bend_strength=1, depth_strength=1.0, thickness_scale=1.0,
                edge_blur="default", show_preview=False, show_basebg=False) -> IO.NodeOutput:
        if edge_blur == "default":
            blur_kernel_size = 0
        else:
            level = int(edge_blur)
            blur_kernel_size = level * 2
        
        h, w = 512, 512
        orig_arr = None
        if image is not None:
            tensor_img = cls.ensure_image_tensor(image)
            orig_arr = (tensor_img[0].cpu().numpy() * 255).astype(np.uint8)
            if orig_arr.ndim >= 2:
                h, w = orig_arr.shape[:2]

        cnt_dir = os.path.join(os.path.dirname(__file__), "cnt")
        target_str = ""
        if pose_file and pose_file != "No pose file":
            file_path = os.path.join(cnt_dir, pose_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        target_str = f.read().strip()
                except Exception as e:
                    print(f"[EasyPoseDepthConverter] File reading error ({pose_file}): {e}")

        joints = parse_pose_str(target_str)
        if not joints:
            empty = torch.zeros(1, h, w, 3, dtype=torch.float32)
            return (empty,)

        z_map = build_z_map(left_arm1, left_arm2, right_arm1, right_arm2, left_leg1, left_leg2, right_leg1, right_leg2)
        depth = draw_optimized_depth_map(joints, z_map, h, w, depth_strength, neck_bias, spine_bias, bend_strength=bend_strength, thickness_scale=thickness_scale, b_kernel=blur_kernel_size)

        depth_rgb = np.stack([depth, depth, depth], axis=-1)
        tensor = torch.from_numpy(depth_rgb.astype(np.float32)).unsqueeze(0)

        # Preview and original alpha blending
        if show_preview:
            preview_arr = depth_rgb.copy()
            if show_basebg and orig_arr is not None:
                if orig_arr.shape[:2] != (h, w):
                    resized_orig = cv2.resize(orig_arr, (w, h))
                else:
                    resized_orig = orig_arr
                
                if resized_orig.ndim == 2:
                    resized_orig = np.stack([resized_orig]*3, axis=-1)
                elif resized_orig.shape[2] == 4:
                    resized_orig = resized_orig[:, :, :3]
                
                orig_float = resized_orig.astype(np.float32) / 255.0
                alpha = 0.4  # 40% original, 60% depth
                preview_arr = (orig_float * alpha) + (depth_rgb * (1.0 - alpha))
                preview_arr = np.clip(preview_arr, 0.0, 1.0)

            preview_tensor = torch.from_numpy(preview_arr.astype(np.float32)).unsqueeze(0)
            return IO.NodeOutput(tensor, ui=UI.PreviewImage(preview_tensor))

        return IO.NodeOutput(tensor)
#----------------------------------------
POSE_NODE_CLASS_MAPPINGS = {
    "EasyOpenPoseEditor": EasyOpenPoseEditor,
    "EasyPoseDepthConverter": EasyPoseDepthConverter,
}

POSE_NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyOpenPoseEditor": "오픈포즈 에디터 및 좌표 추출기",
    "EasyPoseDepthConverter": "포즈 → 뎁스준비이미지 변환기",
}
#----------------------------------------
