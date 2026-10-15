import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "EasyOpenPoseEditor.DragWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EasyOpenPoseEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;
                const canvasSizeWidget = node.widgets.find(w => w.name === "canvas_size");
                node.dragAreaSize = canvasSizeWidget ? canvasSizeWidget.value : 400;
                node.selectedPointKey = null;

                node.cachedBgImage = null;
                node.lastB64Str = null;
                node.b64_img_data = "";
				node.pose_points_updated = "";

                if (node.widgets) {
                    node.widgets.forEach((w, index) => {
                        if (!w.options) w.options = {};
                        if (!w.options.id) w.options.id = `easy_openpose_widget_${index}_${w.name || 'field'}`;
                        if (!w.options.name) w.options.name = w.name || `field_${index}`;
                    });
                }

                const onExecuted = node.onExecuted;
                node.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                    if (message && message.b64_img) {
                        node.b64_img_data = Array.isArray(message.b64_img) ? message.b64_img[0] : message.b64_img;
                        node.setDirtyCanvas(true, true);
                    }
					// 파일 로드 시 pose_points 위젯 갱신 ← 추가
					if (message && message.pose_points_updated) {
                        const w = node.widgets?.find(w => w.name === "pose_points");
                        if (w) {
                            w.value = Array.isArray(message.pose_points_updated)
                                ? message.pose_points_updated[0]
                                : message.pose_points_updated;
                            if (w.callback) w.callback(w.value);
                            if (app.graph) app.graph.change();
                            node.setDirtyCanvas(true, true);
                        }
                    }
                };

                function parsePoints(str) {
                    if (!str || typeof str !== "string") return {};
                    const regex = /([A-Za-z0-9_]+)\s*\{\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\}/g;
                    const points = {};
                    let match;
                    while ((match = regex.exec(str)) !== null) {
                        points[match[1]] = { x: parseFloat(match[2]), y: parseFloat(match[3]) };
                    }
                    return points;
                }

                function serializePoints(pointsObj) {
                    return Object.entries(pointsObj)
                        .map(([label, pos]) => `${label}{${pos.x.toFixed(1)},${pos.y.toFixed(1)}}`)
                        .join(", ");
                }

                node.getWidgetValue = function(name) {
                    if (!node.widgets) return null;
                    const w = node.widgets.find(w => w.name === name);
                    return w ? w.value : null;
                };

                node.getControlPoints = function() {
                    const val = node.getWidgetValue("pose_points");
                    return parsePoints(val);
                };

                node.updateControlPoint = function(key, x, y) {
                    const w = node.widgets.find(w => w.name === "pose_points");
                    if (!w) return;
                    const points = parsePoints(w.value);
                    if (points[key]) {
                        points[key].x = Math.max(0, x);
                        points[key].y = Math.max(0, y);
                    } else {
                        points[key] = { x: x, y: y };
                    }
    
                    const serialized = serializePoints(points);
                    w.value = serialized;
    
                    // 만약 pose_points 위젯이 화면에 보이는 입력창이라면 value 갱신
                    if (w.inputEl) {
                        w.inputEl.value = serialized;
                    }

                    if (w.callback) w.callback(w.value);
                    if (app.graph) app.graph.change();
                };

                const BONE_CONNECTIONS = [
				    ["Nose", "REye", "right"], 
					["REye", "REar", "right"],
                    ["Nose", "LEye", "left"], 
					["LEye", "LEar", "left"],
                    ["Nose", "Neck", "center"],
                    ["Neck", "Chest", "center"],
                    ["Chest", "MidHip", "center"],
                    ["Chest", "LShoulder", "special"],
                    ["Chest", "RShoulder", "special"],
                    ["Chest", "LHip", "special"],
                    ["Chest", "RHip", "special"],
                    ["MidHip", "LHip", "special"],
                    ["MidHip", "RHip", "special"],
                    ["LShoulder", "LElbow", "left"],
                    ["LElbow", "LHand", "left"],
                    ["LHip", "LKnee", "left"],
                    ["LKnee", "LFeet", "left"],
                    ["RShoulder", "RElbow", "right"],
                    ["RElbow", "RHand", "right"],
                    ["RHip", "RKnee", "right"],
                    ["RKnee", "RFeet", "right"],
                ];

                const JOINT_COLORS = {
                    "Nose": "#0000FF", "Neck": "#8000FF", "REye": "#FF00AA", "LEye": "#FF00FF",
                    "REar": "#AA00FF", "LEar": "#5500FF", "Chest": "#FF8000", "MidHip": "#808080",
                    "LShoulder": "#00FF55", "RShoulder": "#00AAFF", "LElbow": "#00FF00", "RElbow": "#00FFFF",
                    "LHand": "#55FF00", "RHand": "#00FFAA", "LHip": "#FF5500", "RHip": "#AAFF00",
                    "LKnee": "#0055FF", "RKnee": "#FFFF00", "LFeet": "#FF0055", "RFeet": "#FFAA00"
                };

                const customWidget = {
                    type: "grid_canvas",
                    name: "openpose_interactive_canvas",
                    label: "OpenPose Interactive Editor",
                    value: "",
                    options: { id: "openpose_interactive_canvas_id", name: "openpose_interactive_canvas" },
                    draw: function(ctx, widgetNode, width, y) {
                        if (widgetNode.flags.collapsed) return;

                        this.last_y = y + 20;
                        ctx.save();

                        const startX = (width - widgetNode.dragAreaSize) / 2;
                        const startY = y + 20;
                        const areaSize = widgetNode.dragAreaSize;

                        const previewBaseBg = widgetNode.getWidgetValue("preview_basebg") ?? true;
                        const expandRatioVal = parseInt(widgetNode.getWidgetValue("expand_ratio") || 0, 10);
                        const b64Data = widgetNode.b64_img_data;

                        ctx.fillStyle = "#111";
                        ctx.fillRect(startX, startY, areaSize, areaSize);

                        if (previewBaseBg && b64Data) {
                            if (widgetNode.lastB64Str !== b64Data) {
                                widgetNode.lastB64Str = b64Data;
                                widgetNode.cachedBgImage = new Image();
                                widgetNode.cachedBgImage.onload = () => {
                                    widgetNode.setDirtyCanvas(true, true);
                                };
                                let imgSrc = b64Data.startsWith("data:") ? b64Data : "data:image/png;base64," + b64Data;
                                widgetNode.cachedBgImage.src = imgSrc;
                            }

                            if (widgetNode.cachedBgImage && widgetNode.cachedBgImage.complete && widgetNode.cachedBgImage.naturalWidth > 0) {
                                const imgW = widgetNode.cachedBgImage.naturalWidth;
                                const imgH = widgetNode.cachedBgImage.naturalHeight;
                                
                                let drawW = areaSize;
                                let drawH = areaSize;
                                let offsetX = startX;
                                let offsetY = startY;

                                if (imgW > imgH) {
                                    drawH = areaSize * (imgH / imgW);
                                    offsetY = startY + (areaSize - drawH) / 2;
                                } else {
                                    drawW = areaSize * (imgW / imgH);
                                    offsetX = startX + (areaSize - drawW) / 2;
                                }

                                widgetNode._renderScaleX = drawW / imgW;
                                widgetNode._renderScaleY = drawH / imgH;
                                widgetNode._renderOffsetX = offsetX;
                                widgetNode._renderOffsetY = offsetY;
                                widgetNode._imgRealW = imgW;
                                widgetNode._imgRealH = imgH;

                                ctx.drawImage(widgetNode.cachedBgImage, offsetX, offsetY, drawW, drawH);
                            }
                        } else {
                            widgetNode._renderScaleX = areaSize / 512;
                            widgetNode._renderScaleY = areaSize / 512;
                            widgetNode._renderOffsetX = startX;
                            widgetNode._renderOffsetY = startY;
                            widgetNode._imgRealW = 512;
                            widgetNode._imgRealH = 512;
                        }

                        // 백엔드의 패딩(expand_ratio) 기준을 반영하기 위한 오프셋 계산
                        // 백엔드에서 원본 이미지 크기(w, h)를 기준으로 패딩이 더해지므로, 
                        // 실제 이미지 영역 내부에서의 상대 좌표계로 변환해줍니다.
                        const padX = expandRatioVal > 1 ? Math.min(expandRatioVal, 512) : 0;
                        const padY = padX;

                        const points = widgetNode.getControlPoints();
                        const scaleX = widgetNode._renderScaleX;
                        const scaleY = widgetNode._renderScaleY;
                        const offX = widgetNode._renderOffsetX;
                        const offY = widgetNode._renderOffsetY;

                        // 1. 뼈대 연결선 렌더링 (패딩 반영)
						const pointRadius = parseFloat(widgetNode.getWidgetValue("point_radius") || 4.0);
						const lineThickness = parseInt(widgetNode.getWidgetValue("line_thickness") || 3, 10);
                        ctx.lineWidth = lineThickness;
                        BONE_CONNECTIONS.forEach(([p1Name, p2Name]) => {
                            const p1 = points[p1Name];
                            const p2 = points[p2Name];
                            if (p1 && p2) {
                                ctx.strokeStyle = JOINT_COLORS[p1Name] || "#FFFFFF";
                                ctx.beginPath();
                                ctx.moveTo(offX + (p1.x + padX) * scaleX, offY + (p1.y + padY) * scaleY);
                                ctx.lineTo(offX + (p2.x + padX) * scaleX, offY + (p2.y + padY) * scaleY);
                                ctx.stroke();
                            }
                        });

                        // 2. 관절 포인트(원) 렌더링 (패딩 반영)
                        for (let [key, p] of Object.entries(points)) {
                            const cx = offX + (p.x + padX) * scaleX;
                            const cy = offY + (p.y + padY) * scaleY;
                            
                            const r = (key === widgetNode.selectedPointKey) ? pointRadius * 1.5 : pointRadius;

							ctx.fillStyle = JOINT_COLORS[key] || "#FFFFFF";
                            ctx.beginPath();
                            ctx.arc(cx, cy, r, 0, Math.PI * 2); 
                            ctx.fill();
                            ctx.strokeStyle = "#FFFFFF";
                            ctx.lineWidth = 1.5;
                            ctx.stroke();
                        }

                        ctx.strokeStyle = "#444";
                        ctx.strokeRect(startX, startY, areaSize, areaSize);
                        ctx.restore();
                    },
                    computeSize: function(width) {
                        return [width, node.dragAreaSize + 35];
                    },
                    mouse: function(event, pos, widgetNode) {
                        const startX = (widgetNode.size[0] - widgetNode.dragAreaSize) / 2;
                        const startY = this.last_y || 20;

                        if (!widgetNode._renderScaleX || !widgetNode._imgRealW) return false;

                        const expandRatioVal = parseInt(widgetNode.getWidgetValue("expand_ratio") || 0, 10);
                        const padX = expandRatioVal > 1 ? Math.min(expandRatioVal, 512) : 0;
                        const padY = padX;

                        // 마우스 좌표를 이미지 원본 좌표계로 역산 (- pad 적용)
                        const localX = (pos[0] - widgetNode._renderOffsetX) / widgetNode._renderScaleX - padX;
                        const localY = (pos[1] - widgetNode._renderOffsetY) / widgetNode._renderScaleY - padY;

                        if (event.type === "pointerdown" || event.type === "mousedown") {
                            const points = widgetNode.getControlPoints();
                            for (let [key, p] of Object.entries(points)) {
                                const dist = Math.hypot(localX - p.x, localY - p.y);
                                if (dist < 20 / widgetNode._renderScaleX) {
                                    widgetNode.selectedPointKey = key;
                                    widgetNode.setDirtyCanvas(true, true);
                                    return true;
                                }
                            }
                            widgetNode.selectedPointKey = null;
                        } else if (event.type === "pointermove" || event.type === "mousemove") {
                            if (widgetNode.selectedPointKey !== null) {
                                const minX = -padX;
                                const maxX = widgetNode._imgRealW + padX;
                                const minY = -padY;
                                const maxY = widgetNode._imgRealH + padY;

                                const nx = Math.max(minX, Math.min(maxX, localX));
                                const ny = Math.max(minY, Math.min(maxY, localY));
                                widgetNode.updateControlPoint(widgetNode.selectedPointKey, nx, ny);
                                widgetNode.setDirtyCanvas(true, true);
                                return true;
                            }
                        } else if (event.type === "pointerup" || event.type === "mouseup") {
                            if (widgetNode.selectedPointKey !== null) {
                                widgetNode.selectedPointKey = null;
                                const w = widgetNode.widgets?.find(w => w.name === "pose_points");
                                if (w && w.callback) w.callback(w.value);
                                if (app.graph) app.graph.change();
                                widgetNode.setDirtyCanvas(true, true);
                            }
                        }
                        return false;
                    }
                };

                node.addCustomWidget(customWidget);
            };
        }
    }
});