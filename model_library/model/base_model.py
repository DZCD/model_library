"""
模型推理类,基于yolo
"""

from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import numpy as np
import math


class BaseModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                     verbose: bool = True):
        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz,
                                         verbose=verbose)
        else:
            results = self.model.predict(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose)
        return results

    def track_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3):
        if classes is not None:
            results = self.model.track(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose,
                                       iou=iou)
        else:
            results = self.model.track(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, iou=iou)
        return results

    def detect_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None,
                     imgsz: tuple = (640, 640), verbose: bool = True, iou=0.3):
        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                         imgsz=imgsz, iou=0.3, verbose=verbose)
        else:
            results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, iou=iou)
        return results

    def track_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3):
        if classes is not None:
            results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                       imgsz=imgsz, iou=iou, verbose=verbose)
        else:
            results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, verbose=verbose,
                                       iou=iou)
        return results

    def post_process(self, results: Results, **kwargs) -> list:
        """提取Results中的推理结果数据，包括框的坐标、类别、置信度"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue
            boxes = result.boxes
            if boxes:
                names = result.names
                xywh = boxes.xywh.tolist()
                xyxy = boxes.xyxy.tolist()
                cls = boxes.cls.tolist()
                conf = boxes.conf.tolist()
            else:
                boxes = result.obb
                if boxes is None:
                    continue
                names = result.names
                xywh = boxes.xywhr.tolist()
                xyxy = boxes.xyxyxyxy.tolist()
                cls = boxes.cls.tolist()
                conf = boxes.conf.tolist()

            try:
                track_id = boxes.id.tolist()
            except:
                track_id = "unknown"

            for i, box in enumerate(xywh):
                box_params = {
                    "x": box[0],
                    "y": box[1],
                    "width": box[2],
                    "height": box[3],
                    "rotation": box[4] if len(box) == 5 else 0,
                    "score": conf[i],
                    "xyxy":xyxy[i],
                    "track_id": track_id[i] if isinstance(track_id, list) else track_id,
                    "classed": cls[i],
                    "className": names[cls[i]],
                    "text": ""
                }
                results_dict.append(box_params)
        return results_dict
    
    def post_process_hbb_to_obb(self, results: Results, expand_ratio: float = 0.1, **kwargs) -> list:
        """
        将HBB检测结果转换为OBB格式
        使用轮廓检测计算旋转角度，返回YOLO OBB任务格式的结果
        
        Args:
            results: YOLO检测结果
            expand_ratio: ROI扩展比例，用于获得更好的轮廓检测效果
            **kwargs: 其他参数
            
        Returns:
            list: OBB格式的检测结果列表，每个元素包含旋转信息
        """
        results_dict = []
        
        for result in results:
            if len(result) == 0:
                continue
                
            # 只处理HBB检测结果
            boxes = result.boxes
            if not boxes:
                continue
                
            names = result.names
            xyxy = boxes.xyxy.tolist()
            cls = boxes.cls.tolist()
            conf = boxes.conf.tolist()
            
            # 获取原图用于轮廓检测
            orig_img = result.orig_img
            if orig_img is None:
                # 如果没有原图，使用普通HBB格式（角度为0）
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 将HBB转换为xyxyxyxy格式
                    hbb_points = [x1, y1, x2, y1, x2, y2, x1, y2]
                    
                    box_params = {
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height,
                        "rotation": 0.0,  # 弧度制的0
                        "score": conf[i],
                        "xyxy": hbb_points,  # 8个顶点坐标
                        "track_id": "unknown",
                        "classed": cls[i],
                        "className": names[cls[i]],
                        "text": ""
                    }
                    results_dict.append(box_params)
                continue
            
            # 获取track_id（如果有的话）
            try:
                track_id = boxes.id.tolist()
            except:
                track_id = "unknown"
                
            h, w = orig_img.shape[:2]
            
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                hbb_points = [x1, y1, x2, y1, x2, y2, x1, y2]

                # 扩展ROI区域以获得更好的轮廓检测效果
                box_w, box_h = x2 - x1, y2 - y1
                expand_w = int(box_w * expand_ratio)
                expand_h = int(box_h * expand_ratio)

                # 额外扩展像素距离，为GrabCut提供更多背景信息
                extra_expand = max(20, min(box_w, box_h) // 3)  # 至少20像素，或框尺寸的1/3

                roi_x1 = max(0, x1 - expand_w - extra_expand)
                roi_y1 = max(0, y1 - expand_h - extra_expand)
                roi_x2 = min(w, x2 + expand_w + extra_expand)
                roi_y2 = min(h, y2 + expand_h + extra_expand)

                # 裁剪ROI
                roi = orig_img[roi_y1:roi_y2, roi_x1:roi_x2]

                # 使用轮廓检测计算旋转矩形
                rotated_rect = self._get_rotated_rect_from_roi(roi)

                if rotated_rect is not None:
                    (center_x, center_y), (rect_w, rect_h), angle = rotated_rect

                    # 转换坐标到原图坐标系
                    global_center_x = center_x + roi_x1
                    global_center_y = center_y + roi_y1

                    # 角度标准化（让长边作为主要方向）
                    if rect_w < rect_h:
                        angle = angle + 90
                        rect_w, rect_h = rect_h, rect_w

                    # 角度范围调整到 [-90, 90]
                    while angle > 90:
                        angle -= 180
                    while angle < -90:
                        angle += 180

                    # 将角度转换为弧度制（YOLO OBB格式要求）
                    angle_radians = math.radians(angle)

                    # 计算旋转矩形的8个顶点坐标（xyxyxyxy格式）
                    box_points = cv2.boxPoints(((global_center_x, global_center_y), (rect_w, rect_h), angle))
                    box_points = box_points.reshape(-1).tolist()  # 转换为8个坐标值的列表

                    # 构建OBB格式的结果
                    box_params = {
                        "x": global_center_x,
                        "y": global_center_y,
                        "width": rect_w,
                        "height": rect_h,
                        "rotation": angle_radians,  # 使用弧度制
                        "score": conf[i],
                        "xyxy": box_points,  # 8个顶点坐标
                        "track_id": track_id[i] if isinstance(track_id, list) else track_id,
                        "classed": cls[i],
                        "className": names[cls[i]],
                        "text": ""
                    }
                else:
                    # 轮廓检测失败，使用原始HBB（角度为0）
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1


                    box_params = {
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height,
                        "rotation": 0.0,  # 弧度制的0
                        "score": conf[i],
                        "xyxy": hbb_points,  # 四个顶点坐标
                        "track_id": track_id[i] if isinstance(track_id, list) else track_id,
                        "classed": cls[i],
                        "className": names[cls[i]],
                        "text": ""
                    }
                
                results_dict.append(box_params)
                
        return results_dict
    
    def _get_rotated_rect_from_roi(self, roi_image):
        """
        从ROI图像中提取轮廓并计算旋转矩形
        使用图像缩放优化GrabCut速度
        
        Args:
            roi_image: ROI区域图像
            
        Returns:
            旋转矩形参数 ((center_x, center_y), (width, height), angle) 或 None
        """
        if roi_image.size == 0:
            return None
        
        h, w = roi_image.shape[:2]
        if h < 30 or w < 30:
            return None
        
        # 缩放到合理尺寸以提高GrabCut速度
        target_size = 400  # 目标最大边长
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            roi_small = cv2.resize(roi_image, (new_w, new_h))
        else:
            roi_small = roi_image
            new_h, new_w = h, w
            scale = 1.0

        try:
            # 在小图上运行GrabCut
            mask = np.zeros((new_h, new_w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 定义矩形区域作为前景的初始估计
            margin_w = max(1, new_w // 6)
            margin_h = max(1, new_h // 6)
            rect = (margin_w, margin_h, new_w - 2*margin_w, new_h - 2*margin_h)
            
            # 确保矩形有效
            if rect[2] <= 0 or rect[3] <= 0:
                rect = (1, 1, max(1, new_w-2), max(1, new_h-2))
            
            # 在小图上快速运行GrabCut
            cv2.grabCut(roi_small, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
            
            # 创建最终的二值mask
            final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # 形态学操作清理mask (使用更小的核)
            kernel = np.ones((2, 2), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            # 检查是否有足够的前景区域
            foreground_pixels = np.sum(final_mask)
            total_pixels = new_h * new_w
            
            # 前景比例检查
            if foreground_pixels < total_pixels * 0.05 or foreground_pixels > total_pixels * 0.8:
                return None
            
            # 将mask缩放回原尺寸
            if scale != 1.0:
                final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 从分割mask中查找轮廓
            final_mask = final_mask * 255
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:
                    rect = cv2.minAreaRect(largest_contour)
                    return rect
            
            return None
            
        except Exception:
            return None

    
