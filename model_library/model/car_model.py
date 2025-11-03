from .base_model import BaseModel
from ultralytics import YOLO
import math
import numpy as np
import torch
from ultralytics.engine.results import OBB
from ultralytics.engine.results import Results
import cv2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CarModel(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                     verbose: bool = True, half=True, cv_obb=True):
        if classes is not None:
            results = self.model.track(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz,
                                       verbose=verbose, half=half, device=device)
        else:
            results = self.model.track(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, half=half,
                                       device=device)

        if cv_obb:
            obb_results = []
            for result in results:
                if len(result)==0:
                    return results
                obb_result = self._hbb_to_obb_yolo_result(result)
                obb_results.append(obb_result)
            return obb_results
        else:
            return results

    def _hbb_to_obb_yolo_result(self, result: Results):
        """将yolo的hbb Result类用cv2的方式转换为hbb"""
        boxes = result.boxes
        xyxy = boxes.xyxy.tolist()
        xywh = boxes.xywh.tolist()
        ori_image = result.orig_img
        ori_image_shape = ori_image.shape[:2]
        config = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        try:
            ids = boxes.id.tolist()
        except:
            ids = None

        obb_msg_list = []
        for i, box in enumerate(xyxy):
            # 计算原始HBB的面积
            x1, y1, x2, y2 = box
            hbb_area = (x2 - x1) * (y2 - y1)

            roi, range = self._get_hbb_to_obb_by_cv2(box, ori_image)
            roi_x1, roi_y1 = range[0:2]
            rect = self._get_rotated_rect_from_roi(roi)

            if not rect:
                angel_trans = 0
                rect = xywh[i]
            else:
                (center_x, center_y), (rect_w, rect_h), angle = rect
                # 转换坐标到原图坐标系
                global_center_x = center_x + roi_x1
                global_center_y = center_y + roi_y1
                if rect_w < rect_h:
                    angle = angle + 90
                    rect_w, rect_h = rect_h, rect_w
                angel_trans = math.radians(angle)

                # 计算转换后OBB的面积
                obb_area = rect_w * rect_h

                # 如果OBB面积比HBB面积小很多（比如小于50%），则使用原始HBB
                area_ratio = obb_area / hbb_area if hbb_area > 0 else 0
                if area_ratio < 0.5:  # 可以调整这个阈值
                    # 使用原始HBB的xywh格式
                    angel_trans = 0
                    rect = xywh[i]
                else:
                    rect = [global_center_x, global_center_y, rect_w, rect_h]
            if ids:
                obb_msg = list(rect) + [angel_trans] + [ids[i]] + [config[i]] + [classes[i]]
            else:
                obb_msg = list(rect) + [angel_trans] + [config[i]] + [classes[i]]

            obb_msg_list.append(obb_msg)
        obb_msg_tensor = torch.tensor(obb_msg_list)
        obb = OBB(obb_msg_tensor, ori_image_shape)
        obb_result = result[0]
        obb_result.obb = obb

        return obb_result

    @staticmethod
    def _get_hbb_to_obb_by_cv2(box, image, expand_ratio=0.3):
        """利用cv2的前后背景分离的方式将hbb转换为obb的方法"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)

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
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi, (roi_x1, roi_y1, roi_x2, roi_y2)

    @staticmethod
    def _get_rotated_rect_from_roi(roi_image):
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
            rect = (margin_w, margin_h, new_w - 2 * margin_w, new_h - 2 * margin_h)

            # 确保矩形有效
            if rect[2] <= 0 or rect[3] <= 0:
                rect = (1, 1, max(1, new_w - 2), max(1, new_h - 2))

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
