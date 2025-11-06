"""
SAHI切片推理辅助类，专门用于大图像和畸变目标的车牌检测
"""
import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics.engine.results import Results
from ultralytics import YOLO
from model_library.model.base_model import device
device_param = device

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("Warning: SAHI not installed. Please install with: pip install sahi")


class SAHIPlateDetector:
    """SAHI车牌检测器"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.15, device: str = None):
        """
        初始化SAHI检测器

        Args:
            model_path: YOLO模型路径
            confidence_threshold: 初始检测置信度阈值
            device: 设备类型 ('cuda:0' 或 'cpu')
        """
        if not SAHI_AVAILABLE:
            raise ImportError("SAHI is required for sliced prediction. Install with: pip install sahi")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # 自动选择设备
        if device is None:
            self.device = device_param
        else:
            self.device = device

        print(f"SAHI检测器初始化，使用设备: {self.device}")

        # 初始化SAHI检测模型
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=self.device,
        )

        # 用于转换回YOLO Results格式的原始模型
        self.yolo_model = YOLO(model_path)

    def _determine_slice_params(self, image_height: int, image_width: int) -> tuple:
        """
        根据图像尺寸动态确定切片参数

        Args:
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            slice_height, slice_width, overlap_height_ratio, overlap_width_ratio
        """
        if image_height > 1500 or image_width > 1500:
            # 大图像使用较大的切片和重叠率
            slice_height = slice_width = 512
            overlap_height_ratio = overlap_width_ratio = 0.25
        elif image_height > 800 or image_width > 800:
            # 中等图像
            slice_height = slice_width = 512
            overlap_height_ratio = overlap_width_ratio = 0.25
        else:
            # 小图像
            slice_height = slice_width = 256
            overlap_height_ratio = overlap_width_ratio = 0.2

        return slice_height, slice_width, overlap_height_ratio, overlap_width_ratio

    def _convert_sahi_to_yolo_results(self, sahi_result, original_image: np.ndarray,
                                    final_confidence_threshold: float = 0.5) -> List[Results]:
        """
        将SAHI检测结果转换为YOLO Results格式

        Args:
            sahi_result: SAHI预测结果
            original_image: 原始图像
            final_confidence_threshold: 最终置信度过滤阈值

        Returns:
            转换后的YOLO Results列表
        """
        # 过滤低置信度目标
        filtered_detections = []
        for obj_pred in sahi_result.object_prediction_list:
            if obj_pred.score.value >= final_confidence_threshold:
                filtered_detections.append(obj_pred)

        print(f"SAHI检测: 原始检测 {len(sahi_result.object_prediction_list)} 个, "
              f"保留 {len(filtered_detections)} 个 (置信度≥{final_confidence_threshold})")

        if not filtered_detections:
            # 返回空结果，简化处理以避免Boxes构造问题
            # 创建一个没有boxes的Results，让post_process方法正确处理空结果
            results = Results(
                orig_img=original_image,
                path=None,
                names={0: 'license plate', 1: 'license plate_double'}
            )
            return [results]

        # 创建检测结果数据
        detections = []
        for obj_pred in filtered_detections:
            bbox = obj_pred.bbox
            cls = obj_pred.category.id
            conf = obj_pred.score.value

            # 转换为YOLO格式: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            detections.append([x1, y1, x2, y2, conf, cls])

        # 转换为numpy数组
        detections = np.array(detections) if detections else np.empty((0, 6))

        # 创建YOLO Results对象
        results = Results(
            orig_img=original_image,
            path=None,
            names={0: 'license plate', 1: 'license plate_double'},  # 支持单双层车牌
            boxes=detections
        )

        return [results]

    def detect_with_sahi(self, image_path: str, final_confidence_threshold: float = 0.5,
                        slice_params: Optional[Dict] = None) -> List[Results]:
        """
        使用SAHI进行切片推理检测

        Args:
            image_path: 图像路径
            final_confidence_threshold: 最终置信度过滤阈值
            slice_params: 自定义切片参数，如果为None则自动确定

        Returns:
            YOLO Results格式的检测结果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 读取图像
        image_for_slicing = read_image(image_path)
        original_image = cv2.imread(image_path)
        h, w = image_for_slicing.shape[:2]

        print(f"SAHI切片推理开始 - 图像尺寸: {w}x{h}")

        # 确定切片参数
        if slice_params is None:
            slice_height, slice_width, overlap_h_ratio, overlap_w_ratio = self._determine_slice_params(h, w)
        elif isinstance(slice_params, dict):
            slice_height = slice_params.get('slice_height', 512)
            slice_width = slice_params.get('slice_width', 512)
            overlap_h_ratio = slice_params.get('overlap_height_ratio', 0.25)
            overlap_w_ratio = slice_params.get('overlap_width_ratio', 0.25)
        else:
            print(f"警告: slice_params不是字典类型，收到: {type(slice_params)}，使用自动参数")
            slice_height, slice_width, overlap_h_ratio, overlap_w_ratio = self._determine_slice_params(h, w)

        print(f"切片参数: {slice_width}x{slice_height}, 重叠率: {overlap_w_ratio:.1%}x{overlap_h_ratio:.1%}")

        # 执行SAHI切片推理
        sahi_result = get_sliced_prediction(
            image=image_for_slicing,
            detection_model=self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_h_ratio,
            overlap_width_ratio=overlap_w_ratio,
        )

        print(f"SAHI切片推理完成，检测到 {len(sahi_result.object_prediction_list)} 个目标")

        # 转换为YOLO Results格式
        yolo_results = self._convert_sahi_to_yolo_results(
            sahi_result, original_image, final_confidence_threshold
        )

        return yolo_results

    def detect_with_standard_yolo(self, image_path: str, conf: float = 0.5,
                                imgsz: tuple = (640, 640)) -> List[Results]:
        """
        使用标准YOLO进行检测（作为SAHI的备选方案）

        Args:
            image_path: 图像路径
            conf: 置信度阈值
            imgsz: 图像尺寸

        Returns:
            YOLO Results格式的检测结果
        """
        results = self.yolo_model.predict(
            source=image_path,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
        return results