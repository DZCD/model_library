"""事故验证策略模块 - 提供多种事故验证策略"""
import math
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from shapely.geometry import Polygon


class AccidentVerificationStrategy(ABC):
    """事故验证策略基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """
        验证事故是否为真实事故

        Args:
            accident_boxes: 事故目标框列表
            pedestrian_boxes: 行人目标框列表

        Returns:
            List[int]: 被验证为真实事故的事故目标框索引列表
        """
        pass

    @abstractmethod
    def should_process_without_pedestrians(self) -> bool:
        """
        当没有行人时是否处理事故

        Returns:
            bool: True表示处理所有事故，False表示忽略所有事故
        """
        pass

    @staticmethod
    def box_to_polygon(box: Dict) -> Polygon:
        """将目标框转换为多边形"""
        x, y, w, h, angle = box['x'], box['y'], box['width'], box['height'], box['rotation']
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
        vertices = [(cx * cos_a - cy * sin_a + x, cx * sin_a + cy * cos_a + y)
                   for cx, cy in corners]
        return Polygon(vertices)

    @staticmethod
    def box_center(box: Dict) -> tuple:
        """获取目标框中心点"""
        return (box['x'], box['y'])


class OverlapStrategy(AccidentVerificationStrategy):
    """面积重叠验证策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.overlap_threshold = config.get('overlap_threshold', 0.3)

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """基于面积重叠验证事故"""
        verified_indices = []

        for i, accident_box in enumerate(accident_boxes):
            accident_poly = self.box_to_polygon(accident_box)

            # 检查与每个行人框的重叠
            has_overlap = False
            for pedestrian_box in pedestrian_boxes:
                pedestrian_poly = self.box_to_polygon(pedestrian_box)

                # 计算重叠面积
                intersection = accident_poly.intersection(pedestrian_poly)
                if intersection.area > 0:
                    overlap_ratio = intersection.area / min(accident_poly.area, pedestrian_poly.area)
                    if overlap_ratio >= self.overlap_threshold:
                        has_overlap = True
                        break

            if has_overlap:
                verified_indices.append(i)

        return verified_indices

    def should_process_without_pedestrians(self) -> bool:
        """重叠策略需要行人验证，无行人时忽略事故"""
        return False


class DistanceStrategy(AccidentVerificationStrategy):
    """距离验证策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.distance_threshold = config.get('distance_threshold', 150)  # 默认150像素
        self.use_center_distance = config.get('use_center_distance', True)

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """基于距离验证事故"""
        verified_indices = []

        for i, accident_box in enumerate(accident_boxes):
            accident_center = self.box_center(accident_box)

            # 检查与每个行人框的距离
            has_nearby_pedestrian = False
            for pedestrian_box in pedestrian_boxes:
                if self.use_center_distance:
                    # 使用中心点距离
                    pedestrian_center = self.box_center(pedestrian_box)
                    distance = math.sqrt(
                        (accident_center[0] - pedestrian_center[0])**2 +
                        (accident_center[1] - pedestrian_center[1])**2
                    )
                else:
                    # 使用边界框之间的最小距离
                    accident_poly = self.box_to_polygon(accident_box)
                    pedestrian_poly = self.box_to_polygon(pedestrian_box)
                    distance = accident_poly.distance(pedestrian_poly)

                if distance <= self.distance_threshold:
                    has_nearby_pedestrian = True
                    break

            if has_nearby_pedestrian:
                verified_indices.append(i)

        return verified_indices

    def should_process_without_pedestrians(self) -> bool:
        """距离策略需要行人验证，无行人时忽略事故"""
        return False


class CombinedStrategy(AccidentVerificationStrategy):
    """联合验证策略 - 满足任一条件即可"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 初始化子策略
        self.overlap_strategy = OverlapStrategy(config)
        self.distance_strategy = DistanceStrategy(config)
        self.require_both = config.get('require_both', False)  # 是否需要同时满足两个条件

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """联合验证事故"""
        overlap_indices = set(self.overlap_strategy.verify(accident_boxes, pedestrian_boxes))
        distance_indices = set(self.distance_strategy.verify(accident_boxes, pedestrian_boxes))

        if self.require_both:
            # 需要同时满足两个条件
            verified_indices = list(overlap_indices & distance_indices)
        else:
            # 满足任一条件即可
            verified_indices = list(overlap_indices | distance_indices)

        return verified_indices

    def should_process_without_pedestrians(self) -> bool:
        """联合策略需要行人验证，无行人时忽略事故"""
        return False


class NoVerificationStrategy(AccidentVerificationStrategy):
    """无验证策略 - 所有检测到的事故都认为是真实的"""

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """不进行验证，返回所有事故索引"""
        return list(range(len(accident_boxes)))

    def should_process_without_pedestrians(self) -> bool:
        """无验证策略处理所有事故，无论是否有行人"""
        return True


class AccidentVerificationManager:
    """事故验证管理器 - 负责完整的事故验证和处理逻辑"""

    def __init__(self, strategy: AccidentVerificationStrategy, config: Dict[str, Any]):
        self.strategy = strategy
        self.config = config
        self.class_confidence = config.get('class_confidence', {})

    def get_verified_accidents(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """
        获取通过验证的事故索引

        Args:
            accident_boxes: 事故目标框列表
            pedestrian_boxes: 行人目标框列表

        Returns:
            List[int]: 通过验证的事故索引列表
        """
        if not accident_boxes:
            return []

        # 如果没有行人，根据策略决定是否处理
        if not pedestrian_boxes:
            if self.strategy.should_process_without_pedestrians():
                return self.strategy.verify(accident_boxes, pedestrian_boxes)
            else:
                return []

        # 有行人时，使用策略进行验证
        return self.strategy.verify(accident_boxes, pedestrian_boxes)

    def apply_class_confidence_thresholds(self, model) -> bool:
        """
        应用分类别置信度阈值到模型

        Args:
            model: 要设置阈值的模型

        Returns:
            bool: 是否成功设置
        """
        try:
            if hasattr(model, 'set_class_thresholds') and self.class_confidence:
                accident_threshold = self.class_confidence.get('accident')
                pedestrian_threshold = self.class_confidence.get('pedestrian')
                model.set_class_thresholds(
                    accident_threshold=accident_threshold,
                    pedestrian_threshold=pedestrian_threshold
                )
                return True
            return False
        except Exception:
            return False

    def plot_verified_accidents_only(self, result, verified_accident_items):
        """
        只绘制验证后的真实事故框，不绘制行人框

        Args:
            result: YOLO检测结果
            verified_accident_items: 验证后的事故检测项列表

        Returns:
            numpy.ndarray: 绘制后的图像
        """
        try:
            from ultralytics.utils.plotting import colors

            # 复制原始图像
            plot_img = result.orig_img.copy()

            # 只绘制验证后的事故框
            for accident_item in verified_accident_items:
                # 获取事故框参数
                x, y, w, h, angle = accident_item['x'], accident_item['y'], accident_item['width'], accident_item['height'], accident_item['rotation']
                track_id = accident_item.get('track_id', 'unknown')
                confidence = accident_item['score']

                # 计算旋转矩形的四个角点
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                rotated_corners = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T + np.array([x, y])

                # 转换为整数坐标
                points = rotated_corners.astype(int)

                # 选择颜色 - 使用红色表示事故
                color = colors(0, True)  # class_id=0 对应事故类别

                # 绘制旋转矩形
                cv2.polylines(plot_img, [points], True, color, 2)

                # 绘制标签背景
                label = f"accident {confidence:.2f} id:{track_id}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # 确定标签位置（使用矩形左上角）
                label_x, label_y = int(points[0][0]), int(points[0][1]) - 10

                # 确保标签不超出图像边界
                label_x = max(0, min(label_x, plot_img.shape[1] - label_width))
                label_y = max(label_height, min(label_y, plot_img.shape[0] - 5))

                # 绘制标签背景
                cv2.rectangle(plot_img, (label_x, label_y - label_height),
                             (label_x + label_width, label_y + 5), color, -1)

                # 绘制标签文字
                cv2.putText(plot_img, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            return plot_img
        except Exception as e:
            # 如果绘制失败，返回原图
            return result.orig_img.copy()

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取当前策略信息"""
        return {
            'strategy_name': self.strategy.__class__.__name__,
            'config': self.strategy.config,
            'class_confidence': self.class_confidence,
            'should_process_without_pedestrians': self.strategy.should_process_without_pedestrians()
        }


class AccidentStrategyFactory:
    """事故验证策略工厂"""

    _strategies = {
        'overlap': OverlapStrategy,
        'distance': DistanceStrategy,
        'combined': CombinedStrategy,
        'none': NoVerificationStrategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> AccidentVerificationStrategy:
        """创建验证策略实例"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"未知的事故验证策略: {strategy_name}. 可用策略: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)

    @classmethod
    def create_manager(cls, strategy_name: str, config: Dict[str, Any]) -> AccidentVerificationManager:
        """创建验证管理器实例"""
        strategy = cls.create_strategy(strategy_name, config)
        return AccidentVerificationManager(strategy, config)

    @classmethod
    def create_complete_accident_system(cls, model_index: int, config, model, task_id: str = None) -> AccidentVerificationManager:
        """
        创建完整的事故识别系统（包括分类阈值设置）

        Args:
            model_index: 模型索引
            config: 配置对象
            model: 模型实例
            task_id: 任务ID

        Returns:
            AccidentVerificationManager: 完整的事故验证管理器
        """
        try:
            # 获取模型配置
            model_config = config.model_list[model_index]

            # 获取验证策略配置
            strategy_name = model_config.get('verification_strategy', 'overlap')
            verification_config = model_config.get('verification_config', {})

            # 获取分类阈值配置
            class_confidence = model_config.get('class_confidence', {})

            # 合并配置
            complete_config = {
                **verification_config,
                'class_confidence': class_confidence
            }

            # 创建管理器
            manager = cls.create_manager(strategy_name, complete_config)

            # 应用分类阈值
            success = manager.apply_class_confidence_thresholds(model)

            if task_id:
                from .logger import log_task, log_task_debug, log_task_error
                strategy_info = manager.get_strategy_info()
                log_task(f"事故识别系统初始化完成 - 任务ID:{task_id}, 策略:{strategy_name}")
                log_task(f"分类阈值应用结果 - 任务ID:{task_id}, 成功:{success}")
                log_task_debug(f"策略详情 - 任务ID:{task_id}, 详情:{strategy_info}")

            return manager

        except Exception as e:
            if task_id:
                from .logger import log_task_error
                log_task_error(f"事故识别系统初始化失败 - 任务ID:{task_id}, 错误:{str(e)}")
            # 使用默认配置
            return cls.create_manager('overlap', {'overlap_threshold': 0.3})

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取所有可用的策略名称"""
        return list(cls._strategies.keys())