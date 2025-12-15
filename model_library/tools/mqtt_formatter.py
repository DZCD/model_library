"""MQTT消息格式化工具模块"""
from typing import Dict, Any, List


class MQTTMessageFormatter:
    """MQTT消息格式化器，负责生成各种类型的MQTT消息格式，保持与现有代码完全一致"""

    @staticmethod
    def format_accident_message(
        object_name: str,
        accident_item: Dict[str, Any],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化事故检测MQTT消息 - 保持与detector.py:151-162完全一致

        Args:
            object_name: 事故图片存储对象名
            accident_item: 事故检测项
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 事故检测MQTT消息
        """
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = accident_item.get("objNum", len(accident_item) if isinstance(accident_item, list) else 1)
        mqtt_message["imageInfo"]["boxs"] = accident_item
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        return mqtt_message

    @staticmethod
    def format_fire_lane_violation_message(
        object_name: str,
        result_item: Dict[str, Any],
        obj_num: int,
        ori_img_shape: tuple
    ) -> Dict[str, Any]:
        """
        格式化消防通道占用违规MQTT消息 - 保持与detector.py:427-437完全一致

        Args:
            object_name: 违规图片存储对象名
            result_item: 检测结果项
            obj_num: 检测到的对象数量 (len(result))
            ori_img_shape: 原始图像尺寸

        Returns:
            dict: 消防通道占用MQTT消息
        """
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = obj_num
        mqtt_message["imageInfo"]["boxs"] = result_item
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["message"] = "检测到消防通道被占用"

        return mqtt_message

    @staticmethod
    def format_general_detection_message(
        object_name: str,
        result_item: Dict[str, Any],
        obj_num: int,
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化通用检测MQTT消息 - 保持与detector.py:604-616完全一致

        Args:
            object_name: 检测图片存储对象名
            result_item: 检测结果项
            obj_num: 检测到的对象数量 (len(result))
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 通用检测MQTT消息
        """
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = obj_num
        mqtt_message["imageInfo"]["boxs"] = result_item
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["message"] = "检测到目标"
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        return mqtt_message