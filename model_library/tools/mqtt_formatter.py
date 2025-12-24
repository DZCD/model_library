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

    @staticmethod
    def format_motorcycle_frame_message(
        object_name: str,
        frame_report: Dict[str, Any],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化夜间红外摩托车飙车帧级别追踪MQTT消息（同一帧的多个飙车目标在一条消息中）

        Args:
            object_name: 检测图片存储对象名
            frame_report: 帧级别的飙车追踪报告，包含:
                - timestamp: 时间戳
                - gathering_count: 聚集数量
                - racing_count: 飙车数量（速度达到阈值）
                - tracking_infos: 追踪信息列表（多个飙车目标）
                - has_new_reports: 是否有新的上报目标
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 摩托车飙车帧级别追踪MQTT消息
        """
        tracking_infos = frame_report['tracking_infos']
        gathering_count = frame_report['gathering_count']
        racing_count = frame_report['racing_count']

        # 构建所有目标框信息
        boxes_data = []
        for tracking_info in tracking_infos:
            track_id = tracking_info['track_id']
            box_info = tracking_info['box']
            speed_info = tracking_info.get('speed')

            # 构建单个目标框信息
            box_data = {
                "x": box_info['x'],
                "y": box_info['y'],
                "width": box_info['width'],
                "height": box_info['height'],
                "score": box_info['score'],
                "track_id": track_id,
                "className": box_info.get('class', 'motorcycle')
            }

            # 所有飙车目标都包含速度信息
            if speed_info:
                box_data["speed_pixels_per_second"] = speed_info['pixels_per_second']
                box_data["direction_angle"] = speed_info['angle_degrees']
                box_data["direction"] = speed_info['direction']

            boxes_data.append(box_data)

        # 构建MQTT消息
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = len(boxes_data)
        mqtt_message["imageInfo"]["boxs"] = boxes_data  # 多个飙车目标框
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 构建消息内容（飙车统计信息）
        message_parts = []
        message_parts.append(f"检测到夜间摩托车飙车")
        message_parts.append(f"聚集数量:{gathering_count}")
        message_parts.append(f"飙车数量:{racing_count}")

        # 计算平均速度
        if tracking_infos:
            speeds = [info['speed']['pixels_per_second'] for info in tracking_infos if info.get('speed')]
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                message_parts.append(f"平均速度:{avg_speed:.1f}像素/秒")

        mqtt_message["imageInfo"]["message"] = ", ".join(message_parts)

        return mqtt_message


