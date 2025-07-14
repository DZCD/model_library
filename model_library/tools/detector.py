"""检测器，视频流后台推理任务。适用于对接开发部的工作流程"""
import cv2
import asyncio
from datetime import datetime

from ..model.base_model import BaseModel
from ..model.track_fireland import TrackFireland
from ..model.track_accident import TrackAccident
from ..tools.utils import Config
from collections import defaultdict
from ..client.mqtt_client import MQTTClient
from ..client.minio_client import MinioClient


class Detector:
    def __init__(self, model_index: int, video_path: str, pixel_position: list = None):
        self.model_index = model_index
        self.video_path = video_path
        self.pixel_position = pixel_position
        self.config = Config()
        self.mqtt_client = MQTTClient()
        self.mqtt_client.connect()
        self.minio_client = MinioClient()
        self.model_name = self.config.model_list[self.model_index]['model_name']
        self.classes = self.config.model_list[self.model_index].get('classes', [0])
        self.time_step = self.config.model_list[self.model_index].get('time_step', 60)  # 推送间隔
        self.topic = self.get_topic()
        self.model = self.load_model()

        # 添加停止控制机制
        self._should_stop = False
        self._stop_event = asyncio.Event()

    def load_model(self):
        model_path = self.config.model_list[self.model_index]['model_path']
        if self.model_index == 0:
            return BaseModel(model_path)
        elif self.model_index == 1:
            return TrackFireland(model_path)
        elif self.model_index == 2:
            return BaseModel(model_path)
        elif self.model_index == 3:
            return TrackAccident(model_path)
        else:
            raise ValueError(f"Invalid model index: {self.model_index}")

    def get_topic(self):
        current_timestamp = datetime.now()
        datetime_str = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")  # 精确到秒
        topic_name = f"{datetime_str}-{self.model_name}"
        return topic_name
    
    def request_stop(self):
        """请求停止workflow"""
        self._should_stop = True
        self._stop_event.set()
        print(f"Workflow停止请求已发送")
    
    def is_stop_requested(self):
        """检查是否收到停止请求"""
        return self._should_stop
    
    async def check_stop(self):
        """异步检查停止请求"""
        if self._should_stop:
            print("检测到停止请求，正在停止workflow...")
            return True
        return False


    async def run_video(self):
        # 获取视频FPS
        max_retries = 3 # 最大重试次数
        current_retry = 0
        while current_retry <= max_retries:
            try:
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    cap.release()
                    raise Exception(f"无法连接到视频流: {self.video_path}")
                fps = cap.get(cv2.CAP_PROP_FPS)
                vid_stride = int(fps/2)  # 每秒推理2帧
                vid_stride  = vid_stride if vid_stride>0 else 1
                width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                break
            except Exception as e:
                print(f"视频流处理失败 (尝试 {current_retry}/{max_retries}): {str(e)}")
                current_retry +=1
                if current_retry <= max_retries:
                    print(f"等待 {3} 秒后重试...")
                    await asyncio.sleep(3)
                else:
                    print("已达到最大重试次数，停止重试")
                    self.mqtt_client.disconnect()
                    raise e

        if self.model_index == 1:
            results = self.model.track_video(self.video_path, stream=True, vid_stride=vid_stride, imgsz=(height, width),verbose=False)
        elif self.model_index == 3:
            results = self.model.track_video(self.video_path, stream=True,vid_stride=vid_stride, classes=self.classes, imgsz=(height, width),verbose=False)
        else:
            # results = self.model.detect_video(self.video_path, stream=True, vid_stride=vid_stride,
            #                                   imgsz=(height, width),verbose=False)
            results = self.model.track_video(self.video_path, stream=True, vid_stride=vid_stride,imgsz=(height, width),verbose=False)
        current_timestamp = datetime.now()
        date_str = current_timestamp.strftime("%Y-%m-%d")
        if self.model_index == 1:
            # 消防通道占用
            track_records = defaultdict(lambda: {'first_seen': None, 'last_seen': None, 'violation': False})
            time_threshold = self.config.model_list[self.model_index].get('time_threshold', 30)  # 默认30秒
            current_frame_ids = set()
            # 添加连续未出现帧数跟踪
            consecutive_missing_frames = defaultdict(int)
            max_missing_frames = 5  # 连续5帧未出现则移除

            frame_count = 0

            for result in results:
                # 检查停止请求
                if await self.check_stop():
                    print("模型1：收到停止请求，退出推理循环")
                    return
                
                ori_img_shape = result.orig_shape
                frame_count += 1
                current_time = frame_count * vid_stride / fps  # 当前视频时间（秒）

                results_dict = self.model.post_process([result], pixel_position=self.pixel_position)

                # 获取当前帧检测到的所有track_id并处理
                current_frame_ids = set()
                for result_item in results_dict:
                    track_id = result_item.get('track_id', None)
                    if track_id is not None:
                        current_frame_ids.add(track_id)

                        # 更新追踪记录
                        if track_records[track_id]['first_seen'] is None:
                            track_records[track_id]['first_seen'] = current_time
                            print(f"检测到新目标 ID:{track_id} 进入消防通道")

                        track_records[track_id]['last_seen'] = current_time

                        # 检查是否违规
                        duration = current_time - track_records[track_id]['first_seen']
                        if duration >= time_threshold:
                            track_records[track_id]['violation'] = True
                            object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{track_id}.jpg"
                            print(object_name)
                            infer_image = result.plot()
                            # _, _ = self.minio_client.upload_image_array(
                            #     image_array=infer_image,
                            #     object_name=object_name,
                            #     image_format='jpg',
                            #     quality=85
                            # )
                            mqtt_message = {"imageInfo": {}}
                            mqtt_message["imageInfo"]["imageId"] = ""
                            mqtt_message["imageInfo"]["dataType"] = "url"
                            mqtt_message["imageInfo"]["imageUrl"] = object_name
                            mqtt_message["imageInfo"]["data"] = ""
                            mqtt_message["imageInfo"]["objNum"] = len(result)
                            mqtt_message["imageInfo"]["boxs"] = result_item
                            mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[0]
                            mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[1]
                            mqtt_message["imageInfo"]["imageSize"] = ""
                            mqtt_message["imageInfo"]["message"] = "检测到消防通道被占用"
                            # 发送到MQTT主题: {类别名}
                            print(mqtt_message)
                            # mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

                # 更新所有track_id的连续未出现帧数
                for track_id in list(track_records.keys()):
                    if track_id not in current_frame_ids:
                        consecutive_missing_frames[track_id] += 1
                        # 如果连续5帧未出现，则从track_records中移除
                        if consecutive_missing_frames[track_id] >= max_missing_frames:
                            print(f"目标ID:{track_id} 连续{max_missing_frames}帧未检测到，已从追踪记录中移除")
                            del track_records[track_id]
                            del consecutive_missing_frames[track_id]
                    else:
                        # 重置连续未出现帧数
                        consecutive_missing_frames[track_id] = 0

        elif self.model_index == 3:
            # 事故检测模型
            accdent_id = []

            frame_count = 0
            for result in results:
                # 检查停止请求
                if await self.check_stop():
                    print("模型3：收到停止请求，退出推理循环")
                    return
                
                if len(result) ==0:
                    continue
                frame_count += 1

                # 后处理检测结果
                results_list = self.model.post_process([result])
                ori_img_shape = result.orig_shape
                if not results_list:
                    continue

                for result_item in results_list:
                    id = result_item.get('track_id', None)
                    if id not in accdent_id:
                        object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{id}.jpg"
                        print(object_name)
                        accdent_id.append(id)
                        infer_image = result.plot()
                        _, _ = self.minio_client.upload_image_array(
                            image_array=infer_image,
                            object_name=object_name,
                            image_format='jpg',
                            quality=85
                        )
                        mqtt_message = {"imageInfo": {}}
                        mqtt_message["imageInfo"]["imageId"] = ""
                        mqtt_message["imageInfo"]["dataType"] = "url"
                        mqtt_message["imageInfo"]["imageUrl"] = object_name
                        mqtt_message["imageInfo"]["data"] = ""
                        mqtt_message["imageInfo"]["objNum"] = len(result)
                        mqtt_message["imageInfo"]["boxs"] = result_item
                        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[0]
                        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[1]
                        mqtt_message["imageInfo"]["imageSize"] = ""
                        # 发送到MQTT主题: {类别名}
                        print(mqtt_message)
                        mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

        else:
            for result in results:
                # 检查停止请求
                if await self.check_stop():
                    print("其他模型：收到停止请求，退出推理循环")
                    return

                ori_img_shape = result.orig_shape
                results_dict = self.model.post_process([result])
                type_id = []
                for result_item in results_dict:
                    id = result_item.get('track_id', None)
                    if id not in type_id:
                        object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{id}.jpg"
                        print(object_name)
                        type_id.append(id)
                        infer_image = result.plot()
                        # _, _ = self.minio_client.upload_image_array(
                        #     image_array=infer_image,
                        #     object_name=object_name,
                        #     image_format='jpg',
                        #     quality=85
                        # )
                        mqtt_message = {"imageInfo": {}}
                        mqtt_message["imageInfo"]["imageId"] = ""
                        mqtt_message["imageInfo"]["dataType"] = "url"
                        mqtt_message["imageInfo"]["imageUrl"] = object_name
                        mqtt_message["imageInfo"]["data"] = ""
                        mqtt_message["imageInfo"]["objNum"] = len(result)
                        mqtt_message["imageInfo"]["boxs"] = result_item
                        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[0]
                        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[1]
                        mqtt_message["imageInfo"]["imageSize"] = ""
                        # 发送到MQTT主题: {类别名}
                        print(mqtt_message)
                        # mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)




