"""
模型推理类,基于yolo
"""

import os
import cv2

# ==================== OpenCV RTMP流修复 ====================
# 在导入YOLO前配置OpenCV后端，这是修复RTMP流问题的关键
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_IMAGES"] = "0"

# 猴子补丁：替换cv2.VideoCapture，强制RTMP流使用FFmpeg后端
_original_videocapture = cv2.VideoCapture

class _FixedVideoCapture(_original_videocapture):
    def __init__(self, *args, **kwargs):
        import time as _time
        
        # 检测是否为流媒体URL
        is_rtmp = len(args) > 0 and isinstance(args[0], str) and 'rtmp' in args[0].lower()
        if len(args) > 0 and isinstance(args[0], str):
            url = args[0].lower()
            is_stream = any(p in url for p in ['rtmp://', 'rtsp://', 'http://', 'https://'])
            
            # 为流媒体强制使用FFmpeg后端
            if is_stream and len(args) == 1 and 'apiPreference' not in kwargs:
                kwargs['apiPreference'] = cv2.CAP_FFMPEG
                print(f"[修复] 为流使用FFmpeg后端")
        
        # 对RTMP流进行多次重试（因为服务器可能有并发限制）
        max_retries = 5 if is_rtmp else 1
        retry_delay = 1.0  # 每次重试等待1秒
        
        for attempt in range(max_retries):
            super().__init__(*args, **kwargs)
            
            # 检查是否成功打开
            if self.isOpened():
                # 为RTMP流设置缓冲区
                if is_rtmp:
                    self.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    if attempt > 0:
                        print(f"[修复] ✅ RTMP流打开成功（第{attempt + 1}次尝试）")
                return
            else:
                # 打开失败
                if is_rtmp:
                    if attempt < max_retries - 1:
                        print(f"[修复] ⚠️ RTMP流打开失败，{retry_delay}秒后重试（{attempt + 1}/{max_retries}）...")
                        _time.sleep(retry_delay)
                    else:
                        print(f"[修复] ❌ RTMP流打开失败（已尝试{max_retries}次）")
                return  # 非RTMP流失败直接返回

# 全局替换
cv2.VideoCapture = _FixedVideoCapture
# ==================== 修复结束 ====================

from ultralytics import YOLO
from ultralytics.engine.results import Results
from datetime import datetime
import torch

# 设备检测：优先使用GPU，不可用时回退到CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


class BaseModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                     verbose: bool = True,half=True):
        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz,
                                         verbose=verbose,half=half, device=device)
        else:
            results = self.model.predict(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose,half=half, device=device)
        return results

    def track_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3,half=True):
        if classes is not None:
            results = self.model.track(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose,
                                       iou=iou,half=half, device=device)
        else:
            results = self.model.track(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, iou=iou,half=half, device=device)
        return results

    def detect_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None,
                     imgsz: tuple = (640, 640), verbose: bool = True, iou=0.3,half=True):
        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                         imgsz=imgsz, iou=0.3, verbose=verbose,half=half, device=device)
        else:
            results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, iou=iou,half=half, device=device)
        return results

    def track_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3,half =True):
        if classes is not None:
            results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                       imgsz=imgsz, iou=iou, verbose=verbose,half=half, device=device,tracker="botsort_cus.yaml")
        else:
            results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, verbose=verbose,
                                       iou=iou,half =half, device=device,tracker="botsort_cus.yaml")
        return results

    def post_process(self, results: Results, **kwargs) -> list:
        """提取Results中的推理结果数据，包括框的坐标、类别、置信度"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue
            boxes = result.obb
            if boxes:
                if boxes is None:
                    continue
                names = result.names
                xywh = boxes.xywhr.tolist()
                xyxy = boxes.xyxyxyxy.tolist()
                cls = boxes.cls.tolist()
                conf = boxes.conf.tolist()
            else:
                boxes = result.boxes
                names = result.names
                xywh = boxes.xywh.tolist()
                xyxy = boxes.xyxy.tolist()
                # 正确的OBB格式转换：将矩形框转换为4个角点的8个坐标值
                # 格式：[x1,y1, x2,y1, x2,y2, x1,y2] (左上、右上、右下、左下)
                # 这样2个人脸就会返回：[[x1,y1,x2,y1,x2,y2,x1,y2], [x3,y3,x4,y3,x4,y4,x3,y4]]
                xyxy = [[[x1, y1, x2, y1, x2, y2, x1, y2]] for x1, y1, x2, y2 in xyxy]
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
    

    
