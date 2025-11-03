"""
模型推理类,基于yolo
"""

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
    

    
