"""
模型推理类,基于yolo
"""

from ultralytics import YOLO
from ultralytics.engine.results import Results


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
