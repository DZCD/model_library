from .base_model import BaseModel
from ultralytics.engine.results import Results
from datetime import datetime



class TrackAccident(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.track_id = 0
        self.track_dict = {}

        # 分类阈值配置
        self.class_thresholds = {
            0: 0.4,  # 事故类别默认阈值
            1: 0.4   # 行人类别默认阈值
        }

    def set_class_thresholds(self, accident_threshold=None, pedestrian_threshold=None):
        """设置分类别的检测阈值"""
        if accident_threshold is not None:
            self.class_thresholds[0] = accident_threshold
        if pedestrian_threshold is not None:
            self.class_thresholds[1] = pedestrian_threshold

    
    def post_process(self,results:Results)->list:
        """OBB目标追踪来定位事故，支持分类别阈值过滤"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue
            obb = result.obb
            if not obb:
                continue
            names = result.names
            xywhr = obb.xywhr.tolist()
            cls = obb.cls.tolist()
            conf = obb.conf.tolist()
            try:
                track_id = obb.id.tolist()
            except:
                # track_id =  f"{datetime.now().timestamp()}"
                track_id =  f"unknown"

            for i,box in enumerate(xywhr):
                class_id = int(cls[i])
                confidence = conf[i]

                # 应用分类别阈值过滤
                threshold = self.class_thresholds.get(class_id, 0.5)
                if confidence < threshold:
                    continue  # 跳过低于阈值的结果

                box_params = {
                    "x":box[0],
                    "y":box[1],
                    "width":box[2],
                    "height":box[3],
                    "rotation":box[4],
                    "score":confidence,
                    "track_id":track_id[i] if isinstance(track_id, list) else track_id,
                    "classed":class_id,
                    "className":names[class_id],
                    "text":""
                }
                results_dict.append(box_params)
        return results_dict

    def post_process_accidents_only(self,results:Results)->list:
        """只返回事故类别的检测结果，用于绘制验证后的真实事故框"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue
            obb = result.obb
            if not obb:
                continue
            names = result.names
            xywhr = obb.xywhr.tolist()
            cls = obb.cls.tolist()
            conf = obb.conf.tolist()
            try:
                track_id = obb.id.tolist()
            except:
                track_id =  f"unknown"

            for i,box in enumerate(xywhr):
                class_id = int(cls[i])
                confidence = conf[i]

                # 只处理事故类别 (class_id=0)
                if class_id != 0:
                    continue  # 跳过非事故类别

                # 应用事故类别阈值过滤
                threshold = self.class_thresholds.get(class_id, 0.5)
                if confidence < threshold:
                    continue  # 跳过低于阈值的结果

                box_params = {
                    "x":box[0],
                    "y":box[1],
                    "width":box[2],
                    "height":box[3],
                    "rotation":box[4],
                    "score":confidence,
                    "track_id":track_id[i] if isinstance(track_id, list) else track_id,
                    "classed":class_id,
                    "className":names[class_id],
                    "text":""
                }
                results_dict.append(box_params)
        return results_dict