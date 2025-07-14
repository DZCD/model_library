from .base_model import BaseModel
from ultralytics.engine.results import Results


class TrackAccident(BaseModel):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.track_id = 0
        self.track_dict = {}

    
    def post_process(self,results:Results)->list:
        """OBB目标追踪来定位事故"""
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
                track_id = "unknown"

            for i,box in enumerate(xywhr):
                box_params = {
                    "x":box[0],
                    "y":box[1],
                    "width":box[2],
                    "height":box[3],
                    "rotation":box[4],
                    "score":conf[i],
                    "track_id":track_id[i] if isinstance(track_id, list) else track_id,
                    "classed":cls[i],
                    "className":names[cls[i]],
                    "text":""
                }
                results_dict.append(box_params)
        return results_dict
                