"""物体追踪,基于yolo"""

from .base_model import BaseModel
from ultralytics.engine.results import Results

class TrackFireland(BaseModel):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.track_id = 0
        self.track_dict = {}
    
    def post_process(self,results:Results,pixel_position:list=None)->list:
        """判断识别结果是否在像素范围内。仅保留像素范围内的结果"""
        results_dict = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                xywh = boxes.xywh.tolist()
                try:
                    track_ids = boxes.id.tolist()
                except:
                    track_ids = "unknown"
                cls = boxes.cls.tolist()
                conf = boxes.conf.tolist()
                names = result.names

                for i, box in enumerate(xywh):
                    # 计算边界框中心点
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    
                    # 检查中心点是否在多边形内
                    if pixel_position is None or self._is_point_in_polygon((center_x, center_y), pixel_position):
                        box_params = {
                            "x": box[0],
                            "y": box[1],
                            "width": box[2],
                            "height": box[3],
                            "score": conf[i],
                            "track_id": track_ids[i] if isinstance(track_ids, list) else track_ids,
                            "classed": cls[i],
                            "className": names[cls[i]],
                            "text": ""
                        }
                        
                        # 添加追踪ID
                        if track_ids is not None:
                            box_params['track_id'] = int(track_ids[i])
                        
                        results_dict.append(box_params)
        
        return results_dict

    @staticmethod
    def _is_point_in_polygon(point, polygon):
        """
        判断点是否在多边形内部
        Args:
            point: (x, y) 坐标点
            polygon: 多边形顶点列表
        Returns:
            bool: 点是否在多边形内部
        """
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
