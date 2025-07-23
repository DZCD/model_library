from .base_model import BaseModel
from ultralytics.engine.results import Results
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


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

    def obb_intersect(self, accident_xywhr: list, vehicle_hbb_boxes: list) -> int:
        """
        计算事故现场(OBB)与车辆检测框(HBB)的交集数量
        
        Args:
            accident_xywhr: 事故现场的OBB参数 [center_x, center_y, width, height, rotation_radians]
            vehicle_hbb_boxes: 车辆检测框列表 [[center_x, center_y, width, height], [center_x, center_y, width, height], ...]
            
        Returns:
            int: 与事故现场相交的车辆数量
        """
        try:
            # 1. 将OBB转换为多边形
            center_x, center_y, width, height, rotation = accident_xywhr
            
            # 计算OBB的四个顶点
            cos_angle = np.cos(rotation)
            sin_angle = np.sin(rotation)
            
            # 相对于中心点的四个顶点坐标
            half_w, half_h = width / 2, height / 2
            corners = np.array([
                [-half_w, -half_h],  # 左下
                [half_w, -half_h],   # 右下
                [half_w, half_h],    # 右上
                [-half_w, half_h]    # 左上
            ])
            
            # 应用旋转矩阵
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])
            
            rotated_corners = corners @ rotation_matrix.T
            rotated_corners[:, 0] += center_x
            rotated_corners[:, 1] += center_y
            
            # 创建事故现场多边形
            accident_polygon = Polygon(rotated_corners)
            
            # 2. 计算相交的车辆数量
            intersect_count = 0
            
            for vehicle_box in vehicle_hbb_boxes:
                # 解析车辆HBB参数 [center_x, center_y, width, height]
                v_cx, v_cy, v_w, v_h = vehicle_box
                
                # HBB的四个顶点
                v_x1, v_y1 = v_cx - v_w/2, v_cy - v_h/2
                v_x2, v_y2 = v_cx + v_w/2, v_cy + v_h/2
                
                vehicle_polygon = Polygon([
                    (v_x1, v_y1), (v_x2, v_y1),
                    (v_x2, v_y2), (v_x1, v_y2)
                ])
                
                # 检查是否相交
                if accident_polygon.intersects(vehicle_polygon):
                    intersection = accident_polygon.intersection(vehicle_polygon)
                    intersection_area = intersection.area
                    vehicle_area = vehicle_polygon.area
                    
                    # 计算重叠率（车辆框被事故现场覆盖的比例）
                    overlap_ratio = intersection_area / vehicle_area if vehicle_area > 0 else 0
                    
                    # 只有重叠率超过阈值才算相交
                    if overlap_ratio > 0:  # 20%重叠阈值，可调整
                        intersect_count += 1
            
            return intersect_count
            
        except Exception as e:
            print(f"计算OBB与HBB相交数量时发生错误: {str(e)}")
            return 0            