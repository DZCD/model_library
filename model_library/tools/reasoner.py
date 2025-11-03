"""图片推理器，识别单张图片"""

import math
import requests
import numpy as np
from io import BytesIO
from typing import Any
from PIL import Image
from shapely.geometry import Polygon

from ..model.base_model import BaseModel
from ..model.plate_model import PlateModel
from ..model.car_model import CarModel
from .utils import Config
from .logger import log_task, log_task_error, log_task_debug


class Reasoner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return  # 避免重复初始化
        log_task_debug("初始化图像推理器 - 开始加载所有模型")
        self.config = Config()
        self._load_model()
        self._initialized = True
        log_task("图像推理器初始化完成 - 所有模型加载成功")

    def _load_model(self):
        try:
            self.model_0 = BaseModel(self.config.model_list[0]["model_path"])
            self.model_1 = BaseModel(self.config.model_list[1]["model_path"])
            self.model_2 = BaseModel(self.config.model_list[2]["model_path"])
            self.model_3 = BaseModel(self.config.model_list[3]["model_path"])
            self.model_4 = PlateModel(self.config.model_list[4]["model_path"],
                                      self.config.model_list[4]["ocr_model_path"])
            # self.model_5 = BaseModel(self.config.model_list[5]["model_path"])
            self.model_5 = CarModel(self.config.model_list[5]["model_path"])
            self.model_6 = BaseModel(self.config.model_list[6]["model_path"])
            self.model_7 = BaseModel(self.config.model_list[7]["model_path"])

        except Exception as e:
            log_task_error(f"模型加载失败 - 错误:{str(e)}")
            raise

    @staticmethod
    def _intersection_judgment(box1, box_list, threshold=0.2):
        """
        输入一个yolo的xywhr格式边界框以及一个8点格式边界框列表，
        判断后者有哪些与前者相交，相交面积占box2面积的比例超过阈值则判断为事故车辆
        返回列表中的索引。
        
        Args:
            box1: 单个边界框坐标 [x, y, width, height, rotation] (xywhr格式，事故区域)
            box_list: 边界框列表，每个元素为 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (8点格式，车辆列表)
            threshold: 相交面积占box2面积的比例阈值，默认0.2
            
        Returns:
            list: 相交的边界框在列表中的索引（事故车辆索引）
        """
        # 将box1从xywhr转换为多边形
        x, y, w, h, angle = box1
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
        box1_vertices = [(cx * cos_a - cy * sin_a + x, cx * sin_a + cy * cos_a + y) 
                        for cx, cy in corners]
        poly1 = Polygon(box1_vertices)
        
        intersecting_indices = []
        for i, box2 in enumerate(box_list):
            poly2 = Polygon(box2)
            intersection = poly1.intersection(poly2)
            
            if intersection.area > 0:
                # 计算相交面积占box2面积的比例
                overlap_ratio = intersection.area / poly2.area
                if overlap_ratio >= threshold:
                    intersecting_indices.append(i)
        
        return intersecting_indices

    async def infer_image(self, image: Any | np.ndarray, model_index: int, conf=0.5, verbose: bool = True,
                          post_msg=True):
        """现有模型的图片推理方案，post_msg为是否输出后处理信息"""
        import time
        start_time = time.time()
        
        try:
            # 确定输入类型和来源
            if isinstance(image, np.ndarray):
                log_task_debug(f"图像推理 - 输入:numpy数组, 模型:{model_index}")
                image = Image.fromarray(image)
            else:
                log_task_debug(f"图像推理 - 输入:URL地址, 模型:{model_index}, 地址:{str(image)[:100]}...")
                response = requests.get(image)
                image = Image.open(BytesIO(response.content))
            
            # image = Image.open(image)  # 本地测试
            width, height = image.size
            log_task_debug(f"图像尺寸 - 模型:{model_index}, 宽:{width}, 高:{height}")
            
            classes = self.config.model_list[model_index].get("classes", None)
            
            # 选择模型
            if model_index == 0:
                model = self.model_0
                model_name = "电梯摩托车检测"
            elif model_index == 1:
                model = self.model_1
                model_name = "消防通道占用检测"
            elif model_index == 2:
                model = self.model_2
                model_name = "火点检测"
            elif model_index == 3:
                model = self.model_3
                model_name = "事故检测"
            elif model_index == 4:
                model = self.model_4
                model_name = "车牌识别"
            elif model_index == 5:
                model = self.model_5
                model_name = "车辆检测"
            elif model_index == 6:
                model = self.model_6
                model_name = "红外行人检测"
            elif model_index == 7:
                model = self.model_7
                model_name = "人脸检测"
            else:
                log_task_error(f"无效的模型索引 - 模型:{model_index}")
                raise ValueError(f"Invalid model index: {model_index}")

            log_task_debug(f"开始推理 - 模型:{model_name}({model_index})")

            model_conf = self.config.model_list[model_index].get("conf", None)
            imgsz_type = self.config.model_list[model_index].get("imgsz", 0)
            imgsz = (640, 640) if imgsz_type == 0 else (height, width)

            if model_conf:
                yolo_result = model.detect_image(image, conf=model_conf, classes=classes, imgsz=imgsz, verbose=verbose)
            else:
                yolo_result = model.detect_image(image, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose)
            
            count = len(yolo_result[0])
            duration = time.time() - start_time
            log_task(f"图像推理完成 - 模型:{model_name}({model_index}), 检测到:{count}个目标, 耗时:{duration:.3f}秒")

            if not post_msg:
                return yolo_result
            else:
                if model_index == 3:  # 事故检测需要特殊处理
                    # 需要添加车辆识别逻辑
                    infer_msg = model.post_process(yolo_result)
                    
                    # 对每个事故区域进行车辆识别
                    for result in yolo_result:
                        if len(result) == 0:
                            continue
                        
                        ori_image = result.orig_img  # 获取原始图像
                        
                        # 使用车辆检测模型（model_index=5）识别车辆
                        car_result = await self.infer_image(ori_image, 5, post_msg=False)
                        
                        # 遍历每个检测到的事故区域
                        for i, item in enumerate(infer_msg):
                            accident_obb = [item['x'], item['y'], item['width'], item['height'], item['rotation']]
                            
                            if len(car_result[0]) == 0:  # 没有检测到车辆
                                item['accident_car_count'] = 0
                                item['accident_car_xyxy'] = []
                                continue
                            
                            car_result_obb = car_result[0].obb.xyxyxyxy.tolist()
                            
                            # 判断事故区域和车辆的交集
                            inter_index = self._intersection_judgment(accident_obb, car_result_obb)
                            accident_car = len(inter_index)
                            accident_obb_list = [car_result_obb[j] for j in inter_index]
                            accident_obb_list = [[coord for point in shape for coord in point] for shape in accident_obb_list]
                            
                            item['accident_car_count'] = accident_car
                            item['accident_car_xyxy'] = accident_obb_list
                    
                    log_task_debug(f"事故检测完成 - 模型:{model_index}, 结果数:{len(infer_msg) if infer_msg else 0}")
                    return infer_msg
                elif model_index == 5:
                    infer_msg = model.post_process(yolo_result)
                else:
                    infer_msg = model.post_process(yolo_result)
                log_task_debug(f"后处理完成 - 模型:{model_index}, 结果数:{len(infer_msg) if infer_msg else 0}")
                return infer_msg
                
        except Exception as e:
            duration = time.time() - start_time
            log_task_error(f"图像推理异常 - 模型:{model_index}, 错误:{str(e)}, 耗时:{duration:.3f}秒")
            raise


reasoner_single = Reasoner()