"""图片推理器，识别单张图片"""

import requests
import numpy as np
from io import BytesIO
from typing import Any
from PIL import Image

from ..model.base_model import BaseModel
from ..model.plate_model import PlateModel
from .utils import Config


class Reasoner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return  # 避免重复初始化
        self.config = Config()
        self._load_model()
        self._initialized = True

    def _load_model(self):
        self.model_0 = BaseModel(self.config.model_list[0]["model_path"])
        self.model_1 = BaseModel(self.config.model_list[1]["model_path"])
        self.model_2 = BaseModel(self.config.model_list[2]["model_path"])
        self.model_3 = BaseModel(self.config.model_list[3]["model_path"])
        self.model_4 = PlateModel(self.config.model_list[4]["model_path"], self.config.model_list[4]["ocr_model_path"])
        self.model_5 = BaseModel(self.config.model_list[5]["model_path"])

    async def infer_image(self, image: Any | np.ndarray, model_index: int, conf=0.5, verbose: bool = True,
                          post_msg=True):
        """现有模型的图片推理方案，post_msg为是否输出后处理信息"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            response = requests.get(image)
            image = Image.open(BytesIO(response.content))
        # image = Image.open(image)  # 本地测试
        width, height = image.size
        classes = self.config.model_list[model_index].get("classes", None)
        if model_index == 0:
            model = self.model_0
        elif model_index == 1:
            model = self.model_1
        elif model_index == 2:
            model = self.model_2
        elif model_index == 3:
            model = self.model_3
        elif model_index == 4:
            model = self.model_4
        elif model_index == 5:
            model = self.model_5
        else:
            raise ValueError(f"Invalid model index: {model_index}")

        model_conf = self.config.model_list[model_index].get("conf", None)
        imgsz_type = self.config.model_list[model_index].get("imgsz", 0)
        imgsz = (640, 640) if imgsz_type == 0 else (height, width)

        if model_conf:
            yolo_result = model.detect_image(image, conf=model_conf, classes=classes, imgsz=imgsz, verbose=verbose)
        else:
            yolo_result = model.detect_image(image, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose)
        count = len(yolo_result[0])
        print(f"[识别到推理结果]：{count}")

        if not post_msg:
            return yolo_result
        else:
            if model_index == 5:
                infer_msg = model.post_process_hbb_to_obb(yolo_result)
            else:
                infer_msg = model.post_process(yolo_result)
            return infer_msg


reasoner_single = Reasoner()
