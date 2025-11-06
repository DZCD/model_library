
import torch
from ultralytics.engine.results import Results

from .base_model import BaseModel, device
from .ocr_model import get_split_merge, image_processing, decodePlate, color, plateName, init_model
from ..utils.sahi_detector import SAHIPlateDetector


class PlateModel(BaseModel):
    def __init__(self, model_path, ocr_model_path, enable_sahi=False, sahi_config=None):
        super().__init__(model_path)
        self.ocr_model = init_model(ocr_model_path, is_color=True)

        # SAHI配置 - 确保是字典类型
        self.enable_sahi = enable_sahi
        if sahi_config is None:
            self.sahi_config = {}
        elif isinstance(sahi_config, dict):
            self.sahi_config = sahi_config
        else:
            print(f"警告: sahi_config不是字典类型，收到: {type(sahi_config)}，使用默认配置")
            self.sahi_config = {}

        if self.enable_sahi:
            try:
                self.sahi_detector = SAHIPlateDetector(
                    model_path=model_path,
                    confidence_threshold=self.sahi_config.get('initial_confidence', 0.15),
                    device=self.sahi_config.get('device', None)
                )
                print("SAHI车牌检测器已启用")
            except ImportError as e:
                print(f"SAHI初始化失败，回退到标准YOLO: {e}")
                self.enable_sahi = False
            except Exception as e:
                print(f"SAHI初始化失败，回退到标准YOLO: {e}")
                self.enable_sahi = False
        else:
            print("使用标准YOLO车牌检测")

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                     verbose: bool = True, half=True):
        """
        重写检测方法，支持SAHI和标准YOLO检测
        """
        if self.enable_sahi:
            # 使用SAHI切片推理
            results = self.detect_image_with_sahi(
                source=source,
                confidence_threshold=conf,
                verbose=verbose
            )
        else:
            # 使用标准YOLO检测
            if classes is not None:
                results = self.model.predict(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz,
                                             verbose=verbose, half=half, device=device)
            else:
                results = self.model.predict(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, half=half, device=device)
        return results

    def detect_image_with_sahi(self, source, confidence_threshold=0.5, verbose=True):
        """
        使用SAHI进行车牌检测

        Args:
            source: 图像路径或PIL图像对象
            confidence_threshold: 最终置信度阈值
            verbose: 是否打印详细信息

        Returns:
            YOLO Results格式的检测结果
        """
        if verbose:
            print("=== 使用SAHI进行车牌检测 ===")

        try:
            # 处理不同类型的输入
            import tempfile
            import os
            from PIL import Image
            import numpy as np

            # 如果是numpy数组或PIL图像，需要保存为临时文件
            if isinstance(source, (np.ndarray, Image.Image)):
                if isinstance(source, np.ndarray):
                    image = Image.fromarray(source)
                else:
                    image = source

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    image.save(temp_path)

                # 使用SAHI检测器
                results = self.sahi_detector.detect_with_sahi(
                    image_path=temp_path,
                    final_confidence_threshold=confidence_threshold,
                    slice_params=self.sahi_config.get('slice_params', None)
                )

                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass

                return results
            else:
                # 假设是文件路径
                results = self.sahi_detector.detect_with_sahi(
                    image_path=source,
                    final_confidence_threshold=confidence_threshold,
                    slice_params=self.sahi_config.get('slice_params', None)
                )
                return results

        except Exception as e:
            print(f"SAHI检测失败，回退到标准YOLO: {e}")
            # 回退到标准YOLO
            return self.model.predict(
                source=source,
                conf=confidence_threshold,
                device=self.sahi_detector.device if hasattr(self, 'sahi_detector') else device,
                verbose=verbose
            )

    def post_process(self, results: Results) -> list:
        """提取Results中的推理结果数据，包括框的坐标、类别、置信度"""
        results_dict = []
        if not results:
            return results_dict

        for result in results:
            if result is None or len(result) == 0:
                continue
            ocr_data = self._plate_ocr_yolo(result)
            if not ocr_data:
                continue
            else:
                # ocr_data是一个列表，需要展开而不是嵌套
                results_dict.extend(ocr_data)
        return results_dict

    def _plate_ocr_yolo(self, yolo_result: Results):
        boxes_xyxy = yolo_result.boxes.xyxy.tolist()
        cls = yolo_result.boxes.cls.tolist()
        boxes_xywh = yolo_result.boxes.xywh.tolist()
        boxes_conf = yolo_result.boxes.conf.tolist()
        ori_image_arr = yolo_result.orig_img
        result_data = []
        for i, boxes in enumerate(boxes_xyxy):

            rect = [int(x) for x in boxes]
            # 先忽略双层的判断,影像前处理
            roi_img = ori_image_arr[rect[1]: rect[3], rect[0]: rect[2]]
            # 如果是双层要额外进行处理
            if int(cls[i]) == 1:
                roi_img = get_split_merge(roi_img)
            input = image_processing(roi_img)
            preds, color_preds = self.ocr_model(input)
            color_preds = torch.softmax(color_preds, dim=-1)
            color_conf, color_index = torch.max(color_preds, dim=-1)
            color_conf = color_conf.item()
            preds = torch.softmax(preds, dim=-1)
            prob, index = preds.max(dim=-1)
            index = index.view(-1).detach().cpu().numpy()
            prob = prob.view(-1).detach().cpu().numpy()

            # preds=preds.view(-1).detach().cpu().numpy()
            newPreds, new_index = decodePlate(index)
            prob = prob[new_index]
            plate = ""
            for str_i in newPreds:
                plate += plateName[str_i]
            x1, y1, x2, y2 = boxes
            hbb_points = [x1, y1, x2, y1, x2, y2, x1, y2]

            box_params = {
                "x": boxes_xywh[i][0],
                "y": boxes_xywh[i][1],
                "width": boxes_xywh[i][2],
                "height": boxes_xywh[i][3],
                "rotation": boxes_xywh[i][4] if len(boxes_xywh[i]) == 5 else 0,
                "score": boxes_conf[i],
                "xyxy": hbb_points,
                "track_id": "unknown",
                "classed": cls[i],
                "className": "license plate",
                "text": plate,
                "plate_status": True if len(plate) in [7, 8] else False,

            }
            result_data.append(box_params)
        # 将识别到的车牌位置以及字符串写入原图
        # plate_texts = [f"{c}:{p}" for c, p in zip(data["color"], data["plate"])]
        # plate_texts = [f"{p}" for p in ocr_data["plate"]]

        return result_data
