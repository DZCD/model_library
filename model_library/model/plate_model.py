
from ultralytics.engine.results import Results

from .base_model import BaseModel, device
from .ocr_provider import create_ocr_provider
from model_library.tools.logger import log_task_error
from model_library.utils.sahi_detector import SAHIPlateDetector
from model_library.tools.utils import Config


class PlateModel(BaseModel):
    def __init__(self, model_path, ocr_model_path=None, enable_sahi=False, sahi_config=None, ocr_provider="local"):
        super().__init__(model_path)
        self.config = Config()
        self.ocr_provider_name = ocr_provider or "local"
        modelscope_conf = self.config.config.get("modelscope", {})

        try:
            self.ocr_provider = create_ocr_provider(
                self.ocr_provider_name,
                ocr_model_path=ocr_model_path,
                modelscope_conf=modelscope_conf,
            )
        except Exception as exc:
            log_task_error(f"初始化 OCR Provider 失败: {exc}")
            raise

        print(f"使用 {self.ocr_provider_name} OCR 识别")

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
            if result is None:
                continue

            # 检查是否有检测结果
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue

            if len(result) == 0:
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

        height, width = ori_image_arr.shape[:2]

        for i, boxes in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = boxes
            x1_int = int(max(0, min(width - 1, x1)))
            y1_int = int(max(0, min(height - 1, y1)))
            x2_int = int(max(0, min(width - 1, x2)))
            y2_int = int(max(0, min(height - 1, y2)))

            if x2_int <= x1_int or y2_int <= y1_int:
                continue

            roi_img = ori_image_arr[y1_int:y2_int, x1_int:x2_int].copy()
            is_double = int(cls[i]) == 1

            plate_text, extra = self.ocr_provider.recognize(roi_img, is_double_layer=is_double)
            plate_text = (plate_text or "").strip()

            hbb_points = [boxes[0], boxes[1], boxes[2], boxes[1], boxes[2], boxes[3], boxes[0], boxes[3]]

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
                "text": plate_text,
                "plate_status": True if plate_text and len(plate_text) in [7, 8] else False,
            }

            if extra:
                box_params["ocr_extra"] = extra

            result_data.append(box_params)

        return result_data
