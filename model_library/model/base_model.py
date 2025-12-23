"""
模型推理类,基于yolo
"""

import os
import sys

# 添加项目路径到sys.path，以便导入video_backend模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

from video_backend import create_video_capture, VideoBackend
from ..tools.gpu_manager import gpu_manager

from ultralytics import YOLO
from ultralytics.engine.results import Results
from datetime import datetime
import torch

# 动态设备分配：使用动态GPU管理器分配最优设备
try:
    device = 'cpu'  # 默认值，实际使用时会通过动态GPU管理器重新分配
    print("使用动态GPU管理器进行智能负载均衡分配")
except Exception as e:
    print(f"动态GPU管理器不可用，使用默认设备分配: {str(e)}")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class BaseModel:
    def __init__(self, model_path, model_index: int = None, estimated_memory: int = 1000, device_override: str = None):
        """
        初始化模型，支持GPU自动分配

        Args:
            model_path: 模型文件路径
            model_index: 模型索引，用于GPU分配
            estimated_memory: 预估显存需求(MB)
            device_override: 强制指定设备，覆盖自动分配
        """
        self.model_index = model_index
        self.model_path = model_path
        self.estimated_memory = estimated_memory

        # 确定设备分配策略
        if device_override:
            # 使用指定的设备
            self.device = device_override
            print(f"使用指定设备: {self.device}")
        elif model_index is not None:
            # 使用动态GPU管理器自动分配
            try:
                allocated_gpu = gpu_manager.deploy_model(model_index, estimated_memory)
                if allocated_gpu is not None:
                    self.device = f"cuda:{allocated_gpu}"
                    self.allocated_gpu_id = allocated_gpu
                    print(f"模型 {model_index} 智能分配到GPU {allocated_gpu} (动态负载均衡)")

                    # 打印当前GPU状态
                    gpu_stats = gpu_manager.get_deployment_stats()
                    total_deployments = gpu_stats["total_deployments"]
                    print(f"当前系统总部署数: {total_deployments}")
                else:
                    self.device = 'cpu'
                    print(f"模型 {model_index} GPU分配失败，使用CPU")
            except Exception as e:
                print(f"动态GPU分配异常，使用默认策略: {str(e)}")
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.allocated_gpu_id = 0 if self.device.startswith('cuda') else None
        else:
            # 使用默认设备
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"使用默认设备: {self.device}")

        # 加载模型到指定设备
        try:
            self.model = YOLO(model_path)
            # 预热模型到指定设备
            if self.device.startswith('cuda'):
                torch.cuda.set_device(int(self.device.split(':')[1]))
                print(f"模型已加载到设备: {self.device}")
            else:
                print(f"模型已加载到设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            # 如果GPU加载失败，回退到CPU
            self.device = 'cpu'
            self.model = YOLO(model_path)
            print(f"回退到CPU设备")

        # 更新全局device变量以保持兼容性
        globals()['device'] = self.device

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                     verbose: bool = True,half=True):
        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz,
                                         verbose=verbose, half=half, device=self.device)
        else:
            results = self.model.predict(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, half=half, device=self.device)
        return results

    def track_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3,half=True):
        if classes is not None:
            results = self.model.track(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose,
                                       iou=iou, half=half, device=self.device)
        else:
            results = self.model.track(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, iou=iou, half=half, device=self.device)
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

    def get_device_info(self) -> dict:
        """获取当前模型的设备信息"""
        return {
            "device": self.device,
            "model_index": self.model_index,
            "estimated_memory": self.estimated_memory,
            "model_path": self.model_path
        }

    def post_process(self, results: Results, **kwargs) -> list:
        """提取Results中的推理结果数据，包括框的坐标、类别、置信度"""
        results_dict = []
        for result in results:
            if result is None or len(result) == 0:
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
                # 正确的OBB格式转换：将矩形框转换为4个角点的8个坐标值
                # 格式：[x1,y1, x2,y1, x2,y2, x1,y2] (左上、右上、右下、左下)
                # 这样2个人脸就会返回：[[x1,y1,x2,y1,x2,y2,x1,y2], [x3,y3,x4,y3,x4,y4,x3,y4]]
                xyxy = [[[x1, y1, x2, y1, x2, y2, x1, y2]] for x1, y1, x2, y2 in xyxy]
                cls = boxes.cls.tolist()
                conf = boxes.conf.tolist()

            try:
                if boxes.id is not None:
                    track_id = boxes.id.tolist()
                else:
                    track_id = "unknown"
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

    def __del__(self):
        """析构函数，自动释放GPU资源"""
        try:
            self.release_resources()
        except:
            pass  # 析构函数中忽略异常

    def release_resources(self):
        """手动释放GPU资源"""
        if hasattr(self, 'model_index') and hasattr(self, 'allocated_gpu_id'):
            if self.model_index is not None:
                try:
                    from ..tools.gpu_manager import gpu_manager
                    gpu_manager.release_model(self.model_index)
                    print(f"模型 {self.model_index} 资源已释放")
                except Exception as e:
                    print(f"释放模型资源时出错: {str(e)}")

    def print_gpu_status(self):
        """打印当前GPU状态"""
        try:
            from ..tools.gpu_manager import gpu_manager
            gpu_manager.print_status()
        except Exception as e:
            print(f"获取GPU状态失败: {str(e)}")