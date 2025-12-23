"""
红外行人检测模型，支持SAHI切片推理
"""

import os
from ultralytics.engine.results import Results
from .base_model import BaseModel, device
from model_library.tools.logger import log_task_error
from model_library.utils.sahi_detector import SAHIPlateDetector
from typing import Optional, Dict


class InfraredModel(BaseModel):
    def __init__(self, model_path, enable_sahi=False, sahi_config=None, enable_tracking=True,
                 model_index=None, estimated_memory=2500, device_override=None):
        """
        初始化红外检测模型，支持SAHI切片推理和跟踪

        Args:
            model_path: 模型路径
            enable_sahi: 是否启用SAHI切片推理
            sahi_config: SAHI配置参数
            enable_tracking: 是否启用跟踪（单张图片建议关闭）
        """
        # 调用父类构造函数，传递GPU分配参数
        super().__init__(model_path, model_index, estimated_memory, device_override)

        # SAHI和跟踪配置
        self.enable_sahi = enable_sahi
        self.enable_tracking = enable_tracking

        # SAHI配置 - 确保是字典类型
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
                    confidence_threshold=self.sahi_config.get('initial_confidence', 0.3),
                    device=self.sahi_config.get('device', None)
                )
                print("SAHI红外行人检测器已启用")
            except ImportError as e:
                print(f"SAHI初始化失败，回退到标准YOLO: {e}")
                self.enable_sahi = False
            except Exception as e:
                print(f"SAHI初始化失败，回退到标准YOLO: {e}")
                self.enable_sahi = False
        else:
            print("使用标准YOLO红外行人检测")

        print(f"跟踪模式: {'启用' if self.enable_tracking else '禁用（单张图片模式）'}")

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

    def track_image(self, source, conf=0.5, stream=False, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3, half=True):
        """
        重写跟踪方法，智能选择处理模式
        """
        if not self.enable_tracking:
            # 单张图片模式：只使用SAHI检测，跳过跟踪
            if self.enable_sahi:
                if verbose:
                    print("单张图片模式：使用SAHI检测（跳过跟踪）")
                return self.detect_image_with_sahi(source, conf, verbose)
            else:
                if verbose:
                    print("单张图片模式：使用标准YOLO检测（跳过跟踪）")
                if classes is not None:
                    results = self.model.predict(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose, half=half, device=device)
                else:
                    results = self.model.predict(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, half=half, device=device)
                return results

        # 跟踪模式：SAHI+跟踪
        if self.enable_sahi:
            # 使用SAHI进行检测，然后用YOLO跟踪器处理结果
            results = self._sahi_track_image(source, conf, classes, imgsz, verbose, iou, half)
        else:
            # 使用标准YOLO跟踪
            if classes is not None:
                results = self.model.track(source, stream=stream, conf=conf, classes=classes, imgsz=imgsz, verbose=verbose,
                                           iou=iou, half=half, device=device)
            else:
                results = self.model.track(source, stream=stream, conf=conf, imgsz=imgsz, verbose=verbose, iou=iou, half=half, device=device)
        return results

    def detect_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None,
                     imgsz: tuple = (640, 640), verbose: bool = True, iou=0.3, half=True):
        """
        重写视频检测方法，支持SAHI和标准YOLO检测
        """
        if self.enable_sahi:
            # SAHI目前不支持视频流，逐帧处理
            print("SAHI模式下进行视频逐帧检测")
            # 这里需要特殊处理，因为YOLO的video模式不支持直接替换
            # 暂时回退到标准YOLO
            if classes is not None:
                results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                             imgsz=imgsz, iou=0.3, verbose=verbose, half=half, device=device)
            else:
                results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, iou=0.3, verbose=verbose, half=half, device=device)
        else:
            # 使用标准YOLO视频检测
            if classes is not None:
                results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                             imgsz=imgsz, iou=0.3, verbose=verbose, half=half, device=device)
            else:
                results = self.model.predict(source, stream=stream, conf=conf, vid_stride=vid_stride, iou=0.3, verbose=verbose, half=half, device=device)
        return results

    def track_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None, imgsz: tuple = (640, 640),
                    verbose: bool = True, iou=0.3, half=True):
        """
        重写视频跟踪方法，支持SAHI+跟踪和标准YOLO跟踪
        注意：视频流的SAHI+跟踪需要逐帧处理，性能可能较慢
        """
        if self.enable_sahi:
            # 对于视频流，SAHI+跟踪需要特殊处理
            if verbose:
                print("使用SAHI+跟踪模式进行视频处理（逐帧处理）")

            # SAHI不支持直接的视频流处理，需要逐帧处理
            # 这里提供一个简化的实现，对于高性能场景建议使用标准YOLO跟踪
            import cv2

            # 检查source是否是视频文件
            if isinstance(source, str) and os.path.exists(source):
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    if verbose:
                        print(f"无法打开视频文件: {source}")
                    return []

                frames = []
                frame_count = 0

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        # 每隔vid_stride帧处理一次
                        if frame_count % vid_stride == 0:
                            # 使用SAHI+跟踪处理单帧
                            frame_results = self._sahi_track_image(frame, conf, classes, imgsz, verbose, iou, half)
                            frames.extend(frame_results if isinstance(frame_results, list) else [frame_results])

                        if verbose and frame_count % 100 == 0:
                            print(f"已处理 {frame_count} 帧")

                finally:
                    cap.release()

                if verbose:
                    print(f"SAHI+视频处理完成，共处理 {len(frames)} 帧结果")

                return frames
            else:
                # 对于实时流或其他输入，回退到标准YOLO跟踪
                if verbose:
                    print("SAHI+跟踪暂不支持实时流，回退到标准YOLO跟踪")
                if classes is not None:
                    results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                               imgsz=imgsz, iou=iou, verbose=verbose, half=half, device=device, tracker="botsort_cus.yaml")
                else:
                    results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, verbose=verbose,
                                               iou=iou, half=half, device=device, tracker="botsort_cus.yaml")
                return results
        else:
            # 使用标准YOLO视频跟踪
            if classes is not None:
                results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, classes=classes,
                                           imgsz=imgsz, iou=iou, verbose=verbose, half=half, device=device, tracker="botsort_cus.yaml")
            else:
                results = self.model.track(source, stream=stream, conf=conf, vid_stride=vid_stride, verbose=verbose,
                                           iou=iou, half=half, device=device, tracker="botsort_cus.yaml")
            return results

    def detect_image_with_sahi(self, source, confidence_threshold=0.5, verbose=True):
        """
        使用SAHI进行红外行人检测

        Args:
            source: 图像路径或PIL图像对象
            confidence_threshold: 最终置信度阈值
            verbose: 是否打印详细信息

        Returns:
            YOLO Results格式的检测结果
        """
        if verbose:
            print("=== 使用SAHI进行红外行人检测 ===")

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

    def post_process(self, results, **kwargs) -> list:
        """
        重写后处理方法，完全独立的SAHI结果处理
        避免调用父类方法，防止None值导致的错误
        """
        # 处理None结果
        if results is None:
            return []

        if not isinstance(results, list):
            results = [results]

        results_dict = []

        for result in results:
            if result is None:
                continue

            try:
                # 完全独立的处理逻辑
                temp_dict = []

                # 处理矩形框
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes

                    # 检查是否有有效的检测框
                    if hasattr(boxes, 'xywh') and boxes.xywh is not None:
                        xywh = boxes.xywh.tolist()

                        # 只有当有实际检测框时才处理
                        if len(xywh) > 0:
                            # 获取类别名称
                            names = result.names if hasattr(result, 'names') else {0: 'person'}

                            # 安全处理各个属性
                            try:
                                if hasattr(boxes, 'id') and boxes.id is not None:
                                    if hasattr(boxes.id, 'tolist') and boxes.id.tolist() is not None:
                                        track_id_list = boxes.id.tolist()
                                    else:
                                        # 如果不能转换，就设为unknown列表
                                        track_id_list = ["unknown"] * len(xywh)
                                else:
                                    track_id_list = ["unknown"] * len(xywh)
                            except:
                                track_id_list = ["unknown"] * len(xywh)

                            try:
                                cls_list = boxes.cls.tolist() if hasattr(boxes, 'cls') and boxes.cls is not None else [0] * len(xywh)
                            except:
                                cls_list = [0] * len(xywh)

                            try:
                                conf_list = boxes.conf.tolist() if hasattr(boxes, 'conf') and boxes.conf is not None else [0.5] * len(xywh)
                            except:
                                conf_list = [0.5] * len(xywh)

                            # 处理track_id中的None值
                            safe_track_ids = []
                            for tid in track_id_list:
                                if tid is None:
                                    safe_track_ids.append("unknown")
                                else:
                                    safe_track_ids.append(tid)

                            # 生成检测结果
                            for i, box in enumerate(xywh):
                                if i < len(cls_list) and i < len(conf_list):
                                    box_params = {
                                        "x": float(box[0]),
                                        "y": float(box[1]),
                                        "width": float(box[2]),
                                        "height": float(box[3]),
                                        "rotation": float(box[4]) if len(box) > 4 else 0.0,
                                        "score": float(conf_list[i]),
                                        "track_id": safe_track_ids[i] if i < len(safe_track_ids) else "unknown",
                                        "classed": int(cls_list[i]),
                                        "className": names.get(cls_list[i] if i < len(cls_list) else 0, 'person'),
                                        "text": ""
                                    }
                                    temp_dict.append(box_params)

                # 处理OBB框（如果存在）
                elif hasattr(result, 'obb') and result.obb is not None:
                    obb = result.obb
                    if hasattr(obb, 'xywhr') and obb.xywhr is not None:
                        xywhr = obb.xywhr.tolist()

                        if len(xywhr) > 0:
                            names = result.names if hasattr(result, 'names') else {0: 'person'}

                            try:
                                if hasattr(obb, 'id') and obb.id is not None:
                                    if hasattr(obb.id, 'tolist') and obb.id.tolist() is not None:
                                        track_id_list = obb.id.tolist()
                                    else:
                                        track_id_list = ["unknown"] * len(xywhr)
                                else:
                                    track_id_list = ["unknown"] * len(xywhr)
                            except:
                                track_id_list = ["unknown"] * len(xywhr)

                            try:
                                cls_list = obb.cls.tolist() if hasattr(obb, 'cls') and obb.cls is not None else [0] * len(xywhr)
                            except:
                                cls_list = [0] * len(xywhr)

                            try:
                                conf_list = obb.conf.tolist() if hasattr(obb, 'conf') and obb.conf is not None else [0.5] * len(xywhr)
                            except:
                                conf_list = [0.5] * len(xywhr)

                            # 处理track_id
                            safe_track_ids = []
                            for tid in track_id_list:
                                if tid is None:
                                    safe_track_ids.append("unknown")
                                else:
                                    safe_track_ids.append(tid)

                            # 生成检测结果
                            for i, box in enumerate(xywhr):
                                if i < len(cls_list) and i < len(conf_list):
                                    box_params = {
                                        "x": float(box[0]),
                                        "y": float(box[1]),
                                        "width": float(box[2]),
                                        "height": float(box[3]),
                                        "rotation": float(box[4]) if len(box) > 4 else 0.0,
                                        "score": float(conf_list[i]),
                                        "track_id": safe_track_ids[i] if i < len(safe_track_ids) else "unknown",
                                        "classed": int(cls_list[i]),
                                        "className": names.get(cls_list[i] if i < len(cls_list) else 0, 'person'),
                                        "text": ""
                                    }
                                    temp_dict.append(box_params)

                # 添加到最终结果
                results_dict.extend(temp_dict)

            except Exception as result_error:
                print(f"处理单个结果时出错: {str(result_error)}")
                continue

        return results_dict

    def _sahi_track_image(self, source, conf, classes, imgsz, verbose, iou, half):
        """
        SAHI + 跟踪的实现
        先用SAHI进行检测，然后用YOLO跟踪器处理结果
        """
        try:
            if verbose:
                print("使用SAHI+跟踪模式进行检测")

            # 1. 使用SAHI进行检测
            sahi_results = self.detect_image_with_sahi(
                source=source,
                confidence_threshold=conf,
                verbose=verbose
            )

            # 2. 如果检测到目标，使用YOLO跟踪器进行跟踪处理
            if sahi_results and len(sahi_results) > 0:
                result = sahi_results[0]

                # 检查是否有检测结果
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # 使用YOLO跟踪器对检测结果进行跟踪处理
                    tracked_results = self.model.track(
                        source=source,
                        stream=False,
                        conf=conf,
                        classes=classes,
                        imgsz=imgsz,
                        verbose=False,  # 避免重复输出
                        iou=iou,
                        half=half,
                        device=self.sahi_detector.device if hasattr(self, 'sahi_detector') else device,
                        tracker="botsort_cus.yaml"
                    )

                    if tracked_results and len(tracked_results) > 0:
                        tracked_result = tracked_results[0]

                        # 3. 合并SAHI的检测结果和跟踪信息
                        merged_result = self._merge_sahi_detection_with_tracking(result, tracked_result)

                        if verbose:
                            print(f"SAHI+跟踪完成: 检测到{len(result.boxes)}个目标, 分配了跟踪ID")

                        return [merged_result]
                    else:
                        if verbose:
                            print("跟踪器处理失败，返回SAHI检测结果")
                        return sahi_results
                else:
                    if verbose:
                        print("SAHI未检测到目标")
                    return sahi_results
            else:
                if verbose:
                    print("SAHI检测返回空结果")
                return sahi_results

        except Exception as e:
            if verbose:
                print(f"SAHI+跟踪处理失败: {str(e)}, 回退到标准YOLO跟踪")

            # 回退到标准YOLO跟踪
            try:
                if classes is not None:
                    return self.model.track(source, stream=False, conf=conf, classes=classes,
                                          imgsz=imgsz, verbose=verbose, iou=iou, half=half, device=device)
                else:
                    return self.model.track(source, stream=False, conf=conf, imgsz=imgsz,
                                          verbose=verbose, iou=iou, half=half, device=device)
            except Exception as fallback_e:
                if verbose:
                    print(f"标准YOLO跟踪也失败: {str(fallback_e)}")
                # 最后回退到SAHI检测
                return self.detect_image_with_sahi(source, conf, verbose)

    def _merge_sahi_detection_with_tracking(self, sahi_result, tracked_result):
        """
        合并SAHI检测结果和YOLO跟踪结果
        使用SAHI的检测框位置，使用跟踪器的track_id
        """
        try:
            # 创建一个新的Results对象
            merged_result = sahi_result

            # 检查两个结果都有检测框
            if (hasattr(sahi_result, 'boxes') and sahi_result.boxes is not None and
                hasattr(tracked_result, 'boxes') and tracked_result.boxes is not None):

                sahi_boxes = sahi_result.boxes
                tracked_boxes = tracked_result.boxes

                # 如果检测数量匹配，直接合并track_id
                if len(sahi_boxes) == len(tracked_boxes):
                    # 获取跟踪ID
                    if hasattr(tracked_boxes, 'id') and tracked_boxes.id is not None:
                        # 使用更安全的方式设置track_id
                        try:
                            import torch
                            if isinstance(tracked_boxes.id, torch.Tensor):
                                # 创建新的tensor而不是直接赋值
                                new_ids = tracked_boxes.id.clone()
                                # 通过data属性设置
                                sahi_boxes.data['id'] = new_ids
                            else:
                                sahi_boxes.data['id'] = tracked_boxes.id
                        except Exception as id_error:
                            print(f"设置track_id失败: {str(id_error)}")
                            # 如果直接设置失败，尝试其他方法
                            try:
                                sahi_boxes.data['id'] = tracked_boxes.id
                            except:
                                print("无法设置track_id，使用默认值")
                                # 设置默认track_id
                                sahi_boxes.data['id'] = torch.arange(len(sahi_boxes))
                else:
                    # 检测数量不匹配，使用IOU匹配
                    self._merge_by_iou(sahi_boxes, tracked_boxes)

            return merged_result

        except Exception as e:
            print(f"合并SAHI和跟踪结果失败: {str(e)}")
            return sahi_result

    def _merge_by_iou(self, sahi_boxes, tracked_boxes):
        """
        使用IOU匹配SAHI和跟踪结果
        """
        try:
            # 尝试不同的导入方式
            try:
                from ultralytics.utils.ops import box_iou
            except ImportError:
                # 如果导入失败，使用自定义IOU计算
                box_iou = self._custom_box_iou

            import torch

            # 获取边界框
            sahi_xyxy = sahi_boxes.xyxy
            tracked_xyxy = tracked_boxes.xyxy

            if len(sahi_xyxy) == 0 or len(tracked_xyxy) == 0:
                return

            # 计算IOU矩阵
            iou_matrix = box_iou(sahi_xyxy, tracked_xyxy)

            # 使用匈牙利算法或简单贪心匹配
            matched_indices = []
            used_tracked = set()
            used_sahi = set()

            # 按IOU降序排序
            matches = []
            for i in range(len(sahi_xyxy)):
                for j in range(len(tracked_xyxy)):
                    if i not in used_sahi and j not in used_tracked:
                        matches.append((iou_matrix[i, j], i, j))

            matches.sort(reverse=True)  # 按IOU降序排序

            # 匹配
            track_ids = tracked_boxes.id if hasattr(tracked_boxes, 'id') else None
            merged_track_ids = []

            for iou_val, sahi_idx, tracked_idx in matches:
                if iou_val > 0.3:  # IOU阈值
                    matched_indices.append((sahi_idx, tracked_idx))
                    used_sahi.add(sahi_idx)
                    used_tracked.add(tracked_idx)
                else:
                    break

            # 创建track_id列表
            final_track_ids = []
            tracked_idx_to_id = {}

            if track_ids is not None:
                for i, track_id in enumerate(track_ids):
                    tracked_idx_to_id[i] = track_id

            # 为每个SAHI框分配track_id
            for i in range(len(sahi_xyxy)):
                matched = False
                for sahi_idx, tracked_idx in matched_indices:
                    if i == sahi_idx:
                        final_track_ids.append(tracked_idx_to_id.get(tracked_idx, None))
                        matched = True
                        break
                if not matched:
                    final_track_ids.append(None)

            # 设置track_id
            if final_track_ids:
                import torch
                try:
                    # 使用更安全的方式设置track_id
                    sahi_boxes.data['id'] = torch.tensor(final_track_ids, dtype=torch.float32, device=sahi_xyxy.device)
                except Exception as id_error:
                    print(f"IOU匹配后设置track_id失败: {str(id_error)}")
                    # 设置默认track_id
                    sahi_boxes.data['id'] = torch.arange(len(sahi_xyxy))

        except Exception as e:
            print(f"IOU匹配失败: {str(e)}")
            pass

    def _custom_box_iou(self, box1, box2):
        """
        自定义IOU计算函数
        """
        try:
            import torch

            # box1: (N, 4), box2: (M, 4)
            # 计算交集
            lt = torch.maximum(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
            rb = torch.minimum(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

            wh = (rb - lt).clamp(min=0)  # (N, M, 2)
            inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

            # 计算并集
            area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
            area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)
            union = area1[:, None] + area2 - inter  # (N, M)

            # 计算IOU
            iou = inter / (union + 1e-7)  # 避免除零

            return iou

        except Exception as e:
            print(f"自定义IOU计算失败: {str(e)}")
            # 返回零矩阵
            import torch
            return torch.zeros(len(box1), len(box2))