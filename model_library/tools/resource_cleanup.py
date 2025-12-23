"""
资源清理工具 - 确保任务完全销毁
"""

import time
import threading
import signal
import gc
from typing import Dict, Any, Optional
from ..tools.logger import log_task, log_task_error, log_task_debug
from ..tools.gpu_manager import gpu_manager


class ResourceCleanupManager:
    """资源清理管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.cleanup_tasks = {}  # task_id -> cleanup_info
        log_task("资源清理管理器初始化完成")

    def register_task(self, task_id: str, detector_instance: Any, model_index: int = None):
        """注册任务用于后续清理"""
        self.cleanup_tasks[task_id] = {
            'detector': detector_instance,
            'model_index': model_index,
            'start_time': time.time(),
            'cleanup_attempted': False
        }

    def cleanup_task(self, task_id: str, force: bool = False) -> bool:
        """
        清理指定任务的所有资源

        Args:
            task_id: 任务ID
            force: 是否强制清理

        Returns:
            清理是否成功
        """
        if task_id not in self.cleanup_tasks:
            log_task_error(f"任务 {task_id} 未注册，无法清理")
            return False

        cleanup_info = self.cleanup_tasks[task_id]
        detector = cleanup_info['detector']
        model_index = cleanup_info['model_index']

        success = True

        try:
            log_task(f"开始清理任务 {task_id}")

            # 1. 停止推理循环
            if hasattr(detector, 'request_stop'):
                detector.request_stop()
                log_task_debug(f"已发送停止信号 - 任务ID:{task_id}")

            # 2. 停止监控线程
            if hasattr(detector, '_stop_monitor_thread'):
                detector._stop_monitor_thread()
                log_task_debug(f"已停止监控线程 - 任务ID:{task_id}")

            # 3. 清理GPU资源
            if model_index is not None:
                self._cleanup_gpu_resources(model_index, task_id)

            # 4. 清理MQTT连接
            if hasattr(detector, 'mqtt_client'):
                try:
                    detector.mqtt_client.disconnect()
                    log_task_debug(f"已断开MQTT连接 - 任务ID:{task_id}")
                except Exception as e:
                    log_task_error(f"MQTT断开失败 - 任务ID:{task_id}, 错误:{str(e)}")

            # 5. 清理MinIO连接
            if hasattr(detector, 'minio_client'):
                try:
                    # MinIO客户端通常不需要主动断开
                    log_task_debug(f"MinIO资源清理完成 - 任务ID:{task_id}")
                except Exception as e:
                    log_task_error(f"MinIO清理失败 - 任务ID:{task_id}, 错误:{str(e)}")

            # 6. 强制内存回收
            if force:
                self._force_memory_cleanup(detector, task_id)

            # 7. 标记清理完成
            cleanup_info['cleanup_attempted'] = True
            cleanup_info['cleanup_time'] = time.time()

            log_task(f"任务 {task_id} 清理完成")
            return True

        except Exception as e:
            log_task_error(f"任务 {task_id} 清理失败: {str(e)}")
            if force:
                # 强制清理模式下的最后手段
                self._emergency_cleanup(task_id, detector)
            return False

    def _cleanup_gpu_resources(self, model_index: int, task_id: str):
        """清理GPU资源"""
        try:
            # 使用动态GPU管理器释放资源
            gpu_manager.release_model(model_index, task_id)
            log_task_debug(f"GPU资源已释放 - 任务ID:{task_id}, 模型:{model_index}")
        except Exception as e:
            log_task_error(f"GPU资源释放失败 - 任务ID:{task_id}, 模型:{model_index}, 错误:{str(e)}")

    def _force_memory_cleanup(self, detector: Any, task_id: str):
        """强制内存清理"""
        try:
            # 检查Python是否正在关闭
            import sys
            if sys.meta_path is None:
                log_task_debug(f"Python正在关闭，跳过强制内存清理 - 任务ID:{task_id}")
                return

            # 清理Detector实例
            if hasattr(detector, 'model'):
                detector.model = None
            if hasattr(detector, 'mqtt_client'):
                detector.mqtt_client = None
            if hasattr(detector, 'minio_client'):
                detector.minio_client = None
            if hasattr(detector, 'verification_manager'):
                detector.verification_manager = None
            if hasattr(detector, 'vlm_verifier'):
                detector.vlm_verifier = None

            # 强制垃圾回收
            import gc
            if gc.isenabled():
                collected = gc.collect()
                log_task_debug(f"垃圾回收完成，清理了 {collected} 个对象 - 任务ID:{task_id}")

            # 如果有CUDA，清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available() and not sys.meta_path is None:
                    torch.cuda.empty_cache()
                    log_task_debug(f"GPU缓存已清理 - 任务ID:{task_id}")
            except ImportError:
                pass
            except Exception as torch_error:
                log_task_debug(f"GPU缓存清理跳过 - 任务ID:{task_id}, 错误:{str(torch_error)}")

            log_task_debug(f"强制内存清理完成 - 任务ID:{task_id}")

        except Exception as e:
            # 在Python关闭时，某些操作可能会失败，这是正常的
            if "sys.meta_path is None" in str(e) or "Python is likely shutting down" in str(e):
                log_task_debug(f"Python正在关闭，清理操作被中断 - 任务ID:{task_id}")
            else:
                log_task_error(f"强制内存清理失败 - 任务ID:{task_id}, 错误:{str(e)}")

    def _emergency_cleanup(self, task_id: str, detector: Any):
        """紧急清理（最后手段）"""
        try:
            log_task_error(f"执行紧急清理 - 任务ID:{task_id}")

            # 设置所有属性为None
            for attr_name in dir(detector):
                if not attr_name.startswith('_'):
                    try:
                        setattr(detector, attr_name, None)
                    except:
                        pass

            # 强制垃圾回收
            import gc
            gc.collect()

            log_task(f"紧急清理完成 - 任务ID:{task_id}")

        except Exception as e:
            log_task_error(f"紧急清理失败 - 任务ID:{task_id}, 错误:{str(e)}")

    def cleanup_all_tasks(self) -> Dict[str, int]:
        """清理所有已注册的任务"""
        results = {
            'total': len(self.cleanup_tasks),
            'success': 0,
            'failed': 0
        }

        log_task(f"开始清理所有任务 (总数: {results['total']})")

        for task_id in list(self.cleanup_tasks.keys()):
            if self.cleanup_task(task_id, force=True):
                results['success'] += 1
            else:
                results['failed'] += 1

        # 清空注册表
        self.cleanup_tasks.clear()

        log_task(f"批量清理完成 - 成功:{results['success']}, 失败:{results['failed']}")
        return results

    def get_cleanup_status(self) -> Dict[str, Any]:
        """获取清理状态"""
        status = {
            'registered_tasks': len(self.cleanup_tasks),
            'cleanup_attempted': 0,
            'pending_cleanup': 0
        }

        for task_id, info in self.cleanup_tasks.items():
            if info.get('cleanup_attempted', False):
                status['cleanup_attempted'] += 1
            else:
                status['pending_cleanup'] += 1

        return status


# 创建全局实例
resource_cleanup_manager = ResourceCleanupManager()