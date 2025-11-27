"""
模型管理器 - 全局模型单例管理
避免重复加载模型，提升系统性能和内存使用效率
"""

import threading
from typing import Dict, Optional
from model_library.tools.logger import log_task_debug, log_task_error, log_task


class ModelManager:
    """全局模型单例管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[int, any] = {}  # 模型索引到模型实例的映射
        self._model_locks: Dict[int, threading.Lock] = {}  # 每个模型的加载锁
        self._initialized = True
        log_task("模型管理器初始化完成")

    def get_model(self, model_index: int, task_id: str = None):
        """
        获取模型实例，如果未加载则自动加载

        Args:
            model_index: 模型索引
            task_id: 任务ID（用于日志）

        Returns:
            模型实例
        """
        # 如果模型已存在，直接返回
        if model_index in self._models:
            log_task_debug(f"使用已加载的模型 - 任务ID:{task_id}, 模型索引:{model_index}")
            return self._models[model_index]

        # 为当前模型创建锁（如果不存在）
        if model_index not in self._model_locks:
            with self._lock:
                if model_index not in self._model_locks:
                    self._model_locks[model_index] = threading.Lock()

        # 使用模型特定的锁来加载模型（避免同一模型被多个线程同时加载）
        with self._model_locks[model_index]:
            # 双重检查，防止在等待锁期间模型已被其他线程加载
            if model_index in self._models:
                log_task_debug(f"使用已加载的模型（双重检查） - 任务ID:{task_id}, 模型索引:{model_index}")
                return self._models[model_index]

            # 加载新模型
            try:
                from .model_loader import ModelLoader
                loader = ModelLoader()
                model = loader.load_model(model_index, task_id=task_id)

                # 存储模型实例
                self._models[model_index] = model
                log_task(f"模型加载并缓存成功 - 任务ID:{task_id}, 模型索引:{model_index}")
                return model

            except Exception as e:
                log_task_error(f"模型加载失败 - 任务ID:{task_id}, 模型索引:{model_index}, 错误:{str(e)}")
                raise

    def clear_model(self, model_index: int):
        """
        清除指定模型的缓存

        Args:
            model_index: 模型索引
        """
        if model_index in self._models:
            with self._model_locks.get(model_index, threading.Lock()):
                if model_index in self._models:
                    del self._models[model_index]
                    log_task(f"模型缓存已清除 - 模型索引:{model_index}")

    def clear_all_models(self):
        """清除所有模型缓存"""
        with self._lock:
            for model_index in list(self._models.keys()):
                self.clear_model(model_index)
            log_task("所有模型缓存已清除")

    def get_loaded_models(self) -> Dict[int, str]:
        """
        获取已加载的模型信息

        Returns:
            已加载模型的字典 {model_index: model_name}
        """
        loaded_info = {}
        for model_index, model in self._models.items():
            try:
                model_name = getattr(model, 'model_name', f'model_{model_index}')
                loaded_info[model_index] = model_name
            except:
                loaded_info[model_index] = f'model_{model_index}'
        return loaded_info

    def get_model_count(self) -> int:
        """获取已加载的模型数量"""
        return len(self._models)


# 创建全局模型管理器实例
model_manager = ModelManager()