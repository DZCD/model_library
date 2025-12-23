"""
动态GPU资源管理器 - 实现智能负载均衡和多实例部署
支持根据GPU负载自动复制模型到不同GPU
"""

import torch
import threading
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import psutil
from ..tools.logger import log_task, log_task_debug, log_task_error


@dataclass
class GPUInfo:
    """GPU信息数据类"""
    gpu_id: int
    device_name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: float  # 0-100
    temperature: float  # 摄氏度
    is_available: bool
    model_count: int = 0  # 已加载的模型数量
    current_tasks: int = 0  # 正在运行的任务数


@dataclass
class ModelDeployment:
    """模型部署信息"""
    model_index: int
    gpu_id: int
    load_time: float
    estimated_memory: int  # MB
    ref_count: int = 0  # 引用计数（多少个任务在使用）
    model_instance: object = None  # 实际模型实例
    deployment_id: str = None  # 部署唯一ID，支持同一模型多实例


class DynamicGPUManager:
    """动态GPU资源管理器"""

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

        self.gpu_list: List[GPUInfo] = []
        self.model_deployments: Dict[str, ModelDeployment] = {}  # deployment_id -> ModelDeployment (支持多实例)
        self.model_index_to_deployments: Dict[int, List[str]] = defaultdict(list)  # model_index -> [deployment_id_list]
        self._deployment_lock = threading.Lock()  # 部署锁，确保线程安全
        self.gpu_models: Dict[int, List[str]] = defaultdict(list)  # gpu_id -> [deployment_id_list]
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_lock = threading.Lock()

        # 动态负载均衡配置
        self.gpu_utilization_threshold = 75  # GPU利用率阈值(%)
        self.max_ref_count_per_instance = 3  # 单个实例最大引用数
        self.enable_dynamic_duplication = True  # 启用动态复制
        self.duplication_check_interval = 10  # 检查间隔(秒)

        # 初始化GPU信息
        self._initialize_gpu_info()
        self._initialized = True
        log_task("动态GPU管理器初始化完成")

    def _initialize_gpu_info(self):
        """初始化GPU信息"""
        if not torch.cuda.is_available():
            log_task_error("未检测到可用的CUDA设备")
            return

        try:
            gpu_count = torch.cuda.device_count()
            log_task(f"检测到 {gpu_count} 张GPU设备")

            for i in range(gpu_count):
                # 设置当前GPU以查询信息
                torch.cuda.set_device(i)

                # 获取GPU属性
                props = torch.cuda.get_device_properties(i)
                device_name = props.name

                # 获取内存信息
                memory_total = props.total_memory // (1024 * 1024)  # 转换为MB

                # 初始化GPU信息（动态更新）
                gpu_info = GPUInfo(
                    gpu_id=i,
                    device_name=device_name,
                    memory_total=memory_total,
                    memory_used=0,
                    memory_free=memory_total,
                    utilization=0.0,
                    temperature=0.0,
                    is_available=True
                )

                self.gpu_list.append(gpu_info)
                log_task(f"GPU {i}: {device_name}, 显存: {memory_total}MB")

            # 启动GPU监控线程
            self._start_monitoring()

        except Exception as e:
            log_task_error(f"GPU初始化失败: {str(e)}")

    def _start_monitoring(self):
        """启动GPU监控线程"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_gpu_status, daemon=True)
        self._monitor_thread.start()
        log_task("动态GPU监控线程已启动")

    def _monitor_gpu_status(self):
        """监控GPU状态"""
        try:
            import GPUtil

            while self._monitoring:
                try:
                    # 获取所有GPU的状态
                    gpus = GPUtil.getGPUs()

                    with self._monitor_lock:
                        for gpu in gpus:
                            if gpu.id < len(self.gpu_list):
                                gpu_info = self.gpu_list[gpu.id]

                                # 更新GPU信息
                                gpu_info.memory_used = int(gpu.memoryUsed)
                                gpu_info.memory_free = int(gpu.memoryFree)
                                gpu_info.utilization = gpu.load * 100
                                gpu_info.temperature = gpu.temperature

                                # 检查GPU可用性
                                gpu_info.is_available = (
                                    gpu_info.memory_free > 1024 and  # 至少1GB可用显存
                                    gpu_info.temperature < 85 and  # 温度低于85度
                                    gpu_info.utilization < 95      # 利用率低于95%
                                )

                except Exception as e:
                    log_task_error(f"GPU状态监控异常: {str(e)}")

                time.sleep(2)  # 每2秒更新一次

        except ImportError:
            log_task_error("GPUtil库未安装，无法监控GPU状态")
            # 回退到基本监控
            self._basic_gpu_monitoring()

        except Exception as e:
            log_task_error(f"GPU监控线程启动失败: {str(e)}")

    def _basic_gpu_monitoring(self):
        """基本GPU监控（不依赖GPUtil）"""
        while self._monitoring:
            try:
                with self._monitor_lock:
                    for i, gpu_info in enumerate(self.gpu_list):
                        if torch.cuda.is_available() and i < torch.cuda.device_count():
                            # 获取内存信息
                            memory_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                            memory_reserved = torch.cuda.memory_reserved(i) // (1024 * 1024)

                            gpu_info.memory_used = memory_allocated
                            gpu_info.memory_free = gpu_info.memory_total - memory_reserved
                            gpu_info.utilization = min(90, (memory_allocated / gpu_info.memory_total) * 100)
                            gpu_info.is_available = gpu_info.memory_free > 1024

            except Exception as e:
                log_task_error(f"基本GPU监控异常: {str(e)}")

            time.sleep(5)  # 每5秒更新一次

    def deploy_model(self, model_index: int, estimated_memory: int = 1000, model_instance=None) -> Optional[int]:
        """
        部署模型到GPU - 支持动态负载均衡和多实例

        Args:
            model_index: 模型索引
            estimated_memory: 预估需要的显存(MB)
            model_instance: 模型实例（可选）

        Returns:
            分配的GPU ID，失败返回None
        """
        with self._deployment_lock:
            # 检查是否需要动态复制
            if self.enable_dynamic_duplication and model_index in self.model_index_to_deployments:
                best_deployment_id, best_gpu_id = self._get_best_deployment_for_model(model_index)

                if best_deployment_id:
                    # 使用现有最优部署
                    deployment = self.model_deployments[best_deployment_id]
                    deployment.ref_count += 1
                    log_task(f"模型 {model_index} 复用GPU {best_gpu_id} 上的实例 {best_deployment_id}，引用计数: {deployment.ref_count}")
                    return best_gpu_id

            # 获取最优GPU进行新部署
            optimal_gpu = self.get_optimal_gpu(estimated_memory)

            if optimal_gpu is None:
                log_task_error(f"无法为模型 {model_index} 分配GPU")
                return None

            # 创建新的部署记录
            deployment_id = f"model_{model_index}_gpu_{optimal_gpu}_{int(time.time())}"
            deployment = ModelDeployment(
                model_index=model_index,
                gpu_id=optimal_gpu,
                load_time=time.time(),
                estimated_memory=estimated_memory,
                ref_count=1,
                model_instance=model_instance,
                deployment_id=deployment_id
            )

            self.model_deployments[deployment_id] = deployment
            self.model_index_to_deployments[model_index].append(deployment_id)
            self.gpu_models[optimal_gpu].append(deployment_id)

            # 更新GPU信息
            gpu_info = self.gpu_list[optimal_gpu]
            gpu_info.model_count += 1

            log_task(f"模型 {model_index} 新部署到GPU {optimal_gpu} (部署ID: {deployment_id}, 预估内存: {estimated_memory}MB)")
            return optimal_gpu

    def _get_best_deployment_for_model(self, model_index: int) -> Tuple[Optional[str], Optional[int]]:
        """
        为模型选择最优的现有部署

        Args:
            model_index: 模型索引

        Returns:
            (deployment_id, gpu_id) 或 (None, None)
        """
        if model_index not in self.model_index_to_deployments:
            return None, None

        deployment_ids = self.model_index_to_deployments[model_index]
        best_deployment_id = None
        best_score = float('inf')

        for deployment_id in deployment_ids:
            deployment = self.model_deployments[deployment_id]
            gpu_info = self.gpu_list[deployment.gpu_id]

            # 检查是否超过最大引用数
            if deployment.ref_count >= self.max_ref_count_per_instance:
                continue

            # 检查GPU利用率
            if gpu_info.utilization > self.gpu_utilization_threshold:
                continue

            # 计算得分（越低越好）
            score = (gpu_info.utilization * 0.7 + deployment.ref_count * 10)

            if score < best_score:
                best_score = score
                best_deployment_id = deployment_id

        if best_deployment_id:
            deployment = self.model_deployments[best_deployment_id]
            return best_deployment_id, deployment.gpu_id

        # 如果现有部署都不可用，检查是否需要动态复制
        return self._should_duplicate_model(model_index)

    def _should_duplicate_model(self, model_index: int) -> Tuple[Optional[str], Optional[int]]:
        """
        判断是否应该将模型复制到其他GPU

        Args:
            model_index: 模型索引

        Returns:
            (deployment_id, gpu_id) 或 (None, None)
        """
        if not self.enable_dynamic_duplication:
            return None, None

        # 检查现有部署的GPU负载
        deployment_ids = self.model_index_to_deployments.get(model_index, [])

        for deployment_id in deployment_ids:
            deployment = self.model_deployments[deployment_id]
            gpu_info = self.gpu_list[deployment.gpu_id]

            # 如果当前GPU负载过高，尝试复制到其他GPU
            if (gpu_info.utilization > self.gpu_utilization_threshold or
                deployment.ref_count >= self.max_ref_count_per_instance):

                # 寻找可用的其他GPU
                target_gpu = self._find_available_gpu_for_duplication(model_index)

                if target_gpu is not None:
                    # 标记需要复制，但不在这里执行（由调用方处理）
                    log_task(f"模型 {model_index} GPU {deployment.gpu_id} 负载过高({gpu_info.utilization}%)，建议复制到GPU {target_gpu}")
                    # 返回None表示需要新部署
                    return None, target_gpu

        return None, None

    def _find_available_gpu_for_duplication(self, model_index: int) -> Optional[int]:
        """
        为模型复制寻找可用的GPU

        Args:
            model_index: 模型索引

        Returns:
            可用的GPU ID，如果没有则返回None
        """
        # 获取模型预估内存
        estimated_memory = 1000  # 默认值，实际应该从配置获取
        if model_index in self.model_index_to_deployments:
            deployment_ids = self.model_index_to_deployments[model_index]
            if deployment_ids:
                estimated_memory = self.model_deployments[deployment_ids[0]].estimated_memory

        best_gpu = None
        best_score = float('inf')

        for gpu_info in self.gpu_list:
            if not gpu_info.is_available:
                continue

            # 检查该GPU是否已有此模型的部署
            gpu_deployment_ids = self.gpu_models[gpu_info.gpu_id]
            has_same_model = any(
                self.model_deployments[did].model_index == model_index
                for did in gpu_deployment_ids
            )

            if has_same_model:
                continue  # 已有部署，跳过

            # 检查内存是否足够
            if gpu_info.memory_free < estimated_memory * 1.2:  # 留20%余量
                continue

            # 检查GPU利用率
            if gpu_info.utilization > self.gpu_utilization_threshold * 0.8:  # 更严格的阈值
                continue

            # 计算得分（利用率越低越好）
            score = gpu_info.utilization

            if score < best_score:
                best_score = score
                best_gpu = gpu_info.gpu_id

        return best_gpu

    def get_optimal_gpu(self, estimated_memory: int = 1000) -> Optional[int]:
        """
        获取最优GPU

        Args:
            estimated_memory: 预估需要的显存(MB)

        Returns:
            最优的GPU ID，如果没有可用的则返回None
        """
        best_gpu = None
        best_score = float('-inf')

        for gpu_info in self.gpu_list:
            if not gpu_info.is_available:
                continue

            # 检查显存是否足够
            if gpu_info.memory_free < estimated_memory:
                continue

            # 计算综合得分
            memory_score = (gpu_info.memory_free / gpu_info.memory_total) * 50
            utilization_score = (100 - gpu_info.utilization) * 30
            temperature_score = max(0, (80 - gpu_info.temperature)) * 0.5
            model_count_penalty = min(gpu_info.model_count * 5, 20)  # 模型数量惩罚

            total_score = memory_score + utilization_score + temperature_score - model_count_penalty

            log_task_debug(f"GPU {gpu_info.gpu_id} 得分: {total_score:.2f} "
                          f"(内存:{memory_score:.1f}, 利用率:{utilization_score:.1f}, "
                          f"温度:{temperature_score:.1f}, 模型数:{-model_count_penalty:.1f})")

            if total_score > best_score:
                best_score = total_score
                best_gpu = gpu_info.gpu_id

        if best_gpu is not None:
            log_task(f"选择最优GPU: {best_gpu} (得分: {best_score:.2f})")

        return best_gpu

    def release_model(self, model_index: int, task_id: str = None):
        """
        释放模型资源 - 支持多实例管理

        Args:
            model_index: 模型索引
            task_id: 任务ID（可选，用于特定实例释放）
        """
        with self._deployment_lock:
            if task_id:
                # 释放特定任务的实例
                deployment_ids = self.model_index_to_deployments.get(model_index, [])
                for deployment_id in deployment_ids[:]:  # 使用切片创建副本
                    deployment = self.model_deployments.get(deployment_id)
                    if deployment and deployment.ref_count > 0:
                        deployment.ref_count -= 1
                        log_task(f"任务 {task_id} 释放模型 {model_index} 的实例 {deployment_id}，剩余引用: {deployment.ref_count}")

                        # 如果引用计数为0，完全清理
                        if deployment.ref_count <= 0:
                            self._cleanup_deployment(deployment_id)
            else:
                # 释放所有实例
                deployment_ids = self.model_index_to_deployments.get(model_index, []).copy()
                for deployment_id in deployment_ids:
                    self._cleanup_deployment(deployment_id)

    def _cleanup_deployment(self, deployment_id: str):
        """清理特定部署"""
        if deployment_id not in self.model_deployments:
            return

        deployment = self.model_deployments[deployment_id]
        model_index = deployment.model_index
        gpu_id = deployment.gpu_id

        # 从各个数据结构中移除
        del self.model_deployments[deployment_id]

        if deployment_id in self.model_index_to_deployments[model_index]:
            self.model_index_to_deployments[model_index].remove(deployment_id)

        if deployment_id in self.gpu_models[gpu_id]:
            self.gpu_models[gpu_id].remove(deployment_id)

        # 更新GPU信息
        if gpu_id < len(self.gpu_list):
            self.gpu_list[gpu_id].model_count = max(0, self.gpu_list[gpu_id].model_count - 1)

        log_task(f"完全清理模型 {model_index} 的部署 {deployment_id} (GPU {gpu_id})")

    def get_model_gpu(self, model_index: int) -> Optional[int]:
        """获取模型最优部署的GPU ID"""
        if model_index not in self.model_index_to_deployments:
            return None

        deployment_ids = self.model_index_to_deployments[model_index]
        if not deployment_ids:
            return None

        # 返回负载最低的GPU
        best_deployment_id, best_gpu_id = self._get_best_deployment_for_model(model_index)
        return best_gpu_id if best_gpu_id is not None else None

    def get_all_gpu_status(self) -> List[GPUInfo]:
        """获取所有GPU状态"""
        with self._monitor_lock:
            return self.gpu_list.copy()

    def get_deployment_stats(self) -> Dict:
        """获取部署统计信息"""
        with self._deployment_lock:
            stats = {
                "total_deployments": len(self.model_deployments),
                "models_deployed": len(self.model_index_to_deployments),
                "gpu_usage": {},
                "model_details": {}
            }

            # GPU使用统计
            for gpu_info in self.gpu_list:
                gpu_deployment_ids = self.gpu_models[gpu_info.gpu_id]
                stats["gpu_usage"][f"gpu_{gpu_info.gpu_id}"] = {
                    "deployments": len(gpu_deployment_ids),
                    "models": [self.model_deployments[did].model_index for did in gpu_deployment_ids],
                    "utilization": gpu_info.utilization,
                    "memory_free": gpu_info.memory_free,
                    "memory_total": gpu_info.memory_total
                }

            # 模型详情
            for model_index, deployment_ids in self.model_index_to_deployments.items():
                model_stats = []
                for deployment_id in deployment_ids:
                    deployment = self.model_deployments[deployment_id]
                    model_stats.append({
                        "deployment_id": deployment_id,
                        "gpu_id": deployment.gpu_id,
                        "ref_count": deployment.ref_count,
                        "estimated_memory": deployment.estimated_memory,
                        "load_time": deployment.load_time
                    })
                stats["model_details"][f"model_{model_index}"] = model_stats

            return stats

    def print_status(self):
        """打印GPU和模型部署状态"""
        print("\n" + "="*60)
        print("动态GPU管理器状态")
        print("="*60)

        # GPU状态
        print("\nGPU状态:")
        for gpu_info in self.gpu_list:
            status = "可用" if gpu_info.is_available else "不可用"
            print(f"  GPU {gpu_info.gpu_id}: {gpu_info.device_name}")
            print(f"    状态: {status}")
            print(f"    显存: {gpu_info.memory_free}MB / {gpu_info.memory_total}MB")
            print(f"    利用率: {gpu_info.utilization:.1f}%")
            print(f"    温度: {gpu_info.temperature:.1f}°C")
            print(f"    部署模型数: {gpu_info.model_count}")

        # 模型部署状态
        print("\n模型部署状态:")
        if not self.model_index_to_deployments:
            print("  暂无部署的模型")
        else:
            for model_index, deployment_ids in self.model_index_to_deployments.items():
                print(f"  模型 {model_index}: {len(deployment_ids)} 个部署实例")
                for deployment_id in deployment_ids:
                    deployment = self.model_deployments[deployment_id]
                    print(f"    {deployment_id}: GPU {deployment.gpu_id}, 引用 {deployment.ref_count}")

        print("="*60)

    def cleanup(self):
        """清理资源"""
        log_task("正在清理动态GPU管理器资源...")

        # 停止监控
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)

        # 清理所有部署
        with self._deployment_lock:
            deployment_ids = list(self.model_deployments.keys())
            for deployment_id in deployment_ids:
                self._cleanup_deployment(deployment_id)

        log_task("动态GPU管理器资源清理完成")


# 创建全局实例
gpu_manager = DynamicGPUManager()