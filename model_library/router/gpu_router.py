"""
GPU管理路由组
提供GPU状态查询、模型部署管理等功能
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Path
import logging

from ..tools.gpu_manager import gpu_manager

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/gpu",
    tags=["GPU管理"],
    responses={
        400: {"description": "请求参数错误"},
        404: {"description": "资源未找到"},
        500: {"description": "服务器内部错误"}
    }
)


@router.get(
    "/status",
    summary="获取GPU状态信息",
    description="""
    获取系统中所有GPU的实时状态信息，包括显存使用、利用率、温度等。

    ## 返回信息
    - GPU设备总数和可用数量
    - 每个GPU的详细信息
    - 模型部署情况
    - 负载均衡状态

    ## 监控指标
    - **显存使用**: 已用/总显存(MB)
    - **GPU利用率**: 0-100%
    - **温度**: 摄氏度
    - **已部署模型**: 加载在该GPU上的模型数量

    ## 使用场景
    - 系统状态监控
    - 性能分析和优化
    - 资源使用情况检查
    - 故障诊断和调试
    """,
    response_description="GPU状态详细信息"
)
async def get_gpu_status():
    """获取GPU状态信息"""
    try:
        stats = gpu_manager.get_gpu_stats()

        return {
            "status": "succeed",
            "code": 200,
            "msg": "查询成功",
            "data": stats
        }

    except Exception as e:
        logger.error(f"获取GPU状态失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"获取GPU状态失败: {str(e)}",
            "data": {}
        }


@router.get(
    "/optimal",
    summary="获取最优GPU",
    description="""
    获取当前最优的GPU ID，基于显存、利用率、温度等因素综合评分。

    ## 评分标准
    - **显存可用性**: 空闲显存比例 (权重40%)
    - **GPU利用率**: 越低越好 (权重40%)
    - **温度**: 越低越好 (权重20%)
    - **模型负载**: 已部署模型数量 (加成10%)

    ## 参数说明
    - estimated_memory: 预估需要的显存大小(MB)

    ## 返回信息
    - 最优GPU ID
    - 评分详情
    - 推荐原因

    ## 使用场景
    - 手动部署模型时的GPU选择
    - 负载均衡策略制定
    - 性能优化参考
    """,
    response_description="最优GPU推荐信息"
)
async def get_optimal_gpu(
    estimated_memory: int = 1000
):
    """获取最优GPU"""
    try:
        optimal_gpu = gpu_manager.get_optimal_gpu(estimated_memory)

        if optimal_gpu is not None:
            return {
                "status": "succeed",
                "code": 200,
                "msg": f"推荐使用GPU {optimal_gpu}",
                "data": {
                    "optimal_gpu_id": optimal_gpu,
                    "estimated_memory": estimated_memory,
                    "available_gpus": len(gpu_manager.get_available_gpus())
                }
            }
        else:
            return {
                "status": "error",
                "code": 404,
                "msg": "没有可用的GPU",
                "data": {}
            }

    except Exception as e:
        logger.error(f"获取最优GPU失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"获取最优GPU失败: {str(e)}",
            "data": {}
        }


@router.post(
    "/deploy/{model_index}",
    summary="部署模型到GPU",
    description="""
    手动部署指定模型到最优GPU。

    ## 功能说明
    - 自动选择最优GPU
    - 支持显存预估
    - 负载均衡部署
    - 重复部署检测

    ## 参数说明
    - model_index: 模型索引 (0-7)
    - estimated_memory: 预估显存需求(MB)

    ## 使用场景
    - 预加载模型
    - 手动模型管理
    - 性能优化预热
    """,
    response_description="模型部署结果"
)
async def deploy_model(
    model_index: int = Path(
        ...,
        description="模型索引，范围0-7",
        ge=0, le=7
    ),
    estimated_memory: int = 1000
):
    """部署模型到GPU"""
    try:
        if model_index not in range(8):
            return {
                "status": "error",
                "code": 400,
                "msg": "模型索引必须在0-7范围内",
                "data": {}
            }

        gpu_id = gpu_manager.deploy_model(model_index, estimated_memory)

        if gpu_id is not None:
            return {
                "status": "succeed",
                "code": 200,
                "msg": f"模型 {model_index} 已部署到GPU {gpu_id}",
                "data": {
                    "model_index": model_index,
                    "gpu_id": gpu_id,
                    "estimated_memory": estimated_memory
                }
            }
        else:
            return {
                "status": "error",
                "code": 503,
                "msg": "无法分配GPU资源",
                "data": {}
            }

    except Exception as e:
        logger.error(f"部署模型失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"部署模型失败: {str(e)}",
            "data": {}
        }


@router.get(
    "/model/{model_index}",
    summary="查询模型GPU部署",
    description="""
    查询指定模型部署在哪个GPU上。

    ## 参数说明
    - model_index: 模型索引 (0-7)

    ## 返回信息
    - 部署的GPU ID
    - 部署时间
    - 显存占用情况

    ## 使用场景
    - 模型部署状态查询
    - 资源使用统计
    - 故障排查
    """,
    response_description="模型部署信息"
)
async def get_model_gpu(
    model_index: int = Path(
        ...,
        description="模型索引，范围0-7",
        ge=0, le=7
    )
):
    """查询模型部署的GPU"""
    try:
        gpu_id = gpu_manager.get_model_gpu(model_index)

        if gpu_id is not None:
            return {
                "status": "succeed",
                "code": 200,
                "msg": f"模型 {model_index} 部署在GPU {gpu_id}",
                "data": {
                    "model_index": model_index,
                    "gpu_id": gpu_id,
                    "device": f"cuda:{gpu_id}"
                }
            }
        else:
            return {
                "status": "error",
                "code": 404,
                "msg": f"模型 {model_index} 未部署",
                "data": {}
            }

    except Exception as e:
        logger.error(f"查询模型GPU失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"查询模型GPU失败: {str(e)}",
            "data": {}
        }


@router.delete(
    "/model/{model_index}",
    summary="移除模型部署",
    description="""
    从GPU上移除指定模型，释放资源。

    ## 功能说明
    - 释放GPU显存
    - 更新部署记录
    - 资源回收

    ## 参数说明
    - model_index: 模型索引 (0-7)

    ## 注意事项
    - 移除后模型需要重新加载
    - 确保没有正在使用该模型的任务
    - 建议在系统负载低时执行
    """,
    response_description="模型移除结果"
)
async def remove_model(
    model_index: int = Path(
        ...,
        description="模型索引，范围0-7",
        ge=0, le=7
    )
):
    """移除模型部署"""
    try:
        gpu_manager.remove_model(model_index)

        logger.info(f"已移除模型 {model_index} 的部署")

        return {
            "status": "succeed",
            "code": 200,
            "msg": f"模型 {model_index} 已从GPU移除",
            "data": {
                "model_index": model_index
            }
        }

    except Exception as e:
        logger.error(f"移除模型失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"移除模型失败: {str(e)}",
            "data": {}
        }


@router.post(
    "/balance",
    summary="自动平衡所有模型",
    description="""
    自动将所有模型平衡部署到多个GPU上。

    ## 平衡策略
    - 按模型大小排序，优先部署大模型
    - 综合考虑显存、利用率、温度
    - 负载均衡分配，避免单GPU过载

    ## 功能特点
    - 智能负载均衡
    - 优化资源利用
    - 提升整体性能

    ## 使用场景
    - 系统初始化
    - 性能优化
    - 资源重新分配

    ## 注意事项
    - 可能需要重新加载模型
    - 短暂影响可用性
    - 建议在维护窗口执行
    """,
    response_description="模型平衡部署结果"
)
async def balance_models():
    """自动平衡所有模型"""
    try:
        # 读取模型配置
        from ..tools.utils import Config
        config = Config()
        model_configs = config.model_list

        # 执行自动平衡
        deployment_plan = gpu_manager.auto_balance_models(model_configs)

        return {
            "status": "succeed",
            "code": 200,
            "msg": f"自动平衡完成，部署了 {len(deployment_plan)} 个模型",
            "data": {
                "deployment_plan": deployment_plan,
                "total_models": len(model_configs),
                "deployed_models": len(deployment_plan)
            }
        }

    except Exception as e:
        logger.error(f"自动平衡失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"自动平衡失败: {str(e)}",
            "data": {}
        }