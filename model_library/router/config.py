from fastapi import APIRouter
from model_library.tools.utils import Config
from fastapi.responses import JSONResponse
from typing import Dict, Any

config_router = APIRouter(
    prefix="/config",
    tags=["配置管理"],
    responses={
        404: {"description": "配置项未找到"},
        500: {"description": "服务器内部错误"}
    }
)


@config_router.get(
    "/model",
    summary="获取模型配置列表",
    description="""
    获取系统中所有可用的AI模型配置信息。

    返回的配置包括：
    - 模型名称和路径
    - 置信度阈值
    - 模型特定参数
    - 输入图像尺寸配置

    此接口常用于：
    - 系统初始化时获取可用模型
    - 前端展示支持的检测类型
    - 模型配置验证
    """,
    response_description="模型配置列表，包含每个模型的详细参数信息"
)
def get_model_config() -> Dict[str, Any]:
    """
    获取系统配置的所有AI模型信息

    Returns:
        Dict[str, Any]: 模型配置列表，包含每个模型的详细参数

    Raises:
        HTTPException: 当配置文件不存在或格式错误时
    """
    model_list = Config().model_list
    return JSONResponse(
        content=model_list,
        status_code=200,
        headers={
            "Cache-Control": "public, max-age=300",  # 缓存5分钟
            "Content-Type": "application/json"
        }
    )
    

