"""
模型推理路由组
提供视频流模型推理服务，支持异步执行和MQTT结果推送
"""

import asyncio
import uuid
import json
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Form
import logging
from starlette.responses import JSONResponse
from PIL import Image

from ..tools.detector import Detector
from ..tools.reasoner import Reasoner

reasoner = Reasoner()

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/infer", tags=["模型推理"])

# 存储运行中的任务
running_tasks: Dict[str, Dict[str, Any]] = {}


def run_workflow_in_thread(workflow: Detector, task_id: str):
    """在线程中运行异步workflow"""
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 运行异步workflow
        loop.run_until_complete(workflow.run_video())

        # 更新状态为完成
        if task_id in running_tasks:
            running_tasks[task_id]["status"] = "completed"
            running_tasks[task_id]["end_time"] = datetime.now().isoformat()

    except Exception as e:
        # 更新状态为失败
        if task_id in running_tasks:
            if workflow.is_stop_requested():
                running_tasks[task_id]["status"] = "stopped"
            else:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error_message"] = str(e)
            running_tasks[task_id]["end_time"] = datetime.now().isoformat()
    finally:
        loop.close()


@router.post("/video")
async def start_inference(
        video_path: str = Form(..., description="视频流地址（支持rtmp、rtsp、本地文件等）"),
        model_index: int = Form(..., description="模型类型 (0:电梯摩托车, 1:消防通道占用, 2:火点检测, 3:事故检测)"),
        pixel_position: Optional[str] = Form(None, description="像素位置（仅模型1需要，JSON格式的多边形顶点坐标列表）")
):
    """
    开始模型推理任务
    
    - **video_path**: 视频流地址（支持rtmp、rtsp、本地文件等）
    - **model_index**: 模型类型 (0:电梯摩托车, 1:消防通道占用, 2:火点检测, 3:事故检测)
    - **pixel_position**: 像素位置（仅模型1需要，JSON格式的多边形顶点坐标列表，如：[[0,941],[0,1342],[2152,1338],[2173,586],[1110,460]]）
    
    返回MQTT主题名称，推理结果将实时推送到该主题
    """
    try:
        # 输入验证
        if not video_path or not video_path.strip():
            return {
                "status": "error",
                "code": 400,
                "msg": "视频路径不能为空",
                "data": {}
            }

        if model_index not in [0, 1, 2, 3]:
            return {
                "status": "error",
                "code": 400,
                "msg": "模型索引必须是 0, 1, 2, 3 中的一个",
                "data": {}
            }

        # 解析pixel_position
        parsed_pixel_position = None
        if pixel_position:
            try:
                parsed_pixel_position = json.loads(pixel_position)
                if not isinstance(parsed_pixel_position, list):
                    raise ValueError("pixel_position必须是坐标列表")
            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "status": "error",
                    "code": 400,
                    "msg": f"pixel_position格式错误: {str(e)}",
                    "data": {}
                }

        # 生成唯一任务ID
        task_id = str(uuid.uuid4())

        # 创建workflow实例
        workflow = Detector(
            model_index=model_index,
            video_path=video_path.strip(),
            pixel_position=parsed_pixel_position,
            task_id = task_id
        )

        # 获取MQTT主题和模型名称
        mqtt_topic = workflow.topic
        model_name = workflow.model_name

        # 在后台线程中启动任务
        thread = threading.Thread(target=run_workflow_in_thread, args=(workflow, task_id), daemon=True)

        # 记录任务信息（包括线程引用）
        running_tasks[task_id] = {
            "workflow": workflow,
            "thread": thread,
            "status": "running",
            "mqtt_topic": mqtt_topic,
            "model_name": model_name,
            "start_time": None,
            "error_message": None
        }

        thread.start()

        logger.info(f"创建推理任务 {task_id}, 模型: {model_name}, MQTT主题: {mqtt_topic}")

        resp = {
            "status": "succeed",
            "code": 200,
            "msg": "推理任务已创建，正在后台执行",
            "data": {
                "task_id": task_id,
                "mqtt_topic": mqtt_topic,
                "model_name": model_name,
                "task_status": "running",
                "start_time": datetime.now().isoformat()
            }
        }
        return resp
    except Exception as e:
        logger.error(f"创建推理任务失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"创建推理任务失败: {str(e)}",
            "data": {}
        }


@router.post("/image")
async def start_inference_image(
        image_path: str = Form(..., description="图片地址（支持本地文件、线上图片）"),
        model_index: int = Form(..., description="模型类型 (0:电梯摩托车, 1:消防通道占用, 2:火点检测, 3:事故检测)"),
        pixel_position: Optional[str] = Form(None, description="像素位置（仅模型1需要，JSON格式的多边形顶点坐标列表）")
):
    """
    开始模型推理任务

    - **image_path**: 图片地址（支持本地文件、线上图片）
    - **model_index**: 模型类型 (0:电梯摩托车, 1:消防通道占用, 2:火点检测, 3:事故检测)
    - **pixel_position**: 像素位置（仅模型1需要，JSON格式的多边形顶点坐标列表，如：[[0,941],[0,1342],[2152,1338],[2173,586],[1110,460]]）

    返回MQTT主题名称，推理结果将实时推送到该主题
    """
    resp = {
        "status": "succeed",
        "code": 200,
        "msg": "推理成功",
        "data": {}
    }
    try:

        result = reasoner.infer_image(image_path, model_index)
        resp["data"]["item"] = result
        if not result:
            resp["code"] = 500
        return resp
    except Exception as e:
        resp["msg"] = f"图片推理失败：{e}"
        resp["status"] = "defend"
        resp["code"] = 500
        return resp


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    查询推理任务状态
    
    - **task_id**: 任务ID
    
    返回任务当前状态信息
    """
    if task_id not in running_tasks:
        return {
            "status": "error",
            "code": 404,
            "msg": "任务不存在",
            "data": {}
        }

    task_info = running_tasks[task_id]

    # 判断当前任务状态  
    current_status = task_info["status"]
    workflow = task_info["workflow"]
    if workflow.is_stop_requested() and current_status == "running":
        current_status = "stopping"

    return {
        "status": "succeed",
        "code": 200,
        "msg": "查询成功",
        "data": {
            "task_id": task_id,
            "task_status": current_status,
            "mqtt_topic": task_info["mqtt_topic"],
            "model_name": task_info["model_name"],
            "start_time": task_info.get("start_time", ""),
            "error_message": task_info.get("error_message"),
            "stop_requested": workflow.is_stop_requested()
        }
    }


@router.delete("/stop/{task_id}")
async def stop_task(task_id: str):
    """
    停止推理任务
    
    - **task_id**: 任务ID
    
    停止指定的推理任务
    """
    if task_id not in running_tasks:
        return {
            "status": "error",
            "code": 404,
            "msg": "任务不存在",
            "data": {}
        }

    task_info = running_tasks[task_id]

    if task_info["status"] in ["completed", "failed", "stopped"]:
        return {
            "status": "succeed",
            "code": 200,
            "msg": f"任务已结束，状态: {task_info['status']}",
            "data": {
                "task_id": task_id,
                "task_status": task_info["status"]
            }
        }

    # 设置停止请求标志
    workflow = task_info["workflow"]
    workflow.request_stop()

    logger.info(f"发送停止请求到推理任务 {task_id}")

    return {
        "status": "succeed",
        "code": 200,
        "msg": "停止请求已发送，任务将在短时间内停止",
        "data": {
            "task_id": task_id,
            "task_status": "stopping"
        }
    }


@router.get("/tasks")
async def list_tasks():
    """
    列出所有推理任务
    
    返回当前所有任务的状态信息
    """
    tasks = []
    for task_id, task_info in running_tasks.items():
        # 判断当前任务状态
        current_status = task_info["status"]
        workflow = task_info["workflow"]
        if workflow.is_stop_requested() and current_status == "running":
            current_status = "stopping"

        tasks.append({
            "task_id": task_id,
            "task_status": current_status,
            "mqtt_topic": task_info["mqtt_topic"],
            "model_name": task_info["model_name"],
            "start_time": task_info.get("start_time", ""),
            "error_message": task_info.get("error_message"),
            "stop_requested": workflow.is_stop_requested()
        })

    return {
        "status": "succeed",
        "code": 200,
        "msg": "查询成功",
        "data": {
            "tasks": tasks,
            "total": len(tasks)
        }
    }


@router.get("/models")
async def get_models():
    """
    获取支持的模型列表
    
    返回所有可用的模型信息
    """
    models = {
        0: {
            "name": "elevator_motor",
            "description": "电梯摩托车检测",
            "requires_pixel_position": False
        },
        1: {
            "name": "fire_lane_blockage",
            "description": "消防通道占用检测",
            "requires_pixel_position": True
        },
        2: {
            "name": "fire_detect",
            "description": "火点检测",
            "requires_pixel_position": False
        },
        3: {
            "name": "accident",
            "description": "事故检测",
            "requires_pixel_position": False
        }
    }

    return {
        "status": "succeed",
        "code": 200,
        "msg": "查询成功",
        "data": {
            "models": models
        }
    }


@router.delete("/cleanup")
async def cleanup_completed_tasks():
    """
    清理已完成的任务
    
    删除所有已完成、失败或停止的任务记录
    """
    try:
        completed_tasks = []
        for task_id, task_info in list(running_tasks.items()):
            if task_info["status"] in ["completed", "failed", "stopped"]:
                completed_tasks.append(task_id)
                del running_tasks[task_id]

        logger.info(f"清理了 {len(completed_tasks)} 个已完成的任务")

        return {
            "status": "succeed",
            "code": 200,
            "msg": f"成功清理 {len(completed_tasks)} 个已完成的任务",
            "data": {
                "cleaned_tasks": completed_tasks,
                "remaining_tasks": len(running_tasks)
            }
        }

    except Exception as e:
        logger.error(f"清理任务失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"清理任务失败: {str(e)}",
            "data": {}
        }
