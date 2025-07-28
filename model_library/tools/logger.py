"""
最简单的双日志系统：API日志 + 任务日志
"""
import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime

# 创建日志目录
Path('logs').mkdir(exist_ok=True)

# 日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 1. API日志器 - 记录接口请求响应
api_logger = logging.getLogger('api')
api_logger.setLevel(logging.INFO)
api_handler = logging.handlers.RotatingFileHandler(
    'logs/api_history.log', maxBytes=10*1024*1024, backupCount=3, encoding='utf-8'
)
api_handler.setFormatter(formatter)
api_logger.addHandler(api_handler)

# 2. 任务日志器 - 记录调试信息
task_logger = logging.getLogger('task')
task_logger.setLevel(logging.DEBUG)
task_handler = logging.handlers.RotatingFileHandler(
    'logs/task_debug.log', maxBytes=10*1024*1024, backupCount=3, encoding='utf-8'
)
task_handler.setFormatter(formatter)
task_logger.addHandler(task_handler)

# API日志函数
def log_api_complete(method: str, path: str, client_ip: str, 
                    request_params: dict = None, 
                    response_data: dict = None, 
                    status_code: int = 200, 
                    duration: float = 0,
                    error: str = None):
    """记录完整的API调用信息"""
    
    # 构建日志消息
    log_data = {
        "method": method,
        "path": path,
        "client_ip": client_ip,
        "request_params": request_params or {},
        "response": response_data or {},
        "status_code": status_code,
        "duration_seconds": round(duration, 3),
        "timestamp": datetime.now().isoformat()
    }
    
    if error:
        log_data["error"] = error
        api_logger.error(f"API调用失败: {json.dumps(log_data, ensure_ascii=False)}")
    else:
        api_logger.info(f"API调用: {json.dumps(log_data, ensure_ascii=False)}")

# 任务日志函数
def log_task(message: str):
    """记录任务信息"""
    task_logger.info(message)

def log_task_error(message: str):
    """记录任务错误"""
    task_logger.error(message)

def log_task_debug(message: str):
    """记录任务调试信息"""
    task_logger.debug(message) 