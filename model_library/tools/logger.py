import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ModelLogger:
    """
    统一的日志管理类，支持文件和控制台输出
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger('model_library')
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_handlers()
        
        self._initialized = True
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 创建日志目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # 日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器 - 通用日志
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / 'model_library.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志单独文件
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / 'errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常日志（自动包含异常堆栈）"""
        self.logger.exception(message, **kwargs)
    
    def set_level(self, level: Union[str, int]):
        """
        设置日志级别
        Args:
            level: 日志级别，可以是字符串('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
                  或整数(10, 20, 30, 40, 50)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
    
    def add_file_handler(self, filename: str, level: Union[str, int] = logging.INFO):
        """
        添加自定义文件处理器
        Args:
            filename: 日志文件名
            level: 日志级别
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_function_call(self, func_name: str, params: dict = None):
        """记录函数调用"""
        if params:
            self.debug(f"调用函数: {func_name}, 参数: {params}")
        else:
            self.debug(f"调用函数: {func_name}")
    
    def log_performance(self, operation: str, duration: float, details: str = ""):
        """记录性能日志"""
        self.info(f"性能统计 - {operation}: {duration:.3f}秒 {details}")
    
    def log_stream_status(self, stream_url: str, status: str, details: str = ""):
        """记录流状态"""
        self.info(f"流状态 - {stream_url}: {status} {details}")
    
    def log_model_inference(self, model_name: str, input_info: str, result_count: int, duration: float):
        """记录模型推理"""
        self.info(f"模型推理 - {model_name}: 输入={input_info}, 结果数={result_count}, 耗时={duration:.3f}秒")


# 创建全局日志实例
logger = ModelLogger()


def get_logger() -> ModelLogger:
    """获取日志实例"""
    return logger


def setup_custom_logger(name: str, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """
    创建自定义日志器
    Args:
        name: 日志器名称
        level: 日志级别
    Returns:
        配置好的日志器
    """
    custom_logger = logging.getLogger(name)
    custom_logger.setLevel(level)
    
    if not custom_logger.handlers:
        # 使用主日志器的配置
        main_logger = get_logger()
        for handler in main_logger.logger.handlers:
            custom_logger.addHandler(handler)
    
    return custom_logger


# 便捷函数
def debug(message: str, **kwargs):
    """快捷调试日志"""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """快捷信息日志"""
    logger.info(message, **kwargs)


def warning(message: str, **kwargs):
    """快捷警告日志"""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """快捷错误日志"""
    logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """快捷严重错误日志"""
    logger.critical(message, **kwargs)


def exception(message: str, **kwargs):
    """快捷异常日志"""
    logger.exception(message, **kwargs)


# 装饰器：自动记录函数调用
def log_function(level: str = 'DEBUG'):
    """
    函数调用日志装饰器
    Args:
        level: 日志级别
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # 记录函数调用
            if level.upper() == 'DEBUG':
                logger.debug(f"开始执行: {func_name}")
            elif level.upper() == 'INFO':
                logger.info(f"开始执行: {func_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"完成执行: {func_name}")
                return result
            except Exception as e:
                logger.error(f"执行异常: {func_name}, 错误: {str(e)}")
                raise
        
        return wrapper
    return decorator


# 装饰器：性能监控
def log_performance(operation_name: str = None):
    """
    性能监控装饰器
    Args:
        operation_name: 操作名称，默认使用函数名
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_performance(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_performance(op_name, duration, f"(异常: {str(e)})")
                raise
        
        return wrapper
    return decorator 