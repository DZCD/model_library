"""
OpenCV VideoCapture后端配置模块
支持多种后端选择和配置，实现代码解耦
"""

import os
import cv2
import time
from typing import Optional, Dict, Any, Union
from enum import Enum


class VideoBackend(Enum):
    """视频后端枚举"""
    FFMPEG = "ffmpeg"
    GSTREAMER = "gstreamer"
    MSMF = "msmf"  # Windows Media Foundation
    DIRECTSHOW = "directshow"  # Windows only
    AVFOUNDATION = "avfoundation"  # macOS only
    AUTO = "auto"  # 自动选择


class VideoBackendConfig:
    """视频后端配置类"""

    # 后端到OpenCV常量的映射
    BACKEND_MAP = {
        VideoBackend.FFMPEG: cv2.CAP_FFMPEG,
        VideoBackend.GSTREAMER: cv2.CAP_GSTREAMER,
        VideoBackend.MSMF: cv2.CAP_MSMF,
        VideoBackend.AUTO: cv2.CAP_ANY
    }

    # 平台特定的后端映射
    PLATFORM_SPECIFIC_MAP = {
        VideoBackend.DIRECTSHOW: getattr(cv2, 'CAP_DIRECTSHOW', None),
        VideoBackend.AVFOUNDATION: getattr(cv2, 'CAP_AVFOUNDATION', None)
    }

    def __init__(self,
                 backend: VideoBackend = VideoBackend.FFMPEG,
                 max_retries: int = 5,
                 retry_delay: float = 1.5,
                 buffer_size: int = 3,
                 stream_timeout: float = 15.0):
        """
        初始化视频后端配置

        Args:
            backend: 选择的后端类型
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            buffer_size: 缓冲区大小
            stream_timeout: 流超时时间（秒）
        """
        self.backend = backend
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.buffer_size = buffer_size
        self.stream_timeout = stream_timeout

        # 设置环境变量
        self._setup_environment()

    def _setup_environment(self):
        """设置环境变量"""
        if self.backend == VideoBackend.FFMPEG:
            os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
            os.environ["OPENCV_VIDEOIO_PRIORITY_IMAGES"] = "0"
        elif self.backend == VideoBackend.GSTREAMER:
            os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "1"

    def get_opencv_backend(self) -> int:
        """获取OpenCV后端常量"""
        # 首先检查通用后端映射
        if self.backend in self.BACKEND_MAP:
            return self.BACKEND_MAP[self.backend]

        # 然后检查平台特定后端映射
        if self.backend in self.PLATFORM_SPECIFIC_MAP:
            backend_value = self.PLATFORM_SPECIFIC_MAP[self.backend]
            if backend_value is not None:
                return backend_value
            else:
                print(f"[后端配置] [警告] {self.backend.value} 后端在当前OpenCV版本中不可用，使用默认后端")

        return cv2.CAP_ANY

    def is_stream_url(self, source: str) -> bool:
        """检测是否为流媒体URL"""
        if not isinstance(source, str):
            return False

        stream_protocols = ['rtmp://', 'rtsp://', 'http://', 'https://', 'udp://', 'tcp://']
        return any(protocol in source.lower() for protocol in stream_protocols)

    def create_gstreamer_pipeline(self, source: str, **kwargs) -> str:
        """
        创建GStreamer管道字符串

        Args:
            source: 视频源URL或文件路径
            **kwargs: 额外的GStreamer参数

        Returns:
            GStreamer管道字符串
        """
        if self.is_stream_url(source):
            # 网络流的GStreamer管道
            pipeline = (
                f"uridecodebin uri={source} ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw,format=BGR ! "
                f"appsink name=appsink"
            )
        else:
            # 本地文件的GStreamer管道
            pipeline = (
                f"filesrc location={source} ! "
                f"decodebin ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw,format=BGR ! "
                f"appsink name=appsink"
            )

        return pipeline

    def get_backend_params(self, source: str) -> Dict[str, Any]:
        """
        获取后端特定的参数

        Args:
            source: 视频源

        Returns:
            包含后端参数的字典
        """
        params = {}

        if self.backend == VideoBackend.GSTREAMER and self.is_stream_url(source):
            # GStreamer需要管道字符串作为源
            pipeline = self.create_gstreamer_pipeline(source)
            params['source'] = pipeline
        elif self.backend == VideoBackend.GSTREAMER:
            # 本地文件也使用GStreamer管道
            pipeline = self.create_gstreamer_pipeline(source)
            params['source'] = pipeline
        else:
            # 其他后端直接使用源
            params['source'] = source

        # 添加通用参数
        params['apiPreference'] = self.get_opencv_backend()

        return params


class EnhancedVideoCapture:
    """增强的VideoCapture类，支持多种后端和重试机制"""

    def __init__(self, source: Any, config: Optional[VideoBackendConfig] = None, **kwargs):
        """
        初始化增强的VideoCapture

        Args:
            source: 视频源（文件路径、URL、摄像头索引等）
            config: 后端配置，如果为None则使用默认配置
            **kwargs: 额外的VideoCapture参数
        """
        self.source = source
        self.config = config or VideoBackendConfig()
        self.kwargs = kwargs

        # 保存原始VideoCapture以备不时之需
        self._original_videocapture = cv2.VideoCapture
        self._capture = None
        self._is_opened = False

        # 尝试打开视频源
        self._open_with_retry()

    def _open_with_retry(self):
        """带重试机制的视频源打开"""
        is_stream = self.config.is_stream_url(str(self.source))

        # 获取后端特定参数
        backend_params = self.config.get_backend_params(str(self.source))
        actual_source = backend_params.pop('source', self.source)

        # 合并配置参数
        final_kwargs = {**self.kwargs, **backend_params}

        print(f"[后端配置] 使用 {self.config.backend.value} 后端")
        if self.config.backend == VideoBackend.GSTREAMER:
            print(f"[后端配置] GStreamer管道: {actual_source}")

        # 重试机制
        for attempt in range(self.config.max_retries):
            try:
                self._capture = self._original_videocapture(actual_source, **final_kwargs)

                if self._capture.isOpened():
                    self._is_opened = True

                    # 设置缓冲区大小
                    if hasattr(self._capture, 'set'):
                        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

                    if attempt > 0:
                        print(f"[后端配置] [成功] 视频源打开成功（第{attempt + 1}次尝试）")
                    else:
                        print(f"[后端配置] [成功] 视频源打开成功")
                    return
                else:
                    self._capture.release()
                    self._capture = None

            except Exception as e:
                print(f"[后端配置] [异常] 打开视频源时发生异常: {e}")
                if self._capture:
                    self._capture.release()
                    self._capture = None

            if is_stream and attempt < self.config.max_retries - 1:
                print(f"[后端配置] [重试] 视频源打开失败，{self.config.retry_delay}秒后重试（{attempt + 1}/{self.config.max_retries}）...")
                time.sleep(self.config.retry_delay)
            else:
                break

        print(f"[后端配置] [失败] 视频源打开失败（已尝试{self.config.max_retries}次）")

    def __getattr__(self, name):
        """代理所有VideoCapture的方法"""
        if self._capture is None:
            raise RuntimeError("VideoCapture未成功初始化")
        return getattr(self._capture, name)

    def __del__(self):
        """析构函数，确保释放资源"""
        if self._capture:
            self._capture.release()

    def release(self):
        """释放视频捕获资源"""
        if self._capture:
            self._capture.release()
            self._is_opened = False
            print("[后端配置] 视频源已释放")

    @property
    def isOpened(self) -> bool:
        """检查视频源是否打开"""
        return self._is_opened and (self._capture is not None) and self._capture.isOpened()


def create_video_capture(source: Any,
                        backend: Union[VideoBackend, str] = VideoBackend.FFMPEG,
                        **kwargs) -> EnhancedVideoCapture:
    """
    便捷函数：创建增强的VideoCapture实例

    Args:
        source: 视频源
        backend: 后端类型（VideoBackend枚举或字符串）
        **kwargs: 额外参数

    Returns:
        EnhancedVideoCapture实例
    """
    if isinstance(backend, str):
        try:
            backend = VideoBackend(backend.lower())
        except ValueError:
            print(f"[后端配置] 未知的后端类型: {backend}，使用默认FFmpeg后端")
            backend = VideoBackend.FFMPEG

    config = VideoBackendConfig(backend=backend)
    return EnhancedVideoCapture(source, config, **kwargs)


# 便捷函数，用于快速切换后端
def create_gstreamer_capture(source: Any, **kwargs) -> EnhancedVideoCapture:
    """创建使用GStreamer后端的VideoCapture"""
    return create_video_capture(source, VideoBackend.GSTREAMER, **kwargs)


def create_ffmpeg_capture(source: Any, **kwargs) -> EnhancedVideoCapture:
    """创建使用FFmpeg后端的VideoCapture"""
    return create_video_capture(source, VideoBackend.FFMPEG, **kwargs)