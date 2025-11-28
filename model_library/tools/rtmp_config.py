"""
RTMP流连接配置管理
提供不同场景下的RTMP连接参数配置
"""

from .video_backend import VideoBackend, VideoBackendConfig

class RTMPConfigManager:
    """RTMP配置管理器"""

    @staticmethod
    def get_rtmp_config(stream_type="default", custom_params=None):
        """
        获取RTMP流配置

        Args:
            stream_type: 流类型 ("default", "stable", "unstable", "high_latency")
            custom_params: 自定义参数覆盖

        Returns:
            VideoBackendConfig: 配置好的后端配置
        """

        # 预定义的配置模板
        configs = {
            "default": {
                "backend": VideoBackend.FFMPEG,
                "max_retries": 5,
                "retry_delay": 2.0,
                "buffer_size": 3,
                "stream_timeout": 15.0
            },
            "stable": {
                "backend": VideoBackend.FFMPEG,
                "max_retries": 3,
                "retry_delay": 1.0,
                "buffer_size": 2,
                "stream_timeout": 10.0
            },
            "unstable": {
                "backend": VideoBackend.FFMPEG,
                "max_retries": 8,
                "retry_delay": 3.0,
                "buffer_size": 5,
                "stream_timeout": 20.0
            },
            "high_latency": {
                "backend": VideoBackend.FFMPEG,
                "max_retries": 5,
                "retry_delay": 5.0,
                "buffer_size": 10,
                "stream_timeout": 30.0
            }
        }

        # 获取基础配置
        config_params = configs.get(stream_type, configs["default"])

        # 应用自定义参数
        if custom_params:
            config_params.update(custom_params)

        return VideoBackendConfig(**config_params)

    @staticmethod
    def get_detector_config(rtmp_url):
        """
        根据RTMP URL自动检测并返回合适的配置

        Args:
            rtmp_url: RTMP流地址

        Returns:
            VideoBackendConfig: 配置好的后端配置
        """
        url_lower = rtmp_url.lower()

        # 根据URL特征判断流类型
        if "10.1.38.201" in url_lower:
            # 内网服务器，相对稳定
            return RTMPConfigManager.get_rtmp_config("default")
        elif any(keyword in url_lower for keyword in ["live", "stream"]):
            # 直播流，可能不稳定
            return RTMPConfigManager.get_rtmp_config("default")
        else:
            # 默认配置
            return RTMPConfigManager.get_rtmp_config("default")

    @staticmethod
    def calculate_vid_stride(fps, target_fps=2):
        """
        计算合适的帧间隔

        Args:
            fps: 视频FPS
            target_fps: 目标推理FPS

        Returns:
            int: 帧间隔
        """
        if fps <= 0:
            return 2  # 默认间隔

        stride = max(1, int(fps / target_fps))

        # 限制最大间隔，避免跳帧太多
        return min(stride, 30)

    @staticmethod
    def get_wait_time_for_rtmp():
        """
        获取RTMP流连接后的等待时间

        Returns:
            float: 等待时间（秒）
        """
        return 2.0  # RTMP流连接后等待2秒


# 全局配置实例
rtmp_config_manager = RTMPConfigManager()

# 便捷函数
def get_rtmp_config(stream_type="default", custom_params=None):
    """获取RTMP配置的便捷函数"""
    return rtmp_config_manager.get_rtmp_config(stream_type, custom_params)

def auto_rtmp_config(rtmp_url):
    """自动检测RTMP配置的便捷函数"""
    return rtmp_config_manager.get_detector_config(rtmp_url)