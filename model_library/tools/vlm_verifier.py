import base64
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any
from model_library.tools.logger import log_task_debug, log_task_error

class VLMVerifier:
    """VLM 多模态验证器"""

    def __init__(self, config: Dict[str, Any], global_modelscope_conf: Dict[str, Any] = None):
        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return

        self.prompt = config.get('prompt', "这张图片中是否发生了交通事故？请只回答是或否。")
        self.timeout = config.get('timeout', 10.0)
        self.stream = config.get('stream', True)  # 是否使用流式输出，默认开启
        self.stream_timeout = config.get('stream_timeout', 10.0)  # 流式接收超时时间

        # 初始化配置列表（优先级从高到低）
        self.configs = []
        self.current_config_index = 0
        self.client = None

        # 优先使用局部配置，否则使用全局配置
        local_ms_conf = config.get('modelscope', {})
        if global_modelscope_conf is None:
            global_modelscope_conf = {}

        # 构建配置列表（按优先级排序）
        configs = []

        # 1. 局部配置优先
        if local_ms_conf.get("api_key"):
            configs.append({
                "api_key": local_ms_conf.get("api_key"),
                "base_url": local_ms_conf.get("base_url", "http://10.1.38.201:8000/v1/"),
                "model": local_ms_conf.get("model", "traffic_accident_qwen2_5vl_32b_detail"),
                "name": "local_config"
            })

        # 2. 全局配置（modelscope1）作为备用
        if global_modelscope_conf.get("api_key"):
            configs.append({
                "api_key": global_modelscope_conf.get("api_key"),
                "base_url": global_modelscope_conf.get("base_url", "http://10.1.38.201:8000/v1/"),
                "model": global_modelscope_conf.get("model", "traffic_accident_qwen2_5vl_32b_detail"),
                "name": "global_config"
            })

        # 3. 默认配置（modelscope2）作为最后备用
        configs.append({
            "api_key": "ms-12f2520f-7ec4-4a83-b40b-6bcd2ebae367",
            "base_url": "https://api-inference.modelscope.cn/v1/",
            "model": "Qwen/Qwen2.5-VL-32B-Instruct",
            "name": "fallback_config"
        })

        self.configs = configs

        if not self.configs:
            log_task_error("VLM验证已启用但未配置任何可用的API配置")
            self.enabled = False
            return

        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
            # 初始化客户端
            self._init_client()
        except ImportError:
            log_task_error("缺少 openai 依赖，VLM验证将不可用")
            self.enabled = False

    def _check_keywords_optimized(self, text: str) -> str:
        """
        优化的关键词检查方法，解决"没有"包含"有"等冲突问题
        Returns: "positive", "negative", "unknown"
        """
        text_lower = text.lower().strip()

        # 1. 优先检查完整的冲突词汇（主要问题解决）
        full_negative_words = ["没有", "不是", "并无", "并未", "并不存在", "并没有", "并无发生", "并无事故"]
        for word in full_negative_words:
            if word in text_lower:
                return "negative"

        # 2. 再检查单字否定词（但要排除肯定词上下文）
        single_negative = ["否", "no", "未", "false"]
        for neg in single_negative:
            if neg in text_lower:
                # 确保不与肯定词形成冲突
                if not any(pos in text_lower for pos in ["有", "是", "yes"]):
                    return "negative"

        # 3. 最后检查肯定词
        positive_words = ["是", "yes", "有", "true", "correct", "发生", "确实", "的确", "确实存在"]
        for word in positive_words:
            if word in text_lower:
                return "positive"

        return "unknown"

    def _init_client(self):
        """初始化OpenAI客户端，支持配置切换"""
        if self.current_config_index >= len(self.configs):
            log_task_error("所有VLM配置都已尝试失败")
            self.client = None
            return False

        config = self.configs[self.current_config_index]
        try:
            self.client = self.OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"].rstrip("/") + "/"
            )
            log_task_debug(f"VLM初始化客户端成功，使用配置: {config['name']}")
            return True
        except Exception as e:
            log_task_error(f"VLM初始化客户端失败，配置: {config['name']}, 错误: {e}")
            return False

    def _try_next_config(self):
        """尝试下一个配置"""
        self.current_config_index += 1
        if self.current_config_index >= len(self.configs):
            log_task_error("所有VLM配置都已尝试失败")
            return False

        log_task_debug(f"VLM尝试切换到配置 #{self.current_config_index + 1}")
        return self._init_client()

    @property
    def current_config(self):
        """获取当前配置"""
        if self.current_config_index < len(self.configs):
            return self.configs[self.current_config_index]
        return None

    def verify_accident(self, image: np.ndarray) -> bool:
        """
        使用 VLM 验证事故（支持流式/非流式，带早期退出和超时降级）
        Returns: True 表示确认为事故，False 表示不是事故
        """
        if not self.enabled:
            return True # 如果未启用，默认通过

        if image is None or image.size == 0:
            return False

        # 重置配置索引，每次验证都从最优配置开始尝试
        self.current_config_index = 0
        # 确保初始化至少一个可用的客户端
        if not self._init_client():
            # 如果第一个配置初始化失败，尝试其他配置
            while self.current_config_index < len(self.configs):
                if self._try_next_config():
                    break

        # 根据配置选择流式或非流式
        if self.stream:
            return self._verify_with_stream(image)
        else:
            return self._verify_without_stream(image)
    
    def _verify_with_stream(self, image: np.ndarray) -> bool:
        """流式验证（支持早期退出和配置切换）"""
        start_time = time.time()

        # 图像编码（只需要编码一次）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".jpg", rgb_image)
        if not success:
            log_task_error("VLM验证: 图像编码失败")
            return False

        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64_image}"

        # 尝试所有可用配置
        while self.current_config_index < len(self.configs):
            config = self.current_config
            if not config or not self.client:
                if not self._try_next_config():
                    break
                continue

            collected_content = []
            config_start_time = time.time()

            try:
                # 调用流式 API
                log_task_debug(f"正在调用VLM模型流式验证: {config['model']}, 配置: {config['name']}")
                response = self.client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    stream=True,  # 开启流式输出
                    timeout=self.timeout,
                    temperature=0.1
                )

                # 逐块接收响应
                for chunk in response:
                    # 检查超时
                    elapsed = time.time() - start_time
                    if elapsed > self.stream_timeout:
                        log_task_error(f"VLM流式接收超时({elapsed:.1f}秒)，尝试下一个配置")
                        break  # 跳出循环，尝试下一个配置

                    # 提取内容
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        delta = chunk.choices[0].delta.content
                        if delta:
                            collected_content.append(delta)

                            # 拼接已收到的内容
                            partial = ''.join(collected_content).strip()

                            # 早期退出：使用优化的关键词检查
                            if len(partial) > 0:
                                keyword_result = self._check_keywords_optimized(partial)

                                # 提前确认：发现肯定关键词
                                if keyword_result == "positive":
                                    elapsed = time.time() - start_time
                                    log_task_debug(f"VLM提前确认事故(耗时{elapsed:.2f}秒, 配置: {config['name']}): {partial}")
                                    return True

                                # 提前否认：需要足够长度且发现否定关键词
                                if len(partial) >= 5 and keyword_result == "negative":
                                    elapsed = time.time() - start_time
                                    log_task_debug(f"VLM提前否认事故(耗时{elapsed:.2f}秒, 配置: {config['name']}): {partial}")
                                    return False

                # 流式接收完成，使用完整内容判断
                content = ''.join(collected_content).strip()
                elapsed = time.time() - start_time
                log_task_debug(f"VLM验证响应成功(耗时{elapsed:.2f}秒, 配置: {config['name']}): {content}")

                # 最终判断使用优化的关键词检查
                keyword_result = self._check_keywords_optimized(content)
                return keyword_result == "positive"

            except Exception as e:
                elapsed = time.time() - config_start_time
                error_type = type(e).__name__
                log_task_error(f"VLM流式验证失败(耗时{elapsed:.2f}秒, 配置: {config['name']}, {error_type}): {e}")

                # 尝试下一个配置
                if not self._try_next_config():
                    break
                continue

        # 所有配置都尝试失败，降级使用算法验证结果
        log_task_error("所有VLM配置都失败，降级使用算法验证结果")
        return True
    
    def _verify_without_stream(self, image: np.ndarray) -> bool:
        """非流式验证（支持配置切换）"""
        start_time = time.time()

        # 图像编码（只需要编码一次）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".jpg", rgb_image)
        if not success:
            log_task_error("VLM验证: 图像编码失败")
            return False

        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64_image}"

        # 尝试所有可用配置
        while self.current_config_index < len(self.configs):
            config = self.current_config
            if not config or not self.client:
                if not self._try_next_config():
                    break
                continue

            config_start_time = time.time()

            try:
                # 调用非流式 API
                log_task_debug(f"正在调用VLM模型非流式验证: {config['model']}, 配置: {config['name']}")
                response = self.client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    stream=False,  # 关闭流式输出
                    timeout=self.timeout,
                    temperature=0.1
                )

                if not response.choices or not response.choices[0].message:
                    log_task_error("VLM响应为空")
                    raise Exception("VLM响应为空")

                content = response.choices[0].message.content.strip()
                elapsed = time.time() - start_time
                log_task_debug(f"VLM验证响应成功(耗时{elapsed:.2f}秒, 配置: {config['name']}): {content}")

                # 使用优化的关键词检查方法
                keyword_result = self._check_keywords_optimized(content)
                return keyword_result == "positive"

            except Exception as e:
                elapsed = time.time() - config_start_time
                error_type = type(e).__name__
                log_task_error(f"VLM非流式验证失败(耗时{elapsed:.2f}秒, 配置: {config['name']}, {error_type}): {e}")

                # 尝试下一个配置
                if not self._try_next_config():
                    break
                continue

        # 所有配置都尝试失败，降级使用算法验证结果
        log_task_error("所有VLM配置都失败，降级使用算法验证结果")
        return True

