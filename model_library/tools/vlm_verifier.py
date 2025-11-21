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
        
        # 优先使用局部配置，否则使用全局配置
        local_ms_conf = config.get('modelscope', {})
        if global_modelscope_conf is None:
            global_modelscope_conf = {}
            
        self.api_key = local_ms_conf.get("api_key") or global_modelscope_conf.get("api_key", "sk-d2d735f9d40b42b5bb2c777df2fc864b")
        self.base_url = local_ms_conf.get("base_url") or global_modelscope_conf.get("base_url", "http://10.1.38.201:8000/v1/")
        self.model = local_ms_conf.get("model") or global_modelscope_conf.get("model", "traffic_accident_qwen2_5vl_32b_detail")
        
        if not self.api_key:
            log_task_error("VLM验证已启用但未配置API Key")
            self.enabled = False
            return
            
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url.rstrip("/") + "/")
        except ImportError:
            log_task_error("缺少 openai 依赖，VLM验证将不可用")
            self.enabled = False

    def verify_accident(self, image: np.ndarray) -> bool:
        """
        使用 VLM 验证事故（支持流式/非流式，带早期退出和超时降级）
        Returns: True 表示确认为事故，False 表示不是事故
        """
        if not self.enabled:
            return True # 如果未启用，默认通过
            
        if image is None or image.size == 0:
            return False
        
        # 根据配置选择流式或非流式
        if self.stream:
            return self._verify_with_stream(image)
        else:
            return self._verify_without_stream(image)
    
    def _verify_with_stream(self, image: np.ndarray) -> bool:
        """流式验证（支持早期退出）"""
        start_time = time.time()
        collected_content = []
        
        try:
            # 图像编码
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            success, buffer = cv2.imencode(".jpg", rgb_image)
            if not success:
                log_task_error("VLM验证: 图像编码失败")
                return False
                
            b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64_image}"
            
            # 调用流式 API
            log_task_debug(f"正在调用VLM模型流式验证: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
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
            
            # 定义关键词
            positive_keywords = ["是", "yes", "有", "true", "correct","发生"]
            negative_keywords = ["否", "no", "没有", "不是", "false", "未"]
            
            # 逐块接收响应
            for chunk in response:
                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > self.stream_timeout:
                    log_task_error(f"VLM流式接收超时({elapsed:.1f}秒)，降级使用算法验证结果")
                    return True  # 超时降级：使用算法验证结果
                
                # 提取内容
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    delta = chunk.choices[0].delta.content
                    if delta:
                        collected_content.append(delta)
                        
                        # 拼接已收到的内容
                        partial = ''.join(collected_content).strip().lower()
                        
                        # 早期退出：检查是否已经包含明确答案
                        if len(partial) > 0:
                            # 检查肯定词
                            for kw in positive_keywords:
                                if kw in partial:
                                    elapsed = time.time() - start_time
                                    log_task_debug(f"VLM提前确认事故(耗时{elapsed:.2f}秒): {partial}")
                                    return True
                            
                            # 检查否定词（需要更谨慎，确保不是误判）
                            # 只有在明确看到否定词且没有肯定词时才返回 False
                            if len(partial) >= 5:  # 至少收到5个字符再判断
                                has_negative = any(kw in partial for kw in negative_keywords)
                                has_positive = any(kw in partial for kw in positive_keywords)
                                
                                if has_negative and not has_positive:
                                    elapsed = time.time() - start_time
                                    log_task_debug(f"VLM提前否认事故(耗时{elapsed:.2f}秒): {partial}")
                                    return False
            
            # 流式接收完成，使用完整内容判断
            content = ''.join(collected_content).strip().lower()
            elapsed = time.time() - start_time
            log_task_debug(f"VLM验证响应(耗时{elapsed:.2f}秒): {content}")
            
            # 最终判断
            for kw in positive_keywords:
                if kw in content:
                    return True
            
            return False
            
        except Exception as e:
            elapsed = time.time() - start_time
            log_task_error(f"VLM流式验证失败(耗时{elapsed:.2f}秒): {e}")
            # 失败/超时策略：降级使用算法验证结果（返回True）
            return True
    
    def _verify_without_stream(self, image: np.ndarray) -> bool:
        """非流式验证（一次性获取完整响应）"""
        start_time = time.time()
        
        try:
            # 图像编码
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            success, buffer = cv2.imencode(".jpg", rgb_image)
            if not success:
                log_task_error("VLM验证: 图像编码失败")
                return False
                
            b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64_image}"
            
            # 调用非流式 API
            log_task_debug(f"正在调用VLM模型非流式验证: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
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
                return False

            content = response.choices[0].message.content.strip().lower()
            elapsed = time.time() - start_time
            log_task_debug(f"VLM验证响应(耗时{elapsed:.2f}秒): {content}")
            
            # 定义关键词并判断
            positive_keywords = ["是", "yes", "有", "true", "correct", "发生"]
            for kw in positive_keywords:
                if kw in content:
                    return True
            
            return False
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_type = type(e).__name__
            log_task_error(f"VLM非流式验证失败(耗时{elapsed:.2f}秒, {error_type}): {e}")
            # 失败/超时策略：降级使用算法验证结果（返回True）
            return True

