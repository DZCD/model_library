import base64
import cv2
import numpy as np
from typing import Optional, Dict, Any
from model_library.tools.logger import log_task_debug, log_task_error

class VLMVerifier:
    """VLM 多模态验证器"""
    
    def __init__(self, config: Dict[str, Any], global_modelscope_conf: Dict[str, Any] = None):
        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return
            
        self.prompt = config.get('prompt', "这张图片中是否发生了交通事故？请只回答是或否。")
        self.timeout = config.get('timeout', 15.0)
        
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
        使用 VLM 验证事故
        Returns: True 表示确认为事故，False 表示不是事故
        """
        if not self.enabled:
            return True # 如果未启用，默认通过
            
        if image is None or image.size == 0:
            return False
            
        try:
            # 图像编码
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            success, buffer = cv2.imencode(".jpg", rgb_image)
            if not success:
                log_task_error("VLM验证: 图像编码失败")
                return False
                
            b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64_image}"
            
            # 调用 API
            log_task_debug(f"正在调用VLM模型进行验证: {self.model}")
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
                timeout=self.timeout,
                temperature=0.1
            )
            
            if not response.choices or not response.choices[0].message:
                log_task_error("VLM响应为空")
                return False

            content = response.choices[0].message.content.strip().lower()
            log_task_debug(f"VLM验证响应: {content}")
            
            # 简单解析响应
            # 只要包含肯定词，就认为是事故
            positive_keywords = ["是", "yes", "有", "true", "correct", "发生"]
            for kw in positive_keywords:
                if kw in content:
                    return True
            
            return False
            
        except Exception as e:
            log_task_error(f"VLM验证调用失败: {e}")
            # 失败策略：这里选择保守策略，验证失败则不上报
            return False

