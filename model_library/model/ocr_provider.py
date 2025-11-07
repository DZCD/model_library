"""OCR Provider 抽象与实现"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from model_library.model.ocr_model import (
    color,
    decodePlate,
    get_split_merge,
    image_processing,
    init_model,
    plateName,
)
from model_library.tools.logger import log_task_debug, log_task_error


class BaseOCRProvider(ABC):
    """OCR 提供者抽象基类"""

    @abstractmethod
    def recognize(self, roi_image: np.ndarray, is_double_layer: bool = False) -> Tuple[str, Dict[str, Any]]:
        """识别车牌"""


class LocalOCRProvider(BaseOCRProvider):
    """本地 OCR 模型实现"""

    def __init__(self, ocr_model_path: str, is_color: bool = True):
        if not ocr_model_path:
            raise ValueError("本地 OCR 模型路径不能为空")
        self.ocr_model = init_model(ocr_model_path, is_color=is_color)
        self.is_color = is_color

    def recognize(self, roi_image: np.ndarray, is_double_layer: bool = False) -> Tuple[str, Dict[str, Any]]:
        if roi_image is None or roi_image.size == 0:
            return "", {}

        if is_double_layer:
            roi_image = get_split_merge(roi_image)

        input_tensor = image_processing(roi_image)
        try:
            if self.is_color:
                preds, color_preds = self.ocr_model(input_tensor)
                color_preds = color_preds.softmax(dim=-1)
                color_conf, color_index = color_preds.max(dim=-1)
                color_conf = color_conf.item()
                color_index = color_index.item()
            else:
                preds = self.ocr_model(input_tensor)
                color_index = None
                color_conf = None

            preds = preds.softmax(dim=-1)
            prob, index = preds.max(dim=-1)
            index = index.view(-1).detach().cpu().numpy()
            prob = prob.view(-1).detach().cpu().numpy()

            new_preds, new_index = decodePlate(index)
            prob = prob[new_index]
            plate_chars = [plateName[i] for i in new_preds]
            plate_text = "".join(plate_chars)

            extra = {
                "char_prob": prob.tolist(),
            }
            if color_index is not None:
                extra.update({
                    "color": color[color_index],
                    "color_conf": color_conf,
                })

            return plate_text, extra
        except Exception as exc:  # pragma: no cover - 推理异常记录日志
            log_task_error(f"本地 OCR 推理失败: {exc}")
            return "", {}


class ModelScopeOCRProvider(BaseOCRProvider):
    """ModelScope 多模态 OCR 实现"""

    PROMPT = "仅输出车牌号"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: Optional[float] = 15.0):
        if not api_key:
            raise ValueError("ModelScope Access Token 未配置")
        if not model:
            raise ValueError("ModelScope 模型ID未配置")

        try:
            from openai import OpenAI  # 延迟导入，避免未安装 openai 包时影响本地 OCR
        except ImportError as exc:  # pragma: no cover - 环境依赖
            raise ImportError("缺少 openai 依赖，请先安装 openai>=1.0.0") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/")
        self.model = model
        self.timeout = timeout

    def recognize(self, roi_image: np.ndarray, is_double_layer: bool = False) -> Tuple[str, Dict[str, Any]]:
        if roi_image is None or roi_image.size == 0:
            return "", {}

        if is_double_layer:
            roi_image = get_split_merge(roi_image)

        rgb_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".png", rgb_image)
        if not success:
            log_task_error("ModelScope OCR: 图像编码失败")
            return "", {}

        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"

        try:
            client = self.client.with_options(timeout=self.timeout) if self.timeout else self.client
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "识别图中车牌"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0,
            )

            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and choice.message.content:
                text = choice.message.content.strip()
            else:
                text = ""

            log_task_debug(f"ModelScope OCR 返回结果: {text}")
            return text, {}
        except Exception as exc:  # pragma: no cover - 网络/服务异常
            log_task_error(f"ModelScope OCR 调用失败: {exc}")
            return "", {}


def create_ocr_provider(
    provider_name: str,
    *,
    ocr_model_path: Optional[str] = None,
    modelscope_conf: Optional[Dict[str, Any]] = None,
) -> BaseOCRProvider:
    provider = provider_name.lower()
    if provider == "local":
        return LocalOCRProvider(ocr_model_path)

    if provider == "modelscope":
        if modelscope_conf is None:
            raise ValueError("未提供 ModelScope 配置")
        api_key = modelscope_conf.get("api_key", "")
        base_url = modelscope_conf.get("base_url", "https://api-inference.modelscope.cn/v1/")
        model = modelscope_conf.get("model", "Qwen/Qwen3-VL-30B-A3B-Instruct")
        return ModelScopeOCRProvider(api_key=api_key, base_url=base_url, model=model)

    raise ValueError(f"未知的 OCR Provider: {provider_name}")

