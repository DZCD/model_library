import json
import logging
import uuid
from typing import Any, Optional
import paho.mqtt.client as mqtt
from ..tools.utils import Config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MQTTClient:
    def __init__(self):
        self.cfg = Config()
        self.mqtt_config = self.cfg.mqtt
        
        # 创建MQTT客户端
        # self.client = mqtt.Client(client_id=self.mqtt_config.get("client-id", "model_library"))
        self.client = mqtt.Client(client_id=uuid.uuid4().hex)  #用uuid来作为客户端id，防止频繁连接中断的问题
        
        # 设置用户名和密码
        if self.mqtt_config.get("username") and self.mqtt_config.get("password"):
            self.client.username_pw_set(
                username=self.mqtt_config["username"],
                password=self.mqtt_config["password"]
            )
        
        # 设置回调函数
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        self.is_connected = False
        
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调函数"""
        if rc == 0:
            self.is_connected = True
            logger.info(f"MQTT客户端连接成功，连接到 {self.mqtt_config['host']}:{self.mqtt_config['port']}")
        else:
            self.is_connected = False
            logger.error(f"MQTT客户端连接失败，错误代码: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调函数"""
        self.is_connected = False
        logger.info("MQTT客户端断开连接")
    
    def _on_publish(self, client, userdata, mid):
        """发布消息回调函数"""
        logger.debug(f"消息发布成功，消息ID: {mid}")
    
    def connect(self) -> bool:
        """连接到MQTT服务器"""
        try:
            self.client.connect(
                host=self.mqtt_config["host"],
                port=self.mqtt_config["port"],
                keepalive=60
            )
            self.client.loop_start()  # 启动网络循环
            return True
        except Exception as e:
            logger.error(f"连接MQTT服务器失败: {e}")
            return False
    
    def disconnect(self):
        """断开MQTT连接"""
        self.client.loop_stop()
        self.client.disconnect()
        self.is_connected = False
    
    def publish_message(self, topic: str, message: Any, qos: int = 0, retain: bool = False) -> bool:
        """
        发布消息到MQTT主题
        
        Args:
            topic (str): 消息主题
            message (Any): 要发送的消息内容，可以是字符串、字典或其他可序列化的对象
            qos (int): 服务质量等级 (0, 1, 2)
            retain (bool): 是否保留消息
            
        Returns:
            bool: 发布是否成功
        """
        try:
            # 如果没有连接，尝试连接
            if not self.is_connected:
                if not self.connect():
                    logger.error("无法连接到MQTT服务器，消息发布失败")
                    return False
            
            # 将消息转换为JSON字符串（如果不是字符串的话）
            if isinstance(message, str):
                payload = message
            else:
                payload = json.dumps(message, ensure_ascii=False)
            
            # 发布消息
            result = self.client.publish(topic, payload, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"消息发布成功到主题 '{topic}': {payload}")
                return True
            else:
                logger.error(f"消息发布失败，错误代码: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"发布消息时发生错误: {e}")
            return False
    
    def publish_detection_result(self, model_name: str, detection_results: dict, camera_id: Optional[str] = None):
        """
        发布检测结果的便捷方法
        
        Args:
            model_name (str): 模型名称
            detection_results (dict): 检测结果
            camera_id (str, optional): 摄像头ID
        """
        topic = f"nanshan/firefighting/{model_name}"
        if camera_id:
            topic += f"/{camera_id}"
        
        message = {
            "model": model_name,
            "timestamp": detection_results.get("timestamp"),
            "camera_id": camera_id,
            "results": detection_results
        }
        
        return self.publish_message(topic, message)
        
