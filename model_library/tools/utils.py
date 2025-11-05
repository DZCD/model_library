"""工具脚本"""

import yaml 

class Config:

    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置"""
        if hasattr(self, '_initialized') and self._initialized:
            return  # 避免重复初始化
        self.config = self.load_config()
        self.model_list = self.config['model']
        self.minio = self.config['minio']
        self.mqtt = self.config['mqtt']

        self._initialized = True

    def load_config(self):
        """加载配置"""
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    
