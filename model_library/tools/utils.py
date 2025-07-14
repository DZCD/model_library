"""工具脚本"""

import yaml 

class Config:
    def __init__(self):
        self.config = self.load_config()
        self.model_list = self.config['model']
        self.minio = self.config['minio']
        self.mqtt = self.config['mqtt']

    def load_config(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    

