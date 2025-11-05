
from importlib import import_module
import inspect
from model_library.tools.logger import log_task_debug, log_task_error
from model_library.tools.utils import Config

class ModelLoader:

    def __init__(self):
        """初始化配置"""
        self.config = Config()

    def load_model(self, model_index: int, task_id: str | None = None):
        try:
            # 读取配置并通过 class_path 反射式加载模型类
            model_cfg = self.config.model_list[model_index]
            model_path = model_cfg['model_path']
            class_path = model_cfg.get('class_path', 'model_library.model.base_model.BaseModel')
            log_task_debug(f"加载模型文件 - 任务ID:{task_id}, 路径:{model_path}, 类:{class_path}")

            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = import_module(module_name)
                model_class = getattr(module, class_name)
            except Exception as import_err:
                log_task_error(f"模型类导入失败 - 任务ID:{task_id}, 类:{class_path}, 错误:{str(import_err)}")
                raise

            # 根据构造函数参数名自动匹配传参（优先使用配置中的同名键）
            try:
                signature = inspect.signature(model_class.__init__)
                init_kwargs = {}
                for param_name, param in signature.parameters.items():
                    if param_name == 'self':
                        continue
                    if param_name in model_cfg:
                        init_kwargs[param_name] = model_cfg[param_name]

                # 确保 model_path 总是可用
                if 'model_path' not in init_kwargs:
                    init_kwargs['model_path'] = model_path

                model = model_class(**init_kwargs)
            except Exception as init_err:
                log_task_error(f"模型实例化失败 - 任务ID:{task_id}, 类:{class_path}, 错误:{str(init_err)}")
                raise

            log_task_debug(f"模型加载成功 - 任务ID:{task_id}")
            return model
        except Exception as e:
            log_task_error(f"模型加载失败 - 任务ID:{task_id}, 错误:{str(e)}")
            raise

model_loader = ModelLoader()