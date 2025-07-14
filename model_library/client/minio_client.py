import urllib3
import io
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from minio import Minio
from minio.error import S3Error
import logging
from ..tools.utils import Config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

timeout_config = urllib3.Timeout(connect=5, read=15)  # 连接超时 5 秒  # 读取超时 15 秒
http_client = urllib3.PoolManager(timeout=timeout_config)

class MinioClient:
    def __init__(self):
        self.cfg = Config()
        
        self.client = Minio(
            endpoint=f'{self.cfg.minio["endpoint"]}:{self.cfg.minio["port"]}',
            access_key=self.cfg.minio["access-key"],
            secret_key=self.cfg.minio["secret-key"],
            secure=False,
            # 设置连接超时和请求超时（单位：秒）
            http_client=http_client,
        )
        self.bucket = self.cfg.minio["bucket"]
        self.check_bucket_exist()
    
    def check_bucket_exist(self):
        """检查存储桶是否存在，如果不存在则创建"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"创建存储桶: {self.bucket}")
            else:
                logger.info(f"存储桶已存在: {self.bucket}")
        except S3Error as e:
            logger.error(f"检查存储桶时发生错误: {e}")
    
    def upload_file(self, object_name: str, file_path: str, content_type: str = None):
        """
        上传文件到Minio
        
        Args:
            object_name (str): 对象名称（存储在Minio中的路径）
            file_path (str): 本地文件路径
            content_type (str): 文件内容类型
        
        Returns:
            bool: 上传是否成功
        """
        try:
            self.client.fput_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type
            )
            logger.info(f"文件上传成功: {object_name}")
            return True
        except S3Error as e:
            logger.error(f"文件上传失败: {e}")
            return False
    
    def download_file(self, object_name: str, file_path: str):
        """
        从Minio下载文件
        
        Args:
            object_name (str): 对象名称
            file_path (str): 本地保存路径
        
        Returns:
            bool: 下载是否成功
        """
        try:
            self.client.fget_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=file_path
            )
            logger.info(f"文件下载成功: {object_name}")
            return True
        except S3Error as e:
            logger.error(f"文件下载失败: {e}")
            return False
    
    def delete_file(self, object_name: str):
        """
        删除Minio中的文件
        
        Args:
            object_name (str): 对象名称
        
        Returns:
            bool: 删除是否成功
        """
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.info(f"文件删除成功: {object_name}")
            return True
        except S3Error as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def upload_image_array(self, image_array: np.ndarray, object_name: str = None, 
                          image_format: str = 'jpg', quality: int = 95, input_format: str = 'BGR'):
        """
        上传numpy数组图像到Minio
        
        Args:
            image_array (np.ndarray): 图像数组
            object_name (str): 对象名称，如果为None则自动生成时间戳命名
            image_format (str): 图像格式 ('jpg', 'png')
            quality (int): JPEG质量 (1-100)
            input_format (str): 输入图像格式 ('RGB', 'BGR')
        
        Returns:
            tuple: (bool, str) 上传是否成功和对象名称
        """
        try:
            # 如果没有指定对象名称，使用时间戳生成
            if object_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                object_name = f"images/{timestamp}.{image_format}"
            
            # 确保图像数组是正确的格式
            if len(image_array.shape) != 3 or image_array.shape[2] != 3:
                raise ValueError("图像数组必须是3通道格式 (height, width, 3)")
            
            # 确保数据类型为uint8
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            
            # 根据输入格式处理图像
            if input_format.upper() == 'BGR':
                # BGR转RGB
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image, mode='RGB')
            elif input_format.upper() == 'RGB':
                # 直接使用RGB
                pil_image = Image.fromarray(image_array, mode='RGB')
            else:
                raise ValueError(f"不支持的输入格式: {input_format}，支持的格式: 'RGB', 'BGR'")
            
            # 创建字节流
            image_bytes = io.BytesIO()
            
            # 编码图像为字节流，保持RGB格式
            if image_format.lower() in ['jpg', 'jpeg']:
                pil_image.save(image_bytes, format='JPEG', quality=quality, optimize=True)
                content_type = 'image/jpeg'
            elif image_format.lower() == 'png':
                pil_image.save(image_bytes, format='PNG', optimize=True)
                content_type = 'image/png'
            else:
                raise ValueError(f"不支持的图像格式: {image_format}")
            
            # 重置字节流位置
            image_bytes.seek(0)
            
            # 上传到Minio
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_name,
                data=image_bytes,
                length=image_bytes.getvalue().__len__(),
                content_type=content_type
            )
            
            logger.info(f"图像数组上传成功: {object_name}")
            return True, object_name
            
        except Exception as e:
            logger.error(f"图像数组上传失败: {e}")
            return False, None
    
    def upload_detection_image(self, image_array: np.ndarray, model_name: str, 
                              camera_id: str = None, detection_id: str = None, input_format: str = 'RGB'):
        """
        上传检测相关的图像，使用规范的命名格式
        
        Args:
            image_array (np.ndarray): 图像数组
            model_name (str): 模型名称
            camera_id (str): 摄像头ID
            detection_id (str): 检测ID
            input_format (str): 输入图像格式 ('RGB', 'BGR')
        
        Returns:
            tuple: (bool, str) 上传是否成功和对象名称
        """
        try:
            # 生成规范的对象名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            path_parts = ["detections", model_name]
            if camera_id:
                path_parts.append(camera_id)
            
            filename_parts = [timestamp]
            if detection_id:
                filename_parts.append(detection_id)
            
            object_name = "/".join(path_parts) + "/" + "_".join(filename_parts) + ".jpg"
            
            return self.upload_image_array(image_array, object_name, input_format=input_format)
            
        except Exception as e:
            logger.error(f"检测图像上传失败: {e}")
            return False, None
