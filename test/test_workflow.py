import os
import sys
import unittest
import asyncio


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        # 确定主目录（main.py所在的目录）
        self.project_root = os.path.dirname(os.path.dirname(__file__))

        # 保存当前工作目录
        self.original_cwd = os.getcwd()

        # 更改工作目录为项目根目录
        os.chdir(self.project_root)

        # 将项目根目录添加到Python路径
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

    def tearDown(self):
        # 恢复原始工作目录
        os.chdir(self.original_cwd)

        # 移除添加的路径
        if self.project_root in sys.path:
            sys.path.remove(self.project_root)

    def test_workflow_model1(self):
        """测试模型1（消防通道占用检测）的workflow"""
        from model_library.tools.detector import Detector

        # 定义测试区域
        polygon_points = [
            (0, 941),  # 左下角
            (0, 1342),  # 最下角
            (2152, 1338),  # 右下角
            (2173, 586),  # 右上角
            (1110, 460),  # 左上角
        ]
        video_path = r"rtmp://113.105.137.153:1935/rtp/34020000001110000001_34020000001320000404"
        # video_path = r"rtmp://10.1.38.245:1935/live/raw_stream3"

        # 创建Workflow实例
        workflow = Detector(
            model_index=1,
            video_path=video_path,  # 请提供实际的视频路径
            pixel_position=polygon_points
        )

        # 运行测试
        asyncio.run(workflow.run_video())

    def test_workflow_model0(self):
        """测试模型0（电梯摩托车检测）的workflow"""
        from model_library.tools.detector import Detector
        video_path = r"rtmp://10.1.38.245:1935/live/raw_stream3"
        # video_path = r"E:\项目\深圳南山智慧消防\电梯电动车\测试视频\测试视频.mp4"

        # 创建Workflow实例
        workflow = Detector(
            model_index=0,
            video_path=video_path,  # 请提供实际的视频路径
            task_id="111"
        )
        topic = workflow.topic
        print(topic)

        # 运行测试
        asyncio.run(workflow.run_video())

    def test_workflow_model2(self):
        """测试模型2（火点）的workflow"""
        from model_library.tools.detector import Detector
        # video_path = "E:\项目\深圳南山智慧消防\消防占用\sjlj\D49_20250531233830.mp4"
        video_path = r"E:\项目\深圳南山智慧消防\烟雾明火\测试视频\明火视频.mp4"

        # 创建Workflow实例
        workflow = Detector(
            model_index=2,
            video_path=video_path,  # 请提供实际的视频路径
            task_id="111"
        )
        topic = workflow.topic
        print(topic)

        # 运行测试
        asyncio.run(workflow.run_video())

    def test_workflow_model3(self):
        """测试模型3。事故检测"""
        from model_library.tools.detector import Detector

        video_path = r"rtmp://10.1.38.245:1935/live/raw_stream3"
        # video_path = r"rtsp://10.5.52.56:554/live/dji?callId=26-173"
        # video_path = r"rtmp://10.5.52.56:1935/live/7CTDM7T00BR29S?callId=26-173"

        # 创建Workflow实例
        workflow = Detector(
            model_index=3,
            video_path=video_path,  # 请提供实际的视频路径
            task_id= "111"
        )
        topic = workflow.topic
        print(topic)

        # 运行测试
        asyncio.run(workflow.run_video())

    def test_workflow_image(self):
        from model_library.tools.reasoner import Reasoner
        reasoner = Reasoner()
        image_path = r"E:\项目\松山湖公安分局无人机自动巡检项目\图片识别测试\DJI_20250721160245_0001_V.jpeg"
        # image_path = r"http://10.1.38.127:8000/2025-04-14_00-00-00-416.jpg"
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(reasoner.infer_image(image_path,3,verbose=True))
        print(result)




