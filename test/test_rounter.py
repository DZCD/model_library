import os
import sys
import asyncio
import unittest


class TestRouter(unittest.TestCase):

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.original_cwd = os.getcwd()
        os.chdir(self.project_root)

        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

    def tearDown(self):
        os.chdir(self.original_cwd)
        if self.project_root  in sys.path:
            sys.path.remove(self.project_root)

    def test_router_video(self):
        import threading
        import time

        from model_library.router.infer import run_workflow_in_thread
        from model_library.tools.detector import Detector


        model_index = 3
        video_path =  r"rtmp://10.1.38.245:1935/live/raw_stream3"
        workflow = Detector(
            model_index=model_index,
            video_path=video_path,
            task_id="111"
        )
        thread = threading.Thread(target=run_workflow_in_thread, args=(workflow, "111"))
        thread.start()
        time.sleep(2000)


