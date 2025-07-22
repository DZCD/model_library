from ultralytics.engine.results import Results
from ultralytics import YOLO

import torch
from .base_model import BaseModel
from .ocr_model import get_split_merge, image_processing, decodePlate, color, plateName, init_model


class PlateModel(BaseModel):
    def __init__(self, model_path, ocr_model_path):
        super().__init__(model_path)
        self.ocr_model = init_model(ocr_model_path, is_color=True)

    def post_process(self, results: Results) -> list:
        """提取Results中的推理结果数据，包括框的坐标、类别、置信度"""
        results_dict = []
        for result in results:
            ocr_data = self._plate_ocr_yolo(result)
            if not ocr_data:
                continue
            else:
                results_dict.append(ocr_data)
        return results_dict

    def _plate_ocr_yolo(self, yolo_result: Results):
        boxes_xyxy = yolo_result.boxes.xyxy.tolist()
        cls = yolo_result.boxes.cls.tolist()
        boxes_xywh = yolo_result.boxes.xywh.tolist()
        boxes_conf = yolo_result.boxes.conf.tolist()
        ori_image_arr = yolo_result.orig_img
        result_data = []
        for i, boxes in enumerate(boxes_xyxy):

            rect = [int(x) for x in boxes]
            # 先忽略双层的判断,影像前处理
            roi_img = ori_image_arr[rect[1]: rect[3], rect[0]: rect[2]]
            # 如果是双层要额外进行处理
            if int(cls[i]) == 1:
                roi_img = get_split_merge(roi_img)
            input = image_processing(roi_img)
            preds, color_preds = self.ocr_model(input)
            color_preds = torch.softmax(color_preds, dim=-1)
            color_conf, color_index = torch.max(color_preds, dim=-1)
            color_conf = color_conf.item()
            preds = torch.softmax(preds, dim=-1)
            prob, index = preds.max(dim=-1)
            index = index.view(-1).detach().cpu().numpy()
            prob = prob.view(-1).detach().cpu().numpy()

            # preds=preds.view(-1).detach().cpu().numpy()
            newPreds, new_index = decodePlate(index)
            prob = prob[new_index]
            plate = ""
            for str_i in newPreds:
                plate += plateName[str_i]
            x1, y1, x2, y2 = boxes
            hbb_points = [x1, y1, x2, y1, x2, y2, x1, y2]

            box_params = {
                "x": boxes_xywh[i][0],
                "y": boxes_xywh[i][1],
                "width": boxes_xywh[i][2],
                "height": boxes_xywh[i][3],
                "rotation": boxes_xywh[i][4] if len(boxes_xywh[i]) == 5 else 0,
                "score": boxes_conf[i],
                "xyxy": hbb_points,
                "track_id": "unknown",
                "classed": cls[i],
                "className": "license plate",
                "text": plate,
                "plate_status": True if len(plate) in [7, 8] else False,

            }
            result_data.append(box_params)
        # 将识别到的车牌位置以及字符串写入原图
        # plate_texts = [f"{c}:{p}" for c, p in zip(data["color"], data["plate"])]
        # plate_texts = [f"{p}" for p in ocr_data["plate"]]

        return result_data
