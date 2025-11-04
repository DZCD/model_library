from io import BytesIO
import requests
import json
from datetime import datetime
from PIL import Image
from PIL import ImageDraw

save_img_path = f"detect_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
image_url = "https://img0.baidu.com/it/u=4206992412,806806134&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=675"


def load_image(img_url) -> Image.Image:
    """加载原始图片，不进行预处理"""
    resp = requests.get(img_url)
    img_pil = Image.open(BytesIO(resp.content))
    return img_pil

def draw_detections(img: Image.Image, items: list):
    """绘制检测结果：优先使用 xyxy，兼容 OBB/普通框；否则从中心点 xywh 换算。"""
    draw = ImageDraw.Draw(img)

    for item in items:
        # 优先使用 xyxy（可能是 8 个数，也可能是 [8 个数] 的套壳）
        xyxy = item.get("xyxy", [])
        points = None
        if isinstance(xyxy, list):
            if len(xyxy) == 8 and all(isinstance(v, (int, float)) for v in xyxy):
                flat = xyxy
                points = [(flat[i], flat[i + 1]) for i in range(0, 8, 2)]
            elif len(xyxy) == 1 and isinstance(xyxy[0], list) and len(xyxy[0]) == 8:
                flat = xyxy[0]
                points = [(flat[i], flat[i + 1]) for i in range(0, 8, 2)]

        if points is not None:
            draw.polygon(points, outline=(255, 0, 0), width=2)
            print(f"检测到: {item.get('className')}, 置信度: {item.get('score'):.3f}")
            print(f"多边形点: {points}")
        else:
            # 从中心点坐标换算到左上-右下
            x, y = item.get("x"), item.get("y")
            width, height = item.get("width"), item.get("height")
            if x is not None and y is not None and width is not None and height is not None:
                x1, y1 = x - width / 2, y - height / 2
                x2, y2 = x + width / 2, y + height / 2
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                print(f"检测到: {item.get('className')}, 置信度: {item.get('score'):.3f}")
                print(f"矩形框(由中心点换算): x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # 其他信息
        score = item.get("score")
        track_id = item.get("track_id")
        class_name = item.get("className")
        text = item.get("text")
        print(f"详细信息 - score: {score}, track_id: {track_id}, className: {class_name}, text: {text}")
        print("---")

def save_img(img: Image.Image, save_img_path: str):
    img.save(save_img_path)
    print(f"结果已保存到: {save_img_path}")

if __name__ == "__main__":
    # 加载原始图片
    img = load_image(image_url)
    print(f"原始图片尺寸: {img.size}")

    # 调用API进行推理
    url = "http://127.0.0.1:5122/ai_model/infer/image"

    data = {
        "image_path": image_url,
        "model_index": "7"  # 修改为有效的模型索引
    }

    print("正在调用API进行推理...")
    resp = requests.post(url, data=data)

    if resp.status_code == 200:
        resp_json = resp.json()
        if resp_json["code"] == 200:
            items = resp_json["data"]["item"]
            print(f"检测到 {len(items)} 个目标")

            # 绘制检测结果
            draw_detections(img, items)

            # 保存结果图片
            save_img(img=img, save_img_path=save_img_path)
        else:
            print(f"API返回错误: {resp_json['msg']}")
    else:
        print(f"请求失败: {resp.text}")
        
