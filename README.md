# 模型算法管理库 (Model Library)

这是一个通用的模型算法管理库，提供统一的模型推理服务接口。支持多种AI模型的加载、管理和推理，包括目标检测、追踪、OCR等功能。

## 功能特点

- 🚀 **统一接口**: 提供RESTful API接口，统一管理多种AI模型
- 🔄 **模型管理**: 支持多种模型类型的动态加载和配置
- 🎯 **实时推理**: 支持图像和视频流的实时推理
- 📊 **异步处理**: 基于FastAPI的异步任务处理
- 🔗 **消息推送**: 集成MQTT客户端，支持推理结果实时推送
- 💾 **存储服务**: 集成MinIO对象存储，自动保存推理结果
- 🎮 **任务控制**: 支持任务的启动、停止和状态查询
- 🌐 **跨域支持**: 支持前端跨域访问

## 支持的模型类型

| 模型索引 | 模型名称 | 功能描述 | 权重文件 |
|---------|---------|---------|---------|
| 0 | elevator_motor | 电梯摩托车检测 | elevator_motor.pt |
| 1 | fire_lane_blockage | 消防通道占用检测 | fire_lane_blockage.pt |
| 2 | fire_detect | 火点检测 | fire_detect.pt |
| 3 | accident | 事故检测 | yolo_accident_obb.pt |
| 4 | license_plate | 车牌检测+OCR | yolo_license_plate.pt + plate_rec_color.pth |
| 5 | car | 车辆检测 | yolo11n.pt |

## 系统架构

```
model_library/
├── config.yaml          # 全局配置文件
├── main.py              # FastAPI主程序
├── model_library/       # 核心模块
│   ├── client/          # 客户端模块
│   │   ├── minio_client.py    # MinIO存储客户端
│   │   └── mqtt_client.py     # MQTT消息客户端
│   ├── model/           # 模型模块
│   │   ├── base_model.py      # 基础模型类
│   │   ├── ocr_model.py       # OCR模型
│   │   ├── plate_model.py     # 车牌识别模型
│   │   ├── track_accident.py  # 事故检测模型
│   │   └── track_fireland.py  # 消防通道检测模型
│   ├── router/          # API路由
│   │   ├── config.py          # 配置管理API
│   │   └── infer.py           # 推理服务API
│   └── tools/           # 工具模块
│       ├── detector.py        # 检测器（视频流处理）
│       ├── reasoner.py        # 推理器（图像处理）
│       └── utils.py           # 工具函数
├── test/                # 测试模块
│   └── test_workflow.py       # 工作流测试
└── weight/              # 模型权重文件
    ├── elevator_motor.pt
    ├── fire_detect.pt
    ├── fire_lane_blockage.pt
    ├── plate_rec_color.pth
    ├── yolo_accident_obb.pt
    ├── yolo_license_plate.pt
    └── yolo11n.pt
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn ultralytics opencv-python PyYAML paho-mqtt minio pillow
```

### 2. 配置文件

在 `config.yaml` 中配置模型参数、MQTT和MinIO连接信息：

```yaml
model:
  0:
    model_name: "elevator_motor"
    model_path: "weight/elevator_motor.pt"
    classes: [0]
  1:
    model_name: "fire_lane_blockage"
    model_path: "weight/fire_lane_blockage.pt"
    time_threshold: 60
    classes: [0]
  # ... 更多模型配置

mqtt:
  host: "your-mqtt-host"
  port: 1884
  username: "your-username"
  password: "your-password"

minio:
  endpoint: "your-minio-endpoint"
  port: 9000
  access-key: "your-access-key"
  secret-key: "your-secret-key"
  bucket: "your-bucket"
```

### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:5122` 启动，可以访问 `http://localhost:5122/docs` 查看API文档。

## API 接口

### 1. 视频流推理

**POST** `/ai_model/infer/video`

启动视频流推理任务：

```bash
curl -X POST "http://localhost:5122/ai_model/infer/video" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "video_path=rtmp://your-stream-url&model_index=1&pixel_position=[[0,0],[100,0],[100,100],[0,100]]"
```

### 2. 图像推理

**POST** `/ai_model/infer/image`

对单张图像进行推理：

```bash
curl -X POST "http://localhost:5122/ai_model/infer/image" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your-image.jpg" \
  -F "model_index=0"
```

### 3. 任务管理

**GET** `/ai_model/infer/status/{task_id}`

查询任务状态：

```bash
curl -X GET "http://localhost:5122/ai_model/infer/status/your-task-id"
```

**DELETE** `/ai_model/infer/stop/{task_id}`

停止推理任务：

```bash
curl -X DELETE "http://localhost:5122/ai_model/infer/stop/your-task-id"
```

### 4. 配置管理

**GET** `/ai_model/config`

获取系统配置信息：

```bash
curl -X GET "http://localhost:5122/ai_model/config"
```

## 使用示例

### Python客户端示例

```python
import requests
import json

# 启动视频流推理
response = requests.post(
    "http://localhost:5122/ai_model/infer/video",
    data={
        "video_path": "rtmp://your-stream-url",
        "model_index": 1,
        "pixel_position": json.dumps([[0, 0], [100, 0], [100, 100], [0, 100]])
    }
)

if response.status_code == 200:
    result = response.json()
    task_id = result["data"]["task_id"]
    mqtt_topic = result["data"]["mqtt_topic"]
    print(f"任务ID: {task_id}")
    print(f"MQTT主题: {mqtt_topic}")
```

### JavaScript客户端示例

```javascript
// 启动视频流推理
const formData = new FormData();
formData.append('video_path', 'rtmp://your-stream-url');
formData.append('model_index', '1');
formData.append('pixel_position', JSON.stringify([[0, 0], [100, 0], [100, 100], [0, 100]]));

fetch('http://localhost:5122/ai_model/infer/video', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('任务ID:', data.data.task_id);
    console.log('MQTT主题:', data.data.mqtt_topic);
});
```

## 特色功能

### 1. 异步任务处理
- 支持长时间运行的视频流推理任务
- 任务状态实时查询和控制
- 优雅的任务停止机制

### 2. 消息推送机制
- 推理结果自动推送到MQTT主题
- 支持自定义主题命名规则
- 结构化的消息格式

### 3. 存储集成
- 自动保存推理结果图像到MinIO
- 规范化的文件命名和路径结构
- 支持多种图像格式

### 4. 模型热加载
- 支持配置文件动态更新
- 无需重启服务即可加载新模型
- 灵活的模型参数配置

## 扩展开发

### 添加新模型

1. 在 `model_library/model/` 目录下创建新的模型类
2. 继承 `BaseModel` 或实现相应接口
3. 在 `config.yaml` 中添加模型配置
4. 在 `tools/detector.py` 和 `tools/reasoner.py` 中添加模型加载逻辑

### 自定义后处理

```python
class CustomModel(BaseModel):
    def post_process(self, results):
        # 自定义后处理逻辑
        processed_results = []
        for result in results:
            # 处理检测结果
            processed_results.append(processed_result)
        return processed_results
```

## 部署说明

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5122

CMD ["python", "main.py"]
```

### 生产环境配置

```bash
# 使用Gunicorn部署
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:5122
```

## 注意事项

1. **模型文件**: 确保所有模型权重文件已正确放置在 `weight/` 目录下
2. **依赖服务**: 确保MQTT和MinIO服务正常运行
3. **资源管理**: 长时间运行的视频流任务会占用较多资源，建议监控系统性能
4. **网络配置**: 确保能够访问视频流地址和外部服务
5. **权限配置**: 确保应用有足够的文件读写权限

## 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目。在提交代码前，请确保：

1. 代码符合项目的编码规范
2. 添加必要的测试用例
3. 更新相关文档

