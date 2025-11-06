# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

- 以瞎猜接口为耻，以认真查询为荣。
- 以模糊执行为耻，以寻求确认为荣。
- 以臆想业务为耻，以人类确认为荣。
- 以创造接口为耻，以复用现有为荣。
- 以跳过验证为耻，以主动测试为荣。
- 以破坏架构为耻，以遵循规范为荣。
- 以假装理解为耻，以诚实无知为荣。
- 以盲目修改为耻，以谨慎重构为荣。

## Project Overview

This is an AI model inference system (南山消防智能检测系统) built with FastAPI that provides unified management and inference services for multiple AI models. The system supports real-time video stream processing, image inference, and integrates with MQTT and MinIO for result publishing and storage.

## Core Architecture

The system follows a modular architecture:

- **FastAPI Application** (`main.py`): Main web server with comprehensive API logging middleware
- **Model Management** (`model_library/model/`): Base classes and specialized model implementations
- **Inference Engines**:
  - `Detector` (`model_library/tools/detector.py`): Handles video stream processing with async task management
  - `Reasoner` (`model_library/tools/reasoner.py`): Handles single image inference (singleton pattern)
- **Client Integration**: MQTT for messaging, MinIO for object storage
- **Configuration Management**: Centralized YAML-based configuration

## Supported Models

The system supports 7 model types (indexed 0-6):
- 0: `elevator_motor` - Elevator motorcycle detection
- 1: `fire_lane_blockage` - Fire lane blockage detection (requires pixel_position)
- 2: `fire_detect` - Fire detection
- 3: `accident` - Accident detection (includes vehicle counting)
- 4: `license plate` - License plate detection + OCR
- 5: `car` - Vehicle detection
- 6: `infrared` - Infrared pedestrian detection

## Development Commands

### Running the Application
```bash
# Development server
python main.py

# Production with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:5122
```

### Docker Operations
```bash
# Build image
docker build -t ai-model-inference .

# Run container
docker run --gpus all -p 5122:5122 ai-model-inference
```

### Testing
```bash
# Run tests
python -m pytest test/

# Run specific test file
python test/test_rounter.py
python test/test_workflow.py
```

## Key Implementation Details

### Configuration System
- All models configured in `config.yaml` with paths, confidence thresholds, and specific parameters
- `Config` class in `model_library/tools/utils.py` provides centralized access
- Models can be configured to use original image size (`imgsz: 1`) or fixed size (`imgsz: 0`)

### Video Stream Processing
- `Detector` class manages long-running video inference tasks
- Each task runs in separate thread with async event loop
- Built-in stream health monitoring with automatic failure detection
- Supports graceful task termination via `request_stop()`

### Model Specializations
- **Accident Detection (Model 3)**: Automatically triggers vehicle detection (Model 5) for accident vehicle counting
- **Fire Lane Blockage (Model 1)**: Tracks object dwell time with configurable thresholds
- **License Plate (Model 4)**: Combines YOLO detection with OCR recognition

### API Design Patterns
- All APIs return consistent structure: `status`, `code`, `msg`, `data`
- Video inference returns task_id and mqtt_topic for async result monitoring
- Comprehensive logging middleware captures all requests/responses
- CORS enabled for cross-origin requests

### Error Handling & Monitoring
- Instance tracking for `Detector` objects to prevent memory leaks
- Automatic stream failure detection after 3 consecutive failures
- Task lifecycle management (running → stopping → completed/failed/stopped)
- Extensive logging through custom logger in `model_library/tools/logger.py`

## Model Integration

### Adding New Models
1. Create model class inheriting from `BaseModel` or implement specialized logic
2. Add model configuration to `config.yaml`
3. Update model loading logic in `detector.py:load_model()` and `reasoner.py:_load_model()`
4. Add any special post-processing logic in respective `post_process()` methods

### Model Post-Processing
- All models implement `post_process()` returning standardized format with bounding box coordinates
- OBB (Oriented Bounding Box) support for rotated objects
- Track ID management for object tracking across frames

## Important Notes

- System is designed for GPU inference but falls back to CPU automatically
- MQTT client uses unique client IDs to prevent connection conflicts
- MinIO integration automatically uploads inference results with structured naming
- Video processing uses frame skipping (vid_stride) to optimize performance
- Task cleanup is automatic but manual cleanup available via `/infer/cleanup` endpoint