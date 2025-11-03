#!/usr/bin/env python3
"""
配置管理系统学习脚本
用于测试和理解配置加载逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_library.tools.utils import Config

def test_config_loading():
    """测试配置加载功能"""
    print("=" * 50)
    print("1. 测试配置基本加载")
    print("=" * 50)

    # 创建配置实例
    config = Config()

    # 检查配置结构
    print(f"配置对象类型: {type(config)}")
    print(f"原始配置类型: {type(config.config)}")
    print(f"模型列表类型: {type(config.model_list)}")
    print(f"MQTT配置类型: {type(config.mqtt)}")
    print(f"MinIO配置类型: {type(config.minio)}")

    print("\n" + "=" * 50)
    print("2. 模型配置详情")
    print("=" * 50)

    # 遍历所有模型配置
    for model_id, model_config in config.model_list.items():
        print(f"\n模型 {model_id}:")
        print(f"  - 名称: {model_config.get('model_name', 'N/A')}")
        print(f"  - 路径: {model_config.get('model_path', 'N/A')}")
        print(f"  - 类别: {model_config.get('classes', 'N/A')}")
        print(f"  - 图像尺寸: {model_config.get('imgsz', 'N/A')}")

        # 特殊配置
        if 'config' in model_config:
            print(f"  - 置信度: {model_config['config']}")
        if 'time_threshold' in model_config:
            print(f"  - 时间阈值: {model_config['time_threshold']}秒")
        if 'time_step' in model_config:
            print(f"  - 推送间隔: {model_config['time_step']}秒")
        if 'ocr_model_path' in model_config:
            print(f"  - OCR模型: {model_config['ocr_model_path']}")

    print("\n" + "=" * 50)
    print("3. 外部服务配置")
    print("=" * 50)

    # MQTT配置
    print("MQTT配置:")
    for key, value in config.mqtt.items():
        # 隐藏敏感信息
        if 'password' in key.lower():
            value = '*' * len(str(value))
        print(f"  - {key}: {value}")

    # MinIO配置
    print("\nMinIO配置:")
    for key, value in config.minio.items():
        # 隐藏敏感信息
        if 'secret' in key.lower() or 'access' in key.lower():
            value = '*' * len(str(value))[:4] + str(value)[4:]
        print(f"  - {key}: {value}")

def test_accident_config():
    """专门测试事故检测配置"""
    print("\n" + "=" * 50)
    print("4. 事故检测配置专项测试")
    print("=" * 50)

    config = Config()
    accident_config = config.model_list[3]

    print("事故检测模型配置详情:")
    for key, value in accident_config.items():
        print(f"  - {key}: {value}")

    # 模拟在代码中如何使用
    print("\n模拟代码使用:")
    model_index = 3
    model_name = config.model_list[model_index]['model_name']
    model_path = config.model_list[model_index]['model_path']
    confidence = config.model_list[model_index].get('config', 0.5)
    time_step = config.model_list[model_index].get('time_step', 60)

    print(f"  model_name = '{model_name}'")
    print(f"  model_path = '{model_path}'")
    print(f"  confidence = {confidence}")
    print(f"  time_step = {time_step}")

def test_config_validation():
    """测试配置有效性"""
    print("\n" + "=" * 50)
    print("5. 配置有效性验证")
    print("=" * 50)

    config = Config()

    # 检查模型文件是否存在
    print("检查模型文件:")
    for model_id, model_config in config.model_list.items():
        model_path = model_config.get('model_path', '')
        if model_path and os.path.exists(model_path):
            print(f"  ✓ 模型 {model_id}: {model_path}")
        else:
            print(f"  ✗ 模型 {model_id}: {model_path} (文件不存在)")

    # 检查权重目录
    weight_dir = 'weight'
    if os.path.exists(weight_dir):
        print(f"\n权重目录存在: {weight_dir}")
        files = os.listdir(weight_dir)
        print(f"包含 {len(files)} 个文件:")
        for f in files[:5]:  # 只显示前5个
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... 还有 {len(files) - 5} 个文件")
    else:
        print(f"\n权重目录不存在: {weight_dir}")

if __name__ == "__main__":
    try:
        test_config_loading()
        test_accident_config()
        test_config_validation()

        print("\n" + "=" * 50)
        print("✓ 配置系统测试完成")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()