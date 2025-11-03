"""
车辆检测和可视化脚本
使用Reasoner进行车辆识别（模型索引5），并可视化OBB检测结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os
import sys
import math

# 添加项目根目录到Python路径

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_library.tools.reasoner import reasoner_single


class VehicleDetectionVisualizer:
    def __init__(self):
        """初始化车辆检测可视化器"""
        self.reasoner = reasoner_single
        self.model_index = 5  # 车辆检测模型索引
        
    def load_image(self, image_path):
        """
        加载图像（支持本地路径和URL）
        
        Args:
            image_path: 图像路径（本地文件或URL）
            
        Returns:
            image: OpenCV格式的图像
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                # 从URL下载图像
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                # 从本地路径加载图像
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"无法加载图像: {image_path}")
            return image
        except Exception as e:
            print(f"加载图像失败: {e}")
            return None
    
    async def detect_vehicles(self, image_path, conf_threshold=0.5, verbose=True):
        """
        检测车辆
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            verbose: 是否输出详细信息
            
        Returns:
            detections: 检测结果列表
        """
        try:
            print(f"正在检测车辆，置信度阈值: {conf_threshold}")
            detections = await self.reasoner.infer_image(
                image=image_path,
                model_index=self.model_index,
                conf=conf_threshold,
                verbose=verbose
            )
            print(f"检测完成，发现 {len(detections)} 个车辆")
            return detections
        except Exception as e:
            print(f"车辆检测失败: {e}")
            return []
    
    def draw_obb_detections(self, image, detections, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制OBB检测结果
        
        Args:
            image: 输入图像
            detections: OBB检测结果列表
            color: 绘制颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            annotated_image: 标注后的图像
        """
        annotated_image = image.copy()
        
        for i, detection in enumerate(detections):
            # 获取检测框信息
            center_x = float(detection['x'])
            center_y = float(detection['y'])
            width = float(detection['width'])
            height = float(detection['height'])
            rotation_radians = float(detection.get('rotation', 0))
            confidence = float(detection['score'])
            class_name = detection.get('className', 'vehicle')
            points = detection['xyxy']
            # 将弧度转换为度数用于显示和计算
            rotation_degrees = math.degrees(rotation_radians)
            
            # 简化绘制逻辑：根据rotation值判断格式
            if len(points) == 8:
                # rotation不为0，使用xyxyxyxy格式的8个顶点坐标
                box_points = np.array([[points[i], points[i+1]] for i in range(0, 8, 2)], dtype=np.int32)
                cv2.drawContours(annotated_image, [box_points], 0, color, thickness)
            else:
                # rotation为0，使用xyxy格式绘制普通矩形
                x1, y1, x2, y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制中心点
            cv2.circle(annotated_image, (int(center_x), int(center_y)), 4, color, -1)
            
            # 添加文本信息
            text_lines = []
            text_lines.append(f"{class_name}: {confidence:.2f}")
            if rotation_radians != 0:
                text_lines.append(f"∠{rotation_degrees:.1f}°")
            text_lines.append(f"#{i+1}")
            
            # 计算文本位置
            text_x = int(center_x - 60)
            text_y = int(center_y - 40)
            
            # 绘制背景框
            for j, line in enumerate(text_lines):
                y_offset = text_y + j * 20
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated_image, 
                             (text_x-2, y_offset-15), 
                             (text_x + text_size[0]+2, y_offset+5), 
                             (0, 0, 0), -1)
                cv2.putText(annotated_image, line, (text_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image
    
    def draw_hbb_detections(self, image, detections, color=(255, 0, 0), thickness=2):
        """
        绘制HBB检测结果（用于对比）
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            color: 绘制颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            annotated_image: 标注后的图像
        """
        annotated_image = image.copy()
        
        for i, detection in enumerate(detections):
            if 'xyxy' in detection:
                confidence = float(detection['score'])
                class_name = detection.get('className', 'vehicle')
                
                if len(detection['xyxy']) == 8:
                    # xyxyxyxy格式，取最小外接矩形来绘制HBB
                    points = detection['xyxy']
                    xs = [points[i] for i in range(0, 8, 2)]
                    ys = [points[i] for i in range(1, 8, 2)]
                    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                else:
                    # 传统xyxy格式
                    x1, y1, x2, y2 = map(int, detection['xyxy'])
                
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
                
                # 添加标签
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated_image
    
    def visualize_results(self, image, detections, save_path=None, show_comparison=True):
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            save_path: 保存路径（可选）
            show_comparison: 是否显示HBB和OBB对比
        """
        if show_comparison:
            # 创建对比图
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            
            # 原图
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image', fontsize=14)
            axes[0].axis('off')
            
            # HBB结果（如果有xyxy信息）
            hbb_image = self.draw_hbb_detections(image, detections, color=(255, 0, 0))
            axes[1].imshow(cv2.cvtColor(hbb_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title('HBB Detection (Red)', fontsize=14)
            axes[1].axis('off')
            
            # OBB结果
            obb_image = self.draw_obb_detections(image, detections, color=(0, 255, 0))
            axes[2].imshow(cv2.cvtColor(obb_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title('OBB Detection (Green)', fontsize=14)
            axes[2].axis('off')
            
        else:
            # 只显示OBB结果
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 原图
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=14)
            ax1.axis('off')
            
            # OBB结果
            obb_image = self.draw_obb_detections(image, detections, color=(0, 255, 0))
            ax2.imshow(cv2.cvtColor(obb_image, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'Vehicle Detection Results ({len(detections)} vehicles)', fontsize=14)
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
        
        # 打印检测统计信息
        self.print_detection_statistics(detections)
    
    def print_detection_statistics(self, detections):
        """
        打印检测结果的统计信息
        
        Args:
            detections: 检测结果列表
        """
        print("\n" + "="*70)
        print("车辆检测结果统计")
        print("="*70)
        print(f"检测到的车辆数量: {len(detections)}")
        
        if detections:
            confidences = [det['score'] for det in detections]
            angles_radians = [det.get('rotation', 0) for det in detections]
            angles_degrees = [math.degrees(angle) for angle in angles_radians]
            
            print(f"平均置信度: {np.mean(confidences):.3f}")
            print(f"置信度范围: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"旋转角度范围: {min(angles_degrees):.1f}° - {max(angles_degrees):.1f}°")
            print(f"平均旋转角度: {np.mean(angles_degrees):.1f}°")
            
            # 按置信度排序
            sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            
            print(f"\n{'序号':<4} {'类别':<12} {'置信度':<8} {'中心坐标':<15} {'尺寸':<15} {'角度':<8}")
            print("-" * 75)
            
            for i, detection in enumerate(sorted_detections):
                class_name = detection.get('className', 'vehicle')
                confidence = detection.get('score', 0)
                x = detection.get('x', 0)
                y = detection.get('y', 0)
                width = detection.get('width', 0)
                height = detection.get('height', 0)
                rotation_radians = detection.get('rotation', 0)
                rotation_degrees = math.degrees(rotation_radians)
                
                center = f"({x:.1f},{y:.1f})"
                size = f"{width:.1f}×{height:.1f}"
                
                print(f"{i+1:<4} {class_name:<12} {confidence:.3f} "
                      f"{center:<15} {size:<15} {rotation_degrees:>6.1f}°")
        else:
            print("未检测到任何车辆")
    
    async def detect_and_visualize(self, image_path, conf_threshold=0.3, save_path=None):
        """
        一键检测和可视化
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            save_path: 保存路径（可选）
        """
        print(f"开始处理图像: {image_path}")
        
        # 加载图像
        image = self.load_image(image_path)
        if image is None:
            return
        
        print(f"图像尺寸: {image.shape[1]} × {image.shape[0]}")
        
        # 检测车辆
        detections = await self.detect_vehicles(image_path, conf_threshold, verbose=False)
        
        if not detections:
            print("未检测到任何车辆")
            return
        
        # 可视化结果
        self.visualize_results(image, detections, save_path)
        
        return detections


async def main():
    """主函数 - 测试示例"""
    # 创建可视化器
    visualizer = VehicleDetectionVisualizer()
    
    # 测试图像路径（你可以修改这些路径）
    test_images = [
        # 网络图片示例
        # "http://113.105.137.161:10019/cloud-bucket/wayline/2C788E2E-E65B-399A-A628-CDFAAD256C45/DJI_202507211550_002_2C788E2E-E65B-399A-A628-CDFAAD256C45/DJI_20250721160245_0001_V.jpeg",
        
        # 本地图片示例（请修改为你的实际路径）
        # r"E:\项目\松山湖公安分局无人机自动巡检项目\图片识别测试\000.png",
        r"E:\项目\松山湖公安分局无人机自动巡检项目\图片识别测试\DJI_20250721160245_0001_V.jpeg",

        # 你可以添加更多测试图片
        # r"E:\your\image\path\vehicle_image.jpg",
    ]
    
    for i, image_path in enumerate(test_images):
        print(f"\n{'='*80}")
        print(f"处理第 {i+1} 张图片")
        print(f"{'='*80}")
        
        try:
            # 检测和可视化
            detections = await visualizer.detect_and_visualize(
                image_path=image_path,
                conf_threshold=0.3,  # 可以调整置信度阈值
                save_path=f"vehicle_detection_result_{i+1}.png"
            )
            
            if detections:
                print(f"处理完成，检测到 {len(detections)} 个车辆")
            
        except Exception as e:
            print(f"处理图片时出错: {e}")
            continue
    
    print("\n所有图片处理完成！")


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
