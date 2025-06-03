import torch
import cv2
import numpy as np
import argparse
import os

# 全局模型变量
global_model = None

def detect_frame(frame, conf_thres=0.25, iou_thres=0.45):
    """
    对输入的单帧图像进行交通标志检测
    Args:
        frame: 输入的图像帧（numpy数组）
        conf_thres: 置信度阈值
        iou_thres: NMS IOU阈值
    Returns:
        processed_frame: 处理后的帧（带有标注的图像）
        labels_str: 检测结果字符串，格式：<class> <x1> <y1> <x2> <y2> <confidence>
    """
    global global_model
    
    # 加载模型（如果是第一次调用）
    if global_model is None:
        global_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                    path='runs/train/tt100k/weights/best.pt', 
                                    trust_repo=True)
    
    # 设置参数
    global_model.conf = conf_thres
    global_model.iou = iou_thres

    # 进行检测
    results = global_model(frame)
    
    # 获取处理后的帧
    processed_frame = results.render()[0]  # 获取标注后的图像
    
    # 构建结果字符串
    labels_str = ""
    if len(results.pred) > 0:
        for det in results.pred[0]:  # 只处理第一个图像的结果
            x1, y1, x2, y2, conf, cls = det.tolist()
            # 格式：<class> <x1> <y1> <x2> <y2> <confidence>
            labels_str += f"{int(cls)} {x1} {y1} {x2} {y2} {conf}\n"
    
    return processed_frame, labels_str

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用 YOLOv5 检测图片中的交通标志')
    parser.add_argument('--image', type=str, required=True, help='要检测的图片路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU 阈值 (默认: 0.45)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查图片是否存在
    if not os.path.exists(args.image):
        print(f"错误：图片 {args.image} 不存在")
        exit(1)
    
    # 读取图片
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误：无法读取图片 {args.image}")
        exit(1)
    
    # 使用detect_frame函数进行检测
    processed_frame, labels_str = detect_frame(image, args.conf, args.iou)
    
    # 打印检测结果字符串
    print(labels_str) 