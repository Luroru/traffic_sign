import torch
import os
import yaml
from pathlib import Path
import yolov5


def train_model(epochs=200, batch_size=8, img_size=640, device='0', cache=True, workers=16, model_size='m', half=True):
    """训练YOLOv5模型"""
    # CUDA相关优化
    if not torch.cuda.is_available():
        print("警告：未检测到可用的CUDA GPU，将使用CPU训练（这会很慢）")
        device = 'cpu'
    else:
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
        torch.cuda.empty_cache()  # 清理GPU缓存
    
 
  
    
    # 创建YOLO模型实例
    model_path = f'yolov5{model_size}.pt'  # 根据参数选择模型大小
    model = yolov5.load(model_path)  # 加载预训练模型
    
    # 训练模型
    results = model.train(
        data='dataset.yaml',      # 数据集配置文件
        epochs=epochs,            # 训练轮数
        imgsz=img_size,          # 输入图像大小
        batch_size=batch_size,    # 批次大小
        device=device,           # GPU设备号
        workers=workers,         # 数据加载进程
        project='runs/train',    # 保存训练结果的目录
        name='tt100k',          # 实验名称
        exist_ok=True,          # 如果实验目录存在则覆盖
        pretrained=True,        # 使用预训练权重

        #训练策略
        optimizer='AdamW',      # 使用AdamW优化器
        cos_lr=True,          # 使用余弦学习率调度
        amp=True,            # 使用混合精度训练
        patience=20,           # 早停机制（早停耐心值）
        seed=42,               # 设置随机种子(确保实验可重复性)

        # 数据增强方法
        augment=True,        # 使用数据增强
        mosaic=1.0,          # 马赛克增强
        #几何变换
        degrees=15.0,        # 旋转角度
        translate=0.2,       # 平移
        scale=0.5,          # 缩放
        shear=5.0,          # 剪切
        perspective=0.001,   # 透视变换
        #翻转增强
        flipud=0.5,         # 上下翻转概率
        fliplr=0.5,         # 左右翻转概率
        #HSV颜色空间增强
        hsv_h=0.02,         # HSV色调增强
        hsv_s=0.8,          # HSV饱和度增强
        hsv_v=0.5,           # HSV亮度增强

        verbose=True,           # 显示详细训练信息
        single_cls=False,      # 多类别模式
        rect=False,           # 关闭矩形训练
        resume=False,         # 不从断点继续训练
        cache=cache         # 是否缓存图像
    )
    
    return results


if __name__ == '__main__':
    # 设置环境变量以优化性能
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    
    # 开始训练
    print("\n=== 训练配置信息 ===")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"批次大小: 8")
    print(f"工作进程数: 16")
    print(f"模型大小: YOLOv5m")
    print(f"图像缓存: 启用")
    print(f"半精度训练: 启用")
    print("==================\n")
    
    results = train_model(
        epochs=200,
        batch_size=8,
        img_size=640,
        device='0',
        cache=True,
        workers=16,
        model_size='m',
        half=True
    )
    print("训练完成！") 