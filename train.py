from ultralytics import YOLO
import os
import argparse
import torch

def train_model(epochs=100, batch_size=4, img_size=640, device='0', cache=True, workers=16, model_size='m', half=True):
    """训练YOLOv8模型"""
    # CUDA相关优化
    if not torch.cuda.is_available():
        print("警告：未检测到可用的CUDA GPU，将使用CPU训练（这会很慢）")
        device = 'cpu'
    else:
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
        torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 创建YOLO模型实例
    model_path = f'yolov8{model_size}.pt'  # 根据参数选择模型大小
    model = YOLO(model_path)  # 使用预训练的YOLOv8模型
    
    # 训练模型
    results = model.train(
        data='dataset.yaml',      # 数据集配置文件
        epochs=200,            # 训练轮数：增加训练轮次
        imgsz=800,          # 输入图像大小：增大以获取更清晰特征
        batch=4,        # 批次大小：适当减小维持显存
        device=device,           # GPU设备号
        workers=4,         # 数据加载进程：适度以防过载
        patience=100,           # 早停：增加耐心值
        save=True,               # 保存训练结果
        project='runs/train',    # 保存训练结果的目录
        name='tt100k',          # 实验名称
        exist_ok=True,          # 如果实验目录存在则覆盖
        pretrained=True,        # 使用预训练权重
        optimizer='AdamW',      # 使用AdamW优化器
        verbose=True,           # 显示详细训练信息
        seed=42,               # 设置随机种子
        deterministic=True,   # 开启确定性保证复现
        single_cls=False,      # 多类别模式
        rect=False,           # 关闭矩形训练提升泛化
        cos_lr=True,          # 使用余弦学习率调度
        close_mosaic=20,      # 更多epoch后关闭马赛克增强
        resume=False,         # 不从断点继续训练
        amp=True,             # 使用混合精度训练
        fraction=1.0,         # 使用全部训练数据
        cache=cache,          # 是否缓存图像到内存
        overlap_mask=True,    # 允许掩码重叠
        mask_ratio=4,         # 掩码下采样率
        dropout=0.2,          # 使用dropout防止过拟合
        val=True,             # 训练时进行验证
        plots=True,           # 保存训练过程图
        # 增强数据增强强度
        augment=True,         # 使用数据增强
        mosaic=1.0,           # 马赛克增强频率最大化
        mixup=0.1,            # MixUp增强
        copy_paste=0.2,       # 复制粘贴增强
        degrees=15.0,         # 更强的旋转
        translate=0.2,        # 更强的平移
        scale=0.5,            # 更强的缩放
        shear=5.0,            # 激活剪切增强
        perspective=0.001,    # 少量透视变换
        flipud=0.5,           # 上下翻转概率
        fliplr=0.5,           # 左右翻转概率
        hsv_h=0.02,           # HSV色调增强
        hsv_s=0.8,            # HSV饱和度增强
        hsv_v=0.5             # HSV亮度增强
    )
    
    return results

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练交通标志检测模型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小，根据GPU显存调整')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--device', type=str, default='0', help='训练设备，使用CPU请设置为cpu')
    parser.add_argument('--cache', action='store_true', default=True, help='是否缓存图像到内存')
    parser.add_argument('--workers', type=int, default=16, help='数据加载的工作进程数，设为CPU逻辑处理器数量')
    parser.add_argument('--model-size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'], 
                        help='模型大小: n(nano), s(small), m(medium), l(large), x(xlarge)')
    
    # 优化相关参数
    parser.add_argument('--half', action='store_true', default=True, help='使用FP16半精度训练以提升速度')
    parser.add_argument('--img-size-test', type=int, default=None, help='测试时的图像尺寸，默认与训练尺寸相同')
    
    # 高级优化参数
    parser.add_argument('--multi-scale', action='store_true', help='使用多尺度训练')
    parser.add_argument('--adam', action='store_true', help='使用Adam优化器')
    parser.add_argument('--sync-bn', action='store_true', help='使用同步批归一化')
    parser.add_argument('--freeze', type=int, default=0, help='冻结前N层')
    
    args = parser.parse_args()
    
    # 设置环境变量以优化性能
    os.environ['OMP_NUM_THREADS'] = '16'  # OpenMP线程数
    os.environ['MKL_NUM_THREADS'] = '16'  # MKL线程数
    
    # 开始训练
    print("\n=== 训练配置信息 ===")
    print(f"使用设备: {'GPU' if args.device.startswith('cuda') or args.device.isdigit() and torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"批次大小: {args.batch_size}")
    print(f"工作进程数: {args.workers}")
    print(f"模型大小: YOLOv8{args.model_size}")
    print(f"图像缓存: {'启用' if args.cache else '禁用'}")
    print(f"半精度训练: {'启用' if args.half else '禁用'}")
    print("==================\n")
    
    results = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        cache=args.cache,
        workers=args.workers,
        model_size=args.model_size,
        half=args.half
    )
    print("训练完成！") 