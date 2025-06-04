import torch
import sys
import traceback
import os
import argparse

def detect(image_path, conf_thres=0.25, iou_thres=0.45):
    try:
        print(f"开始加载模型...")
        # 加载模型
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/tt100k/weights/best.pt', trust_repo=True)
        print("模型加载成功！")

        # 设置参数
        model.conf = conf_thres  # 置信度阈值
        model.iou = iou_thres    # NMS IOU 阈值

        print(f"开始检测图片: {image_path}")
        # 进行检测
        results = model(image_path)
        
        # 保存结果
        print("保存检测结果...")
        save_dir = 'runs/detect/results'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存检测结果图片
        for i, result in enumerate(results.ims):
            results.save(save_dir=save_dir)  # 保存检测结果图片
            
            # 手动保存检测结果为txt格式
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            txt_path = os.path.join(save_dir, f'{base_name}_{i}.txt')
            with open(txt_path, 'w') as f:
                for det in results.pred[i]:
                    # 获取边界框坐标和置信度
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    # 写入格式：<class> <x1> <y1> <x2> <y2> <confidence>
                    f.write(f'{int(cls)} {x1} {y1} {x2} {y2} {conf}\n')

        print(f"检测完成！结果保存在 {save_dir} 目录下")
        print(f"检测到的目标：")
        for i, pred in enumerate(results.pred):
            print(f"\n图片 {i+1} 中检测到 {len(pred)} 个目标:")
            for det in pred:
                cls = int(det[5])
                conf = float(det[4])
                # 获取类别英文名称
                cls_name = model.names[cls]
                print(f"- 类别 {cls_name}, 置信度: {conf:.2f}")

    except Exception as e:
        print("\n发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用 YOLOv5 检测图片中的交通标志')
    parser.add_argument('--image', type=str, required=True, help='要检测的图片路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU 阈值 (默认: 0.45)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行检测
    detect(args.image, args.conf, args.iou) 