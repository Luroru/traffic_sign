import torch
import cv2
import numpy as np

class Detector:
    def __init__(self, model_path='runs/train/tt100k/weights/best.pt', conf_thres=0.5, iou_thres=0.45):
        """
        初始化交通标志检测器
        Args:
            model_path (str): YOLOv5 模型路径
            conf_thres (float): 置信度阈值
            iou_thres (float): IOU 阈值
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = self._load_model()

    def _load_model(self):
        """
        加载 YOLOv5 模型
        Returns:
            model: 加载好的模型
        """
        print("加载模型中...")
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=self.model_path, trust_repo=True)
        model.conf = self.conf_thres
        model.iou = self.iou_thres
        print("模型加载成功！")
        return model

    def detect(self, frame):
        """
        对图像帧进行目标检测
        Args:
            frame (np.ndarray): 图像帧（BGR格式）
        Returns:
            processed_frame (np.ndarray): 标注后的图像帧
            labels_str (str): 检测结果字符串（每行格式：<class> <x1> <y1> <x2> <y2> <confidence>）
        """
        try:
            # YOLOv5 接受 RGB 格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame)

            # 绘制检测结果
            rendered = results.render()
            processed_frame = cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR)

            # 生成检测信息字符串
            labels_str = ""
            for det in results.pred[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                cls_name = self.model.names[int(cls)]
                labels_str += f"{cls_name} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.2f}\n"

            return processed_frame, labels_str

        except Exception as e:
            print(f"[检测错误] {e}")
            return frame, ""

