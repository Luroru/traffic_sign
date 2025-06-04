import torch
import cv2
import numpy as np
import os

class Detector:
    def __init__(self, model_path='runs/train/tt100k/weights/best.pt', conf_thres=0.4, iou_thres=0.45):
        """
        初始化交通标志检测器
        Args:
            model_path: YOLOv5模型路径
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = self._load_model()

    def _load_model(self):
        """
        加载YOLOv5模型
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=self.model_path,
                               trust_repo=True)
        model.conf = self.conf_thres
        model.iou = self.iou_thres
        return model

    def detect(self, frame):
        """
        对输入图像帧进行交通标志检测
        Args:
            frame: 输入的图像帧（numpy数组）
        Returns:
            processed_frame: 带有检测标注的图像帧
            labels_str: 检测结果字符串，格式：<class> <x1> <y1> <x2> <y2> <confidence>
        """
        results = self.model(frame)
        processed_frame = results.render()[0]

        labels_str = ""
        if len(results.pred) > 0:
            for det in results.pred[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                labels_str += f"{int(cls)} {x1} {y1} {x2} {y2} {conf}\n"

        return processed_frame, labels_str
