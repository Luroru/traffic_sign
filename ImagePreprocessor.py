'''
create on 2025/5/28
@author: 李志涵lizhihan
@description: 
@version 1.0
'''
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self,
                 denoise_strength=0.5,
                 contrast_clip_limit=2.0,
                 contrast_tile_grid_size=(8, 8),
                 color_balance=True):
        """
        初始化预处理参数
        :param denoise_strength: 去噪强度（h值）
        :param contrast_clip_limit: 对比度限制（CLAHE参数）
        :param contrast_tile_grid_size: 网格大小（CLAHE参数）
        :param color_balance: 是否启用颜色校正
        """
        self.denoise_strength = denoise_strength
        self.contrast_clip_limit = contrast_clip_limit
        self.grid_size = contrast_tile_grid_size
        self.enable_color_balance = color_balance

    def get_gaussian_parameters(self,denoise_strength: float):
        """
        将 denoise_strength 映射为高斯滤波参数
        :param denoise_strength: 0.0 到 1.0 之间的浮点数
        :return: (ksize, sigmaX)，若强度为 0 返回 None
        """
        # 限制范围
        denoise_strength = max(0.0, min(1.0, denoise_strength))

        if denoise_strength == 0.0:
            return None  # 不滤波

        # 映射 sigmaX: 0.5 ~ 3.0
        sigmaX = 0.5 + denoise_strength * 2.5

        # 映射 ksize: 3 ~ 9（确保为奇数）
        size = int(3 + denoise_strength * 6)
        if size % 2 == 0:
            size += 1

        ksize = (size, size)
        return ksize, sigmaX

    def preprocess(self, frame):
        # 高斯滤波轻量去噪
        params = self.get_gaussian_parameters(self.denoise_strength)
        if params:
            frame = cv2.GaussianBlur(frame, *params)

        # CLAHE增强
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = l.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.contrast_clip_limit, tileGridSize=self.grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        frame = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # 白平衡（可选）
        if self.enable_color_balance:
            frame = self.simple_white_balance(frame)

        return frame

    def simple_white_balance(self, img):
        """
        简单灰世界假设的白平衡算法
        :param img: BGR 图像
        :return: 校正后的图像
        """
        result = img.copy()
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])

        avg_gray = (avg_b + avg_g + avg_r) / 3
        scale_b = avg_gray / avg_b
        scale_g = avg_gray / avg_g
        scale_r = avg_gray / avg_r

        result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)

        return result.astype(np.uint8)

    def update_parameters(self, **kwargs):
        """
        动态更新预处理参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
