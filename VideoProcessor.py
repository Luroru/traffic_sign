'''
create on 2025/5/27
@author: 李志涵lizhihan
@description: 
@version 1.0
'''
import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise IOError(f"❌ 无法打开视频文件：{video_path}")
        print(f"✅ 成功打开视频文件：{video_path}")

    def is_opened(self):
        return self.capture.isOpened()

    def read_next_frame(self):
        if not self.capture:
            raise ValueError("⚠️ 视频未初始化")
        success, frame = self.capture.read()
        if not success:
            return None
        return frame

    def release(self):
        if self.capture:
            self.capture.release()
