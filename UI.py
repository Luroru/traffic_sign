import os
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QFileDialog, QGroupBox, QGridLayout, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from datetime import datetime
from ImagePreprocessor import ImagePreprocessor
from VideoProcessor import VideoProcessor
from detect_yolov5_copy import Detector

class UI(QWidget):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("实时交通标志检测系统")
        self.setGeometry(100, 100, 1200, 800)
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.initUI()

    def initUI(self):
        # 视频显示区
        self.video_label = QLabel("视频显示区")
        self.video_label.setStyleSheet("background-color: #000; color: white;")
        self.video_label.setFixedSize(800, 450)
        self.video_label.setAlignment(Qt.AlignCenter)

        # 控制按钮
        self.btn_open = QPushButton("打开视频")
        self.btn_start = QPushButton("开始检测")
        self.btn_pause = QPushButton("暂停检测")
        self.btn_save = QPushButton("保存结果")

        self.btn_open.clicked.connect(self.open_video)
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_pause.clicked.connect(self.pause_detection)
        self.btn_save.clicked.connect(self.save_results)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_save)
        control_group = QGroupBox("控制面板")
        control_group.setLayout(control_layout)


        self.slider_filter = QSlider(Qt.Horizontal)
        self.slider_filter.setRange(0, 10)
        self.slider_filter.setValue(3)  # 默认值可以设定
        self.slider_filter.valueChanged.connect(self.update_filter)

        self.spin_clip_limit = QDoubleSpinBox()
        self.spin_clip_limit.setRange(1.0, 10.0)
        self.spin_clip_limit.setSingleStep(0.1)
        self.spin_clip_limit.setValue(2.5)
        self.spin_clip_limit.valueChanged.connect(self.update_contrast)



        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("滤波强度"), 0, 0)
        param_layout.addWidget(self.slider_filter, 0, 1)
        param_layout.addWidget(QLabel("对比度裁剪限制"), 2, 0)
        param_layout.addWidget(self.spin_clip_limit, 2, 1)


        param_group = QGroupBox("参数调整")
        param_group.setLayout(param_layout)

        # 另外，定义成员变量保存当前参数，方便调用图像处理函数时读取
        self.current_filter = self.slider_filter.value()/10.0
        self.current_contrast = self.spin_clip_limit.value()


        self.preprocessor = ImagePreprocessor(self.current_filter,self.current_contrast)
        self.detector = Detector()

        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("日志输出区域")
        log_group = QGroupBox("识别日志")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        # 左侧控制区
        left_layout = QVBoxLayout()
        left_layout.addWidget(control_group)
        left_layout.addWidget(param_group)

        # 右侧展示区
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_label)
        right_layout.addWidget(log_group)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")

        if not file_name:
            return  # 用户取消选择，直接返回

        # 判断是否为视频文件
        video_ext = ['.mp4', '.avi', '.mov', '.mkv']
        _, ext = os.path.splitext(file_name)
        if ext.lower() not in video_ext:
            self.log_result(f"❌ 不支持的文件类型：{ext}")
            return

        abs_path = os.path.abspath(file_name)
        self.video_path = abs_path  # ✅ 保存绝对路径到 self.video_path（供其他模块使用）
        self.capture = cv2.VideoCapture(abs_path)

        if not self.capture.isOpened():
            self.log_result(f"❌ 无法打开视频文件：{abs_path}")
            return

        self.log_result(f"✅ 成功打开视频文件：{abs_path}")

        # 读取第一帧展示
        success, frame = self.capture.read()
        if success:
            self.show_frame_on_label(frame)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 👈 关键：重设回第一帧
        else:
            self.log_result("⚠️ 无法读取第一帧图像")

    def start_detection(self):
        try:
            self.cnt = 0
            self.processor = VideoProcessor(self.video_path)
            self.timer.start(30)  # ≈ 30fps
            self.btn_pause.setText("⏸️ 暂停检测")  # 按钮显示“暂停”
            self.is_paused = False
        except Exception as e:
            print(f"❌ 错误：{str(e)}")


    def update_frame(self):
        frame = self.processor.read_next_frame()
        if frame is None:
            #self.timer.stop()
            #self.processor.release()
            print("✅ 视频处理完毕")
            self.processor.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 👈 关键：重设回第一帧
            frame = self.processor.read_next_frame()
            if frame is None:
                print("❌ 无法重新读取视频第一帧")
                return
            #return

        # 图像预处理
        frame = self.preprocessor.preprocess(frame)
        # 可在此处调用 YOLO 等检测逻辑（frame 输入）
        frame,text = self.detector.detect(frame)
        # 获取当前时间并格式化
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 拼接时间和识别结果
        text_with_time = f"[{current_time}] {text}"

        # 显示
        self.log_result(text_with_time)
        self.show_frame_on_label(frame)


    def show_frame_on_label(self, frame):
        # 如果需要灰度图：frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 将 BGR 转为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        # 转换为 QImage
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放图像以适配标签尺寸（可选）
        scaled_image = image.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)

        # 设置到 QLabel
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def pause_detection(self):
        # 如果没有暂停过，就暂停
        if not hasattr(self, 'is_paused'):
            self.is_paused = False  # 初始化

        if not self.is_paused:
            self.timer.stop()
            self.log_result("暂停检测")
            self.btn_pause.setText("▶️ 继续检测")  # 按钮显示“继续”
            self.is_paused = True
        else:
            self.timer.start(30)
            self.log_result("继续检测")
            self.btn_pause.setText("⏸️ 暂停检测")  # 按钮显示“暂停”
            self.is_paused = False

    def save_results(self):
        # TODO: 实现结果保存逻辑
        self.log_result("保存识别结果（待实现）")

    def show_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_filter(self, val):
        self.current_filter = val/10.0
        self.preprocessor.update_parameters(denoise_strength=self.current_filter)
        self.log_result(f"滤波强度设置为: {val}")

    def update_contrast(self, val):
        self.current_contrast = val
        self.preprocessor.update_parameters(contrast_clip_limit=self.current_contrast)
        self.log_result(f"对比度限制设置为: {val}")


    def log_result(self, message):
        self.log_text.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())
