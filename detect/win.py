import pathlib
import sys
import threading
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from ultralytics import YOLO

from proc import Process
from rvc import ScreenRecorder

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from PyQt5.QtCore import pyqtSignal,QRect, Qt
from PyQt5.QtGui import QImage,QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QFrame, \
    QScrollArea, QTabWidget, QSpacerItem, QSizePolicy, QTableWidget, \
    QAbstractItemView, QTableWidgetItem
class MainWindow(QWidget):
    update_box_signal = pyqtSignal(int, tuple, tuple)
    clear_box_signal = pyqtSignal(int)
    dis_com_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()

        self.model = None
        threading.Thread(target=self.load_yolo).start()

        # 设置主窗口
        self.setWindowFlags(Qt.WindowStaysOnBottomHint)
        self.setWindowTitle("股票延时对比软件")
        self.showMaximized()  # 打开时最大化

        # 主布局：水平布局，左4/5，右1/5
        main_layout = QHBoxLayout(self)

        # 左侧布局
        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.StyledPanel)
        main_layout.addWidget(self.left_frame, 4)

        # 右侧布局 - 包含选项卡和滚动区域
        self.right_tab_widget = QTabWidget()
        main_layout.addWidget(self.right_tab_widget, 1)

        # 添加对比和日志选项卡
        self.compare_tab = QWidget()
        self.log_tab = QWidget()

        self.right_tab_widget.addTab(self.compare_tab, "对比")
        self.right_tab_widget.addTab(self.log_tab, "日志")

        # 对比页面 - 使用滚动区域
        self.compare_scroll_area = QScrollArea(self.compare_tab)
        self.compare_scroll_area.setStyleSheet("QScrollArea { border: none; }")
        self.compare_scroll_area.setWidgetResizable(True)
        self.compare_widget = QWidget()
        self.compare_layout = QVBoxLayout(self.compare_widget)
        self.compare_scroll_area.setWidget(self.compare_widget)

        # 添加滚动区域到对比标签
        self.compare_tab_layout = QVBoxLayout(self.compare_tab)
        self.compare_tab_layout.addWidget(self.compare_scroll_area)

        # 在滚动区域底部留白
        self.compare_tab_layout.addSpacerItem(QSpacerItem(2, 60, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 日志页面内容 - 设置为不可编辑的 QLabel，并调整大小
        self.log_layout = QVBoxLayout(self.log_tab)

        # 使用 QScrollArea 包裹日志内容以实现滚动
        self.log_scroll_area = QScrollArea(self.log_tab)
        self.log_scroll_area.setWidgetResizable(True)
        self.log_scroll_area.setStyleSheet("QScrollArea { border: none; }")
        self.log_text = QLabel(self.log_tab)
        self.log_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_text.setAutoFillBackground(True)
        self.log_text.setWordWrap(True)

        # 将 QLabel 设置为 QScrollArea 的子组件
        self.log_scroll_area.setWidget(self.log_text)
        # 设置固定高度

        # 将日志滚动区域添加到日志布局，并在底部留白
        self.log_layout.addWidget(self.log_scroll_area)
        self.log_layout.addSpacerItem(QSpacerItem(2, 60, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 左侧内容布局
        self.left_layout = QVBoxLayout(self.left_frame)

        # 添加输入框和确定按钮
        self.input_label = QLabel("输入软件个数：", self)
        self.left_layout.addWidget(self.input_label, alignment=Qt.AlignCenter)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("请输入一个整数")
        self.input_field.returnPressed.connect(self.setup_areas)
        self.left_layout.addWidget(self.input_field, alignment=Qt.AlignCenter)

        self.confirm_button = QPushButton("确定", self)
        self.confirm_button.clicked.connect(self.setup_areas)
        self.left_layout.addWidget(self.confirm_button, alignment=Qt.AlignCenter)

        # 添加清空返回和开始按钮到右侧
        self.clear_button = QPushButton("清空", self)
        self.clear_button.clicked.connect(self.clear_log_content)

        self.return_button = QPushButton("返回", self)
        self.return_button.clicked.connect(self.clear_and_return)

        self.start_button = QPushButton("开始", self)
        self.start_button.clicked.connect(self.toggle_start_stop)

        # 添加按钮到右侧布局
        self.right_layout = QVBoxLayout(self.right_tab_widget)
        self.right_layout.addStretch()

        # 创建水平布局来放置“返回”和“开始”按钮，并对齐到右侧
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()  # 添加一个伸缩项将按钮推到右侧
        button_layout.addWidget(self.return_button)
        button_layout.addWidget(self.start_button)
        self.right_layout.addLayout(button_layout)

        # 初始隐藏按钮
        self.clear_button.hide()
        self.return_button.hide()
        self.start_button.hide()

        # 用于存储创建的框和坐标，以便后续删除和查看
        self.frames = []
        self.coordinates = []  # 用于存储每个框的四个角坐标
        self.labels_and_coords = []
        self.n = 0

        self.recorder = ScreenRecorder(correction=self.geometry().getRect()[1])  # 只开启了线程
        self.processer = Process(self.recorder, self.append_log, self.update_ranking)

        self.overlay_boxes = OverlayFrameManager(self)
        self.overlay_areas = OverlayFrameManager(self)

        # self.processer.update_box_signal.connect(self.update_box_ui)
        # self.processer.clear_box_signal.connect(self.clear_box_ui)
        # self.processer.dis_com_signal.connect(self.update_ranking)

        # 输入框获得焦点
        self.input_field.setFocus()

    def load_yolo(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = torch.hub.load(r'C:\Users\86198\桌面\software-yolov5\yolov5', 'custom', path='best.pt',
        #                        source='local', force_reload=True).to(device)
        # yolov8 from ultralytics
        self.model = YOLO("best.pt").to(device)

        self.append_log("模型加载完成")

    def setup_areas(self):
        # 获取输入的软件数量
        try:
            self.n = int(self.input_field.text())
            if self.n <= 0:
                raise ValueError
        except ValueError:
            self.input_label.setText("请输入有效的正整数！")
            return

        # 隐藏输入框和按钮
        self.input_label.hide()
        self.input_field.hide()
        self.confirm_button.hide()
        self.clear_button.show()  # 显示清空按钮
        self.return_button.show()  # 显示返回按钮
        self.start_button.show()  # 显示开始按钮

        # 清除之前的组件
        for i in reversed(range(self.compare_layout.count())):
            widget_to_remove = self.compare_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()

        # 创建表格组件
        self.table_widget = QTableWidget(self.n, 2, self.compare_widget)  # n 行，2 列
        self.table_widget.setHorizontalHeaderLabels(["名称", "延迟/s"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)  # 自动拉伸最后一列
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 禁止编辑

        # 设置表格样式
        self.table_widget.setShowGrid(False)  # 启用网格线
        self.table_widget.setStyleSheet(
            """
            QTableWidget {
                border: none;  /* 去掉整个表格外框 */
                background-color: transparent;  /* 使表格背景与应用背景一致 */
            }
            """
        )

        # 隐藏行号
        self.table_widget.verticalHeader().setVisible(False)

        # 添加初始数据到表格中
        for i in range(self.n):
            name_item = QTableWidgetItem(f"软件 {i + 1}")
            name_item.setTextAlignment(Qt.AlignCenter)
            delay_item = QTableWidgetItem("0")  # 初始延迟为 0
            delay_item.setTextAlignment(Qt.AlignCenter)

            self.table_widget.setItem(i, 0, name_item)
            self.table_widget.setItem(i, 1, delay_item)

        # 将表格添加到布局中
        self.compare_layout.addWidget(self.table_widget)

        # 在左侧布局管理器中设置区域框架和坐标
        self.frames = []
        self.coordinates = []

        self.setup_coordinate_frames(self.n)
        # self.recorder.start(self.coordinates)
        self.processer.start(self.n)


    def setup_coordinate_frames(self, n, spacing=10):
        # 获取左侧区域的宽度和高度
        left_width = self.left_frame.width()
        left_height = self.left_frame.height()

        # 获取左侧区域在屏幕中的绝对坐标
        left_frame_x = self.left_frame.x()
        left_frame_y = self.left_frame.y()

        best_width, best_height, max_area = 0, 0, 0
        for width in range(10, left_width // 2, 10):
            height = width * 2
            rows, cols = left_height // (height + spacing), left_width // (width + spacing)
            total_regions = rows * cols

            if total_regions >= n:
                area = (width + spacing) * (height + spacing)
                if area > max_area:
                    max_area, best_width, best_height = area, width, height

        # 考虑间距调整列数和行数
        cols = min(left_width // (best_width + spacing), n)
        rows = (n + cols - 1) // cols
        total_width = cols * best_width + (cols - 1) * spacing
        total_height = rows * best_height + (rows - 1) * spacing
        x_offset = (left_width - total_width) // 2 + left_frame_x
        y_offset = (left_height - total_height) // 2 + left_frame_y

        current_x, current_y = x_offset, y_offset
        for i in range(n):
            line_frame = QFrame(self)
            line_frame.setFrameShape(QFrame.Box)
            line_frame.setGeometry(current_x - left_frame_x, current_y - left_frame_y, best_width, best_height)
            line_frame.setParent(self.left_frame)
            line_frame.show()
            self.frames.append(line_frame)

            self.coordinates.append((current_x, current_y, best_width, best_height))
            # print('checking coordinates', self.coordinates[-1])
            # 获取left_frame的绝对坐标
            # print(self.left_frame.geometry().getRect().x(), self.left_frame.geometry().getRect().y())

            # 用OverlayFrameManager绘制coordinate框
            # print(left_frame_x, left_frame_y)
            # overlay_manager = OverlayFrameManager(self)
            # overlay_manager.draw_boxes(self.coordinates)

            # 更新 `current_x` 和 `current_y`，加上间距
            current_x += best_width + spacing
            if current_x + best_width > x_offset + total_width:
                current_x = x_offset
                current_y += best_height + spacing

    def clear_and_return(self):
        # 清除所有生成的框
        for frame in self.frames:
            frame.deleteLater()
        self.frames.clear()

        # 清空坐标列表
        self.coordinates.clear()
        self.recorder.pause()
        self.start_button.setText("开始")
        self.append_log("停止检测\n返回")

        # 清空表格内容
        self.table_widget.clearContents()
        self.overlay_boxes.clear_boxes()
        self.overlay_areas.clear_boxes()

        # 全选输入框并获得焦点
        self.input_field.selectAll()
        self.input_field.setFocus()

        # 恢复输入界面
        self.input_label.show()
        self.input_field.show()
        self.confirm_button.show()
        self.return_button.hide()
        self.start_button.hide()

    def toggle_start_stop(self):
        if self.start_button.text() == "开始":
            if not self.model:
                self.append_log("模型未加载完成，请稍后再试")
                return
            # 执行开始时的操作
            # print(self.left_frame.geometry().getRect())

            # overlay_manager = OverlayFrameManager(self)
            # overlay_manager.draw_boxes([self.left_frame.geometry().getRect()])

            self.overlay_areas.draw_boxes([self.left_frame.geometry().getRect()], True)

            self.append_log("开始检测")  # 这里可以替换为实1际操作
            self.start_button.setText("停止")

            self.capture_and_detect()

            # todo 应该传yolo识别后的坐标1/3进去
            detection_coordinates = [coords for _, coords in self.labels_and_coords]
            # Start the recorder with the detected coordinates
            print("yolo返回的坐标为：", detection_coordinates)
            self.recorder.start(detection_coordinates)
            self.recorder.resume()

            for i, (label, _) in enumerate(self.labels_and_coords):
                self.update_name(i, label)
            # self.processer.start(self.n)

        else:
            # 执行停止时的操作
            self.recorder.pause()
            self.append_log("停止检测")  # 这里可以替换为实际操作
            self.start_button.setText("开始")

            self.overlay_boxes.clear_boxes()
            self.overlay_areas.clear_boxes()

    def qimage_to_cv2(self, qimage):
        # 将QImage转换为numpy数组
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)

        return arr

    def capture_and_detect(self):
        # 获取左侧区域在屏幕中的绝对坐标
        left_frame_x = self.left_frame.x()
        left_frame_y = self.left_frame.y()

        # 截图并调用YOLO识别
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0)
        img = screenshot.toImage()

        # 将 QImage 转换为 OpenCV 格式
        arr = self.qimage_to_cv2(img)

        # 裁剪图片左边4/5宽度
        width = arr.shape[1]
        cropped_arr = arr[:, :int(width * 4 / 5), :]

        # 调用YOLO模型进行检测
        results = self.model(cropped_arr)

        # 解析检测结果
        results_list = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, classes):
                results_list.append([*box, conf, cls])

        results_list.sort(key=lambda x: x[4], reverse=True)

        # IoU计算函数
        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1_, y1_, x2_, y2_ = box2

            xi1 = max(x1, x1_)
            yi1 = max(y1, y1_)
            xi2 = min(x2, x2_)
            yi2 = min(y2, y2_)

            inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2_ - x1_) * (y2_ - y1_)

            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area

        # 提取检测结果并转换为相对于电脑屏幕的xywh格式
        self.labels_and_coords = []
        selected_boxes = []
        for *box, conf, cls in results_list:
            if len(self.labels_and_coords) >= self.n:
                break
            if not any(iou(box, selected_box) > 0.3 for selected_box in selected_boxes):
                x1, y1, x2, y2 = box
                label = self.model.names[int(cls)]
                x = int(x1)
                y = int(y1 - self.geometry().getRect()[1])
                w = int(x2 - x1)
                h = int(y2 - y1) // 3
                self.labels_and_coords.append((label, (x, y, w, h)))
                selected_boxes.append(box)

        # 根据xy从上到下，从左到右排序
        self.labels_and_coords.sort(key=lambda item: (item[1][0], item[1][1]))

        # 记录到日志
        for label, coordinate in self.labels_and_coords:
            self.append_log(f"{label}: {coordinate}")

        self.overlay_boxes.draw_boxes(coordinate for _, coordinate in self.labels_and_coords)

    def clear_log_content(self):
        self.log_text.clear()

    def append_log(self, text):
        # 添加日志内容
        self.log_text.setText(self.log_text.text() + text + '\n')

    # def update_box_ui(self, idx, bj_box, cle_box):  # 获取当前指定的 frame 坐标
    #     print('checking idx', idx)
    #     base_x, base_y, _, _ = self.coordinates[idx]  # 获取框架左上角的 (x, y) 坐标
    #
    #     # 转换 bj_box 和 cle_box 的坐标为绝对位置
    #     # print(bj_box, cle_box)
    #     if -1 not in bj_box:
    #         bj_absolute = (
    #             base_x + bj_box[0],  # x
    #             base_y + bj_box[1],  # y
    #             bj_box[2],  # width
    #             bj_box[3]  # height
    #         )
    #     else:
    #         bj_absolute = None
    #
    #     if -1 not in cle_box:
    #         cle_absolute = (
    #             base_x + cle_box[0],  # x
    #             base_y + cle_box[1],  # y
    #             cle_box[2],  # width
    #             cle_box[3]  # height
    #         )
    #     else:
    #         cle_absolute = None
    #
    #     # 如果指定的 frame 没有 overlay 管理器，则创建一个
    #     if idx not in self.overlay_managers:
    #         # 创建一个新的 OverlayFrameManager，并将其绑定到特定的 frame 区域
    #         overlay_manager = OverlayFrameManager(self)
    #         self.overlay_managers[idx] = overlay_manager
    #     else:
    #         overlay_manager = self.overlay_managers[idx]
    #     # print(bj_absolute, cle_absolute)
    #     # 设置 overlay 的框位置，并显示在屏幕上
    #     print('checking bj_absolute, cle_absolute', idx)
    #     overlay_manager.draw_frames([bj_absolute, cle_absolute])

    # def clear_box_ui(self, idx):
    #     # 如果指定的 frame 没有 overlay 管理器，则不执行任何操作
    #     if idx not in self.overlay_managers:
    #         return
    #
    #     # 清除指定 frame 的 overlay
    #     self.overlay_managers[idx].clear_frames()

    def update_ranking(self, new):
        if not new:
            return

        # 假设 new 是类似 [[index, delay], [index, delay], ...] 的形式
        for i, (idx, delay) in enumerate(new):
            index_item = QTableWidgetItem(f"{self.labels_and_coords[idx][0]}")
            index_item.setTextAlignment(Qt.AlignCenter)
            self.table_widget.setItem(i, 0, index_item)

            delay_item = QTableWidgetItem(str(delay))
            delay_item.setTextAlignment(Qt.AlignCenter)
            self.table_widget.setItem(i, 1, delay_item)

        # self.table_widget.sortItems(1, Qt.AscendingOrder)
        # 马上更新
        self.table_widget.viewport().update()

    def update_name(self, idx, name):
        # 创建新的名称项
        name_item = QTableWidgetItem(name)
        name_item.setTextAlignment(Qt.AlignCenter)

        # 更新表格中的名称
        self.table_widget.setItem(idx, 0, name_item)




class OverlayBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rectangles = []  # 用于存储所有矩形的坐标和填充状态

        # 设置无边框和总是位于最上层
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        # 设置窗口透明度
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 确保窗口接受鼠标事件
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        # 使窗口不会在任务栏中显示
        self.setWindowFlag(Qt.Tool)

    def set_rectangles(self, rectangles):
        """
        设置多个矩形的坐标和填充状态
        :param rectangles: [(x, y, width, height, fill), ...]
        """
        # 找出变化了的矩形框
        changed_rectangles = [rect for rect in rectangles if rect not in self.rectangles]
        if changed_rectangles:
            self.rectangles = rectangles
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0), 2)  # 设置画笔颜色和宽度
        painter.setPen(pen)

        for rect in self.rectangles:
            x, y, w, h, fill = rect
            if fill:
                painter.fillRect(QRect(x, y, w, h), QColor(0, 0, 0, 20))  # 半透明填充
            else:
                painter.drawRect(QRect(x, y, w, h))


class OverlayFrameManager:
    def __init__(self, parent=None):
        self.parent = parent
        self.overlay_box = OverlayBox(parent)
        self.overlay_box.hide()

    def draw_boxes(self, coordinates, fill=False):
        """
        绘制矩形框
        :param coordinates: [(x, y, width, height), ...]
        :param fill: 是否填充矩形框
        """
        self.overlay_box.set_rectangles([(x, y, w, h, fill) for x, y, w, h in coordinates])
        self.overlay_box.setGeometry(self.parent.geometry())
        self.overlay_box.show()

    def clear_boxes(self):
        """
        隐藏所有矩形框
        """
        self.overlay_box.hide()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
