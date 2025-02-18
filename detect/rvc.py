import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import queue
import time
import numpy as np
from PIL import ImageGrab
import cv2
import threading


class ScreenRecorder:
    def __init__(self, interval=0.5, correction=0.0):
        self.coordinates = None
        self.previous_frames = None  # 上一帧图片
        self.interval = interval  # 每隔多少秒截取一次屏幕
        self.is_paused = True  # Start in paused state
        self.thread = None
        self.change_queue = queue.Queue()
        self.correction = correction

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = torch.hub.load(r'C:\Users\86198\桌面\software-yolov5\yolov5', 'custom', path='best.pt', source='local',
        #                        force_reload=True).to(device)

        # Start the recording in a separate thread to handle the generator
        self.thread = threading.Thread(target=self.record)
        self.thread.daemon = True
        self.thread.start()

    def start(self, coordinates):
        self.coordinates = coordinates
        print(f"Starting recording with coordinates {coordinates}...")
        self.previous_frames = [None] * len(coordinates)

    # todo: 截取指定区域
    def capture_frame(self, coord):
        x, y, width, height = coord
        # bbox = (x, y + self.correction, x + width, y + height)
        bbox = (x, y, x + width, y + height)

        print(f"bbox: {bbox}")
        screen = ImageGrab.grab(bbox)
        return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)


    def is_different(self, frame1, frame2):
        return np.sum(cv2.absdiff(frame1, frame2)) > 0

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def record(self):
        while True:
            if self.is_paused:
                time.sleep(self.interval)
                continue

            time_record = time.time()

            for idx, coord in enumerate(self.coordinates):
                print(f"Recording area {coord}...")
                current_frame = self.capture_frame(coord)

                if self.previous_frames[idx] is None or self.is_different(current_frame, self.previous_frames[idx]):
                    # time_record = time.time()
                    # result = self.model(current_frame)
                    # print(result.xyxyn)
                    # print(time_record - time.time())
                    self.previous_frames[idx] = current_frame
                    # print(f"Area {idx + 1} has changed.")
                    self.change_queue.put(
                        (idx, current_frame, time_record))  # Yield immediately upon detecting changes

            time.sleep(self.interval)

    #
    #         # Capture the full screen
    #         screen_frame = self.capture_screen()
    #
    #         # Detect software windows using YOLO
    #         labels, boxes = self.detect_software_windows(screen_frame)
    #
    #         for label, box in zip(labels, boxes):
    #             x1, y1, x2, y2 = box
    #             current_frame = screen_frame[y1:y2, x1:x2]
    #
    #             # Retrieve previous frame for this label
    #             previous_frame = self.previous_frames.get(label)
    #
    #             if previous_frame is None or self.is_different(current_frame, previous_frame):
    #                 self.previous_frames[label] = current_frame
    #                 # Place the change in the queue with the label
    #                 self.change_queue.put((label, current_frame, time_record))
    #
    #         time.sleep(self.interval)
    def get_changes(self):
        while True:
            yield [self.change_queue.get()]
