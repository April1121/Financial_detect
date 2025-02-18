import threading
from PyQt5.QtCore import pyqtSignal, QObject

from ocr import ProcessImage


class Process(QObject):
    update_box_signal = pyqtSignal(int, list, list)
    clear_box_signal = pyqtSignal(int)
    dis_com_signal = pyqtSignal(list)

    def __init__(self, recorder, show_log=None, show_com=None):
        super().__init__()

        self.processer = ProcessImage()  # ocr.ProcessImage()
        self.record = recorder.get_changes()  # 获取queue中的数据：上一帧图片和时间
        self.show_log = show_log
        self.show_com = show_com
        self.change_list = None

        # if dis_box:
        #     self.update_box_signal.connect(dis_box)

        # Start the process_changes function in a new thread
        self.recorder_thread = threading.Thread(target=self.process_changes)
        self.recorder_thread.daemon = True
        self.recorder_thread.start()

    def start(self, n):
        self.change_list = [None] * n
        # [n][2]
        # self.ranking = [[0, 0] for _ in range(n)]  # idx, delay

    def process_changes(self):
        # This function runs in a separate thread
        for changed_areas in self.record:
            if not changed_areas:
                print("No changes detected.")
                continue
            for idx, frame, time in changed_areas:
                # print(f"Area {idx + 1} has changed.")
                # self.clear_box_signal.emit(idx)
                # 打印图片的坐标
                print("ocr检测的区域为:", frame.shape)

                info = list(self.processer.info_of(frame))  # ocr识别图片 todo
                if info[1] == [-1, -1]:
                    info[1] = None
                if info[3] == [-1, -1]:
                    info[3] = None

                self.show_log(f"Area {idx + 1} — 报价：{info[1]} 成交额：{info[3]}")
                # self.update_box_signal.emit(idx, info[0], info[2])

                if None in info:
                    continue
                self.show_com(self.update_ranking(idx, info[1], info[3], time))
                # self.dis_com_signal.emit(self.update_ranking(idx, info[1], info[3], time))

    def update_ranking(self, idx, bj, cje, t):
        if not self.change_list[idx]:
            self.change_list[idx] = [(bj, cje, t)]
        elif self.change_list[idx][-1][0] == bj and self.change_list[idx][-1][1] == cje:
            return []
        else:
            self.change_list[idx] = self.change_list[idx][-4:] + [(bj, cje, t)]

        if None in self.change_list:
            return []

        # print('change list', self.change_list)
        # Find the first (bj, cje) tuple from the end that exists in all idxs
        common_tuple = None
        for i in range(len(self.change_list[0]) - 1, -1, -1):
            candidate = self.change_list[0][i][:2]  # Take (bj, cje) from the first list
            if all(candidate in [(x[0], x[1]) for x in cl] for cl in self.change_list if cl):
                # Found a common (bj, cje) tuple, keep checking to find the last one
                common_tuple = candidate
                break

        if common_tuple is None:
            return []  # No common (bj, cje) found to compare

        # Get the time of the last occurrence of the common_tuple in each idx
        latest_times = []
        for i, changes in enumerate(self.change_list):
            for change in reversed(changes):
                if (change[0], change[1]) == common_tuple:
                    latest_times.append((i, change[2]))
                    break

        # Find the minimum time from latest_times to use as baseline
        min_time = min(latest_times, key=lambda x: x[1])[1]

        # Calculate delays relative to the minimum time and update ranking
        delays = [(i, round(max(0, t - min_time), 2)) for i, t in latest_times]

        # Sort delays by time (smallest delay first means fastest update)
        return sorted(delays, key=lambda x: x[1])
