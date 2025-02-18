import sys

import torch
# import torch
# import cv2
# from matplotlib import pyplot as plt
# from pathlib import Path
#
# # 添加YOLOv5路径
# import sys
# sys.path.append('../yolov5')
#
# # 导入YOLOv5的detect模块
# # from models.common import DetectMultiBackend
# # from utils.augmentations import letterbox
# # from utils.general import (non_max_suppression, scale_coords)
# # from utils.torch_utils import select_device
#
# # 定义推理函数
# # def run_inference(model, img_path, device):
# #     # 加载图像
# #     img0 = cv2.imread(img_path)
# #     assert img0 is not None, 'Image Not Found ' + img_path
# #
# #     # Padded resize
# #     img = letterbox(img0, new_shape=640)[0]
# #
# #     # Convert
# #     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
# #     img = np.ascontiguousarray(img)
# #
# #     # 推理
# #     img = torch.from_numpy(img).to(device)
# #     img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
# #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
# #     if img.ndimension() == 3:
# #         img = img.unsqueeze(0)
# #
# #     # Inference
# #     pred = model(img, augment=False, visualize=False)
# #
# #     # Apply NMS
# #     pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
# #
# #     # Process detections
# #     for i, det in enumerate(pred):  # per image
# #         if len(det):
# #             # Rescale boxes from img_size to img0 size
# #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
# #
# #     return img0, det
#
# # 主函数
# if __name__ == '__main__':
#     # 加载模型
#     device = select_device('')
#     model = DetectMultiBackend('best.pt', device=device, dnn=False, data=None, fp16=False)
#
#     # 进行推理
#     img_path = '../yolov5/data/mydata/all_images/同花顺/image_1.jpg'
#     img0, det = run_inference(model, img_path, device)
#
#     # 可视化结果
#     for *xyxy, conf, cls in det:
#         label = f'{model.names[int(cls)]} {conf:.2f}'
#         c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
#         cv2.rectangle(img0, c1, c2, (0, 255, 0), 2)
#         cv2.putText(img0, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # 使用Matplotlib显示图片
#     img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
#     plt.imshow(img0)
#     plt.axis('off')
#     plt.show()

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
# from pathlib import Path

# 加载模型
model = torch.hub.load(r'C:\Users\86198\桌面\software-yolov5\yolov5', 'custom', path='best.pt', source='local', force_reload=True)

# 加载图像
img = r'C:\Users\86198\桌面\software-yolov5\yolov5\data\img_1.png'  # 替换为你的图像路径

# 推理
results = model(img)

# 显示结果
results.show()  # 在窗口中显示结果
# 或者保存结果
# results.save(save_dir='path/to/save/results')  # 替换为你想保存结果的路径

# 获取检测结果
print(results.pandas().xyxy[0])  # 打印检测结果

