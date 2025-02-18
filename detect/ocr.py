import logging
import re
import jieba
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

logging.disable(logging.DEBUG)


class ProcessImage:
    def __init__(self):
        self.custom_words = ['量比', '亿', '万', '元', '换', '额', '成交额', '额', '金额', '总额']
        self.bj_words = ['价格', '固定价格']
        self.model = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)

        # 添加自定义词汇到 jieba
        if self.custom_words:
            for word in self.custom_words:
                jieba.add_word(word)

    # def get_imgs(self,):
    #     image_files = [f for f in os.listdir(self.path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    #     image_files = [os.path.join(self.path, f) for f in image_files]
    #     return image_files

    def info_of(self, img):
        phrases = self.get_phrase(img)
        # print(phrases)
        if not phrases:
            return [-1, -1], [-1, -1], [-1, -1], [-1, -1]

        bj = self.get_bj(phrases)
        cle = self.get_cje(phrases)
        # print(bj, cle)

        if not bj and not cle:
            return [-1, -1], [-1, -1], [-1, -1], [-1, -1]
        # if (not bj and cle) or (bj and not cle):
        #     phrases = self.get_phrase(img)
        #     bj = self.get_bj(phrases)
        #     cle = self.get_cje(phrases)
        if not bj:
            return [-1, -1], [-1, -1], self.convert_to_xywh(cle[0]), cle[1]
        if not cle:
            return self.convert_to_xywh(bj[0]), bj[1], [-1, -1], [-1, -1]

        return self.convert_to_xywh(bj[0]), bj[1], self.convert_to_xywh(cle[0]), cle[1]
        # boxes = [bj[0], cle[0]]
        # img = draw_ocr(img, boxes)
        # img = Image.fromarray(np.uint8(img))
        # img.show()

    def convert_to_xywh(self, corner):
        if -1 in corner:
            return None
        # corners is expected to be [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        min_x = int(corner[0][0])
        min_y = int(corner[0][1])
        max_x = int(corner[1][0])
        max_y = int(corner[2][1])

        # Calculate width and height
        width = max_x - min_x
        height = max_y - min_y

        return [min_x, min_y, width, height]

    def is_num(self, txt):
        try:
            float(txt)
            return True
        except ValueError:
            return False

    def get_phrase(self, img):
        result = self.model.ocr(img, cls=True)[0]
        if not result:
            return None
        boxes = [line[0] for line in result if line[1][1] > 0.95]
        txts = [line[1][0] for line in result if line[1][1] > 0.95]
        # 用图像显示识别结果
        # img = Image.open(img)
        # img = draw_ocr(img, boxes)
        # img = Image.fromarray(np.uint8(img))
        # img.show()
        return self.combine_words(self.pos_encode(boxes, txts), img)

    def pos_encode(self, boxes, txts, img=None):
        char_boxes = []
        for txt, box in zip(txts, boxes):
            # 计算每个字符的宽度（假设字符均匀分布）
            line_width = abs(box[1][0] - box[0][0])
            line_height = abs(box[2][1] - box[0][1])
            char_width = min((line_width / len(txt)), line_height)

            # 遍历干净的文本中的每个字符
            for i, char in enumerate(txt):
                # 计算每个字符的框（box）
                char_box = [
                    [box[0][0] + i * char_width, box[0][1]],  # 左上角
                    [box[0][0] + (i + 1) * char_width, box[0][1]],  # 右上角
                    [box[0][0] + (i + 1) * char_width, box[2][1]],  # 右下角
                    [box[0][0] + i * char_width, box[2][1]]  # 左下角
                ]
                char_boxes.append((char, char_box))
        # img = Image.open(img)
        # img = draw_ocr(img, [box for _, box in char_boxes])
        # img = Image.fromarray(np.uint8(img))
        # img.show()
        return char_boxes, txts

    def combine_words(self, info, img=None):
        char_boxes, txts = info
        # 一个个处理txts，用jieba
        words = []
        for txt in txts:
            words += jieba.cut(txt)

        # 用于保存结果的列表
        result = []
        char_index = 0  # 指向 char_boxes 的当前字符位置

        for word in words:
            word_length = len(word)
            word_boxes = [char_boxes[char_index + i][1] for i in range(word_length)]

            # 计算词的整体 box 坐标（覆盖该词的最小矩形框）
            min_x = min(box[0][0] for box in word_boxes)
            max_x = max(box[1][0] for box in word_boxes)
            min_y = min(box[0][1] for box in word_boxes)
            max_y = max(box[2][1] for box in word_boxes)
            word_box = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

            # 计算合并后的高度和第一个字的高度
            combined_height = max_y - min_y
            first_char_height = abs(word_boxes[0][2][1] - word_boxes[0][0][1])

            # 如果合并后的高度大于第一个字高度的 1.5 倍，则只框选第一个字
            if combined_height > 1.5 * first_char_height:
                word_box = word_boxes[0]  # 将词组框设为第一个字的框

            # 如果不是标点，则将词与其 box 添加到结果
            if re.search(r'[\w\u4e00-\u9fff]', word) is not None:
                result.append((word_box, word))

            # 更新 char_index
            char_index += word_length

        # img = draw_ocr(img, [box for box, _ in result])
        # img = Image.fromarray(np.uint8(img))
        # img.show()
        return result

    def get_bj(self, phrases):
        max_size = 0
        max_txt = ''
        max_box = []
        # max_only = True
        for box, txt in phrases:
            size = abs(box[2][1] - box[0][1])
            # print(txt, size)
            if size > max_size and '.' in txt:
                max_size = size
                max_txt = txt
                max_box = box
            #     max_only = True
            # elif size == max_size:
            #     max_only = False

        # return (max_box, max_txt) if max_only and self.is_num(max_txt) else None
        return (max_box, max_txt) if self.is_num(max_txt) else None

    def get_cje(self, phrases):
        pure_words = [txt for box, txt in phrases]
        cje_words = ['成交额', '金额', '总额', '额', '总值', '交易金额', 'Turnover', 'Volume', 'Trade Value', 'value']

        for i in range(len(pure_words) - 1):
            if pure_words[i] in cje_words:
                next_word = pure_words[i + 1]
                if self.is_num(next_word):
                    cje_box = phrases[i][0]  # 当前金额词的 box
                    num_box = phrases[i + 1][0]  # 下一个数字的 box

                    # 合并 box，取最小和最大坐标形成新的框
                    min_x = min(cje_box[0][0], num_box[0][0])
                    max_x = max(cje_box[1][0], num_box[1][0])
                    min_y = min(cje_box[0][1], num_box[0][1])
                    max_y = max(cje_box[2][1], num_box[2][1])
                    merged_box = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

                    return merged_box, next_word

        return None


if __name__ == '__main__':
    processer = ProcessImage()
    print(processer.info_of('jj/NiuGuWang_self_down_9.jpg'))
