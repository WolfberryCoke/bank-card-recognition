'''
处理图片所需
'''
from PIL import ImageDraw, Image
import numpy as np

def get_data(annotation_line, input_shape, max_boxes=20, proc_img=True):
    '''
    图像处理，宽高归一化，生成新的box的坐标

    :param annotation_line: *_label中的列表
    :param input_shape: 输入图片大小
    :param max_boxes: 可以传入最大box的数量
    :param proc_img: 是否添加背景并且归一化
    :return: 图像的大小和box位置信息
    '''

    line = annotation_line.split()
    image = Image.open(line[0])
    init_width, init_height = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    # 用以缩放大小
    scale = min(w / init_width, h / init_height)
    new_weight = int(init_width * scale)
    new_height = int(init_height * scale)
    # 求出图像的中心坐标
    x_center = (w - new_weight) // 2
    y_center = (h - new_height) // 2
    image_data = 0
    if proc_img:
        image = image.resize((new_weight, new_height), Image.BICUBIC)
        # 添加背景
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # 黏贴图
        new_image.paste(image, (x_center, y_center))
        # 数据归一化，使得RGB值在[0,1]之间
        image_data = np.array(new_image) / 255.

    # 定义新的BOX位置
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        # 最大 20 个 BOX
        if len(box) > max_boxes:
            box = box[:max_boxes]
        # 根据缩放大小，生成新图中的 BOX 位置
        box[:, [0, 2]] = box[:, [0, 2]] * scale + x_center
        box[:, [1, 3]] = box[:, [1, 3]] * scale + y_center
        box_data[:len(box)] = box

    return image_data, box_data

def letterbox_image(image, size):
    '''
    调整image的图片大小为size
    :param image: 待调整的图片
    :param size:(w,h) 元组
    :return:
    '''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    padding = ((w - nw) // 2, (h - nh) // 2)
    return new_image, scale, padding


def draw_box(image, box, result_save_path, scale, padding):
    '''
    在测试图片上标出数字串区域
    :param image: 图片
    :param box:   (top, left, bottom, right) 元组
    :param result_save_path: 保存路径
    :return: 无
    '''
    top, left, bottom, right = box
    draw = ImageDraw.Draw(image)
    line = 7
    width = (right - left) / scale
    height = (bottom - top) / scale
    left = (left - padding[0]) / scale
    top = (top - padding[1]) / scale
    for i in range(1, line + 1):
        draw.rectangle((left + (line - i), top + (line - i), left + width + i, top + height + i), outline='red')

    image = image.convert("RGB")
    image.save(result_save_path)

