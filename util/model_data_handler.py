'''
获取模型数据所需
'''
import numpy as np
import os


def get_classes(classes_path):
    '''
    获取标签
    :param classes_path:
    :return:装有标签名的 list
    '''
    # classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''
    获取anchor
    :param anchors_path:
    :return: 装有anchor的numpy数组
    '''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def compute_iou(rec1, rec2):
    '''
    计算交并比 IOU
    :param rec1:  (top, left, bottom, right) 元组
    :param rec2:  (top, left, bottom, right) 元组
    :return: IOU的值
    '''
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)
