'''
预测部分
'''
from PIL import Image
from keras import backend as K, Input
import os
from util.model_data_handler import get_anchors, get_classes
from timeit import default_timer as timer
import numpy as np
from card_recognize.yolo3.model import yolo_eval, yolo_body
from util.xml_handler import read_one_xml
from util.image_handler import letterbox_image
from util.model_data_handler import compute_iou
from util.image_handler import draw_box

# 参数
score_threshold = 0.6
iou = 0.45
image_size = (416, 416)
sess = K.get_session()
anchors_path = '../card_recognize/model_data/yolo_anchors.txt'
classes_path = '../card_recognize/model_data/classes.txt'
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
num_classes = len(class_names)


def get_predict_result(image_set_path, result_set_path, model_path):
    '''
    对于给定的图片集，给出预测的数字串集合
    :param image_set_path:
    :param result_set_path:
    :param model_path:
    :return: 符合格式的数字串
    '''
    all_result_list = []
    # 定义网络
    num_anchors = len(anchors)
    model_path = os.path.expanduser(model_path)

    yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
    yolo_model.load_weights(model_path)
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, num_classes, image_size, score_threshold, iou)
    # 遍历文件夹下的每张图片
    for file in os.listdir(image_set_path):
        file_path = os.path.join(image_set_path, file)
        image = Image.open(file_path)
        boxed_image, scale, padding = letterbox_image(image, tuple(reversed(image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        # 送进网络做预测
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes], feed_dict={
                yolo_model.input: image_data,
                K.learning_phase(): 0
            })
        # 取出预测结果
        pre_rec = []
        pre_trust = []
        pre_class = []
        for i, _ in sorted(list(enumerate(out_boxes[:, 1])), key=lambda e: e[1]):
            c = out_classes[i]
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            pre_rec.append((top, left, bottom, right))
            pre_trust.append(score)
            pre_class.append(predicted_class)
        # 做类间极大值抑制
        one_result_list, one_result_rec, one_result_trust = nms_inter_class(pre_rec, pre_trust, pre_class)
        result_save_path = os.path.join(result_set_path, file)

        one_result_str = file_path.split('\\')[-1][0:-4] + ':'
        for i in range(len(one_result_list)):
            if one_result_list[i] == 'number':
                draw_box(image, one_result_rec[i], result_save_path, scale, padding)
                continue
            one_result_str += one_result_list[i]
        all_result_list.append(one_result_str)

    return all_result_list


def nms_inter_class(pre_rec, pre_trust, pre_class):
    '''
    NMS只能去掉类内多预测框，本函数处理类间多预测框的问题
    :param pre_rec:
    :param pre_trust:
    :param pre_class:
    :return: ['1','2','3']的预测数字串以及每个预测数字对应的区域、可信度
    '''
    result_rec = []
    result_trust = []
    result_class = []
    for i in range(len(pre_class)):
        if len(result_rec) == 0:
            result_rec.append(pre_rec[i])
            result_trust.append(pre_trust[i])
            result_class.append(pre_class[i])
        else:
            flag = True
            for j in range(len(result_rec)):
                # 比较iou  若超过阈值，则比较置信度
                if compute_iou(result_rec[j], pre_rec[i]) > iou:
                    flag = False
                    # 将置信度高的替换掉result
                    if pre_trust[i] > result_trust[j]:
                        result_rec[j] = pre_rec[i]
                        result_trust[j] = pre_trust[i]
                        result_class[j] = pre_class[i]
                    break
            # 若无重复  则添加到result
            if flag == True:
                result_rec.append(pre_rec[i])
                result_trust.append(pre_trust[i])
                result_class.append(pre_class[i])
    return result_class, result_rec, result_trust


def get_accuracy(result_path, xml_set_path):
    '''
    计算准确率
    :param result_path: 预测结果路径
    :param xml_set_path: 标签文件路径
    :return: 无
    '''
    # 非数字标签
    non_digital_label = ['number', 'up', 'left', 'right', 'up', 'down', 'bottom']
    pre_digital_num = 0
    real_digital_num = 0
    pre_whole_num = 0
    real_whole_num = 1
    with open(result_path, 'r') as f:
        line = f.readline()
        while line:
            name = line.split(':')[0]
            pre_digital_str = line.split(':')[1]
            # 读取标记数据
            xml_path = xml_set_path + name + '.xml'
            real_digital_str = []
            node_list = read_one_xml(xml_path)
            for node in node_list:
                if node.name not in non_digital_label:
                    real_digital_str.append(node.name)
            # 取两份文件里面的最小长度
            length = min(len(pre_digital_str), len(real_digital_str))
            for i in range(length):
                if real_digital_str[i] == pre_digital_str[i]:
                    pre_digital_num += 1

            # 说明整个图片上的数字都预测正确
            if real_digital_str == pre_digital_str:
                pre_whole_num += 1

            line = f.readline()
            real_whole_num += 1
            real_digital_num += len(real_digital_str)

    print('按数字算准确率为：{}'.format(float(pre_digital_num) / float(real_digital_num)))
    print('按整张图片算准确率为：{}'.format(float(pre_whole_num) / float(real_whole_num)))


def get_test_label_accuracy(test_label_path, model_path):
    non_digital_label = ['number', 'up', 'left', 'right', 'up', 'down', 'bottom']
    pre_digital_num = 0
    real_digital_num = 0
    pre_whole_num = 0
    real_whole_num = 0
    # 定义网络
    num_anchors = len(anchors)
    model_path = os.path.expanduser(model_path)

    yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
    yolo_model.load_weights(model_path)
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, num_classes, image_size, score_threshold, iou)

    with open(test_label_path, 'r') as f:
        line = f.readline()
        while line:

            image_path = line.split(' ')[0]
            print(image_path)
            image = Image.open(image_path)
            boxed_image, scale, padding = letterbox_image(image, tuple(reversed(image_size)))
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)
            # 送进网络做预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes], feed_dict={
                    yolo_model.input: image_data,
                    K.learning_phase(): 0
                })
            # 取出预测结果
            pre_rec = []
            pre_trust = []
            pre_class = []
            for i, _ in sorted(list(enumerate(out_boxes[:, 1])), key=lambda e: e[1]):
                c = out_classes[i]
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                pre_rec.append((top, left, bottom, right))
                pre_trust.append(score)
                pre_class.append(predicted_class)
            # 做类间极大值抑制
            one_result_list, one_result_rec, one_result_trust = nms_inter_class(pre_rec, pre_trust, pre_class)
            # ['1','2','3']
            name = image_path.split('/')[-1][:-4]
            pre_digital_str = ''
            for one in one_result_list:
                if one not in non_digital_label:
                    pre_digital_str += one
            # 读取标记数据
            xml_path = '../dataset/annotation/' + name + '.xml'
            real_digital_str = []
            node_list = read_one_xml(xml_path)
            for node in node_list:
                if node.name not in non_digital_label:
                    real_digital_str.append(node.name)
            flag = True
            # 取两份文件里面的最小长度
            length = min(len(pre_digital_str), len(real_digital_str))
            for i in range(length):
                if real_digital_str[i] == pre_digital_str[i]:
                    pre_digital_num += 1
                else:
                    flag = False
            # 说明整个图片上的数字都预测正确
            if real_digital_str == pre_digital_str:
                pre_whole_num += 1

            line = f.readline()
            real_whole_num += 1
            real_digital_num += len(real_digital_str)

    print('按数字算准确率为：{}'.format(float(pre_digital_num) / float(real_digital_num)))
    print('按整张图片算准确率为：{}'.format(float(pre_whole_num) / float(real_whole_num)))


if __name__ == '__main__':
    get_test_label_accuracy('../dataset/label/test_label.txt',
                            '../card_recognize/weights/weights_2019_06_04_09_56_37.h5')
