'''
训练部分
'''
import datetime

import keras.backend as K
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Lambda
from keras.models import Model
from card_recognize.yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from util.image_handler import get_data
from util.model_data_handler import get_classes, get_anchors
from card_recognize.kmeans import Kmeans
from card_recognize.data_make import data_split, add_path
import os

# 去掉GPU的信息提示
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 获取网络训练当前时间
nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# 超参数
batch_size = 1
iterations = 100


def main():
    '''
    模型训练的主函数
    :return:无
    '''
    # 填写类别文件
    classes_path = 'model_data/classes.txt'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'number']
    write_classes(classes_path, classes)

    # 划分训练测试集
    data_split()
    add_path()

    # 通过kmeans找聚类中心
    cluster_number = 9
    annotation_path = '../dataset/label/train_label.txt'
    Kmeans(cluster_number, annotation_path)

    # 创建模型
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)
    model = create_model(input_shape, anchors, len(class_names))

    # 做训练
    log_dir = ('weights/weights_%s' % nowTime + '.h5')
    train(model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir)


def write_classes(classes_path, classes):
    with open(classes_path, 'w') as f:
        for one in classes:
            f.write(one + '\n')


def train(model, annotation_path, input_shape, anchors, num_classes, log_dir):
    '''
    训练网络
    :param model:网络
    :param annotation_path:标记文件目录
    :param input_shape:输入张量大小
    :param anchors:  shape 为 (N, 2)，值为聚类中心长宽的数组
    :param num_classes: 标签个数
    :param log_dir: 保存路径
    :return: 无
    '''
    # 该回调函数将日志信息写入TensorBorad
    logging = TensorBoard(log_dir=log_dir)
    # 该回调函数将在每5个epoch后保存最优的模型参数
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    # 当评价指标不再提升时，减少学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # 当监测值不再改善时，该回调函数将中止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # 训练验证集9：1分类
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, validate on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    model.fit_generator(generator=data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=iterations)
    model.save_weights(log_dir)
    print('model has been trained!\n')


def create_model(input_shape, anchors, num_classes):
    '''
    定义网络
    :param input_shape: 输入图片大小
    :param anchors:
    :param num_classes:
    :return: 模型 model
    '''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''
    从划分数据的结果文件中读取图片并转换成适合送进网络的图片格式
    :param annotation_lines:
    :param batch_size:
    :param input_shape:
    :param anchors:
    :param num_classes:
    :return:
    '''
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_data(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # 将标记数据转换成YOLO网络需要的数据格式
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


if __name__ == '__main__':
    main()
