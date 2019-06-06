'''
预测部分
'''
from card_recognize.predict import get_predict_result, get_accuracy


def recognize_image_set(image_set_path, result_set_path, result_path):
    '''
    对 image_set_path 路径下的所有的图片做测试
    标记数字区域的图片放在 result_set_path 下
    预测的数字串放在 result_path 下
    :param image_set_path:
    :param result_set_path:
    :param result_path:
    :return: 无
    '''
    model_path = '../card_recognize/weights/weights_2019_06_04_09_56_37.h5'
    all_result_list = get_predict_result(image_set_path, result_set_path, model_path)
    # 结果写入文件
    with open(result_path, 'w') as f:
        for one in all_result_list:
            f.write(one + '\n')

    xml_set_path = '../dataset/annotation/'
    get_accuracy(result_path, xml_set_path)


if __name__ == '__main__':
    recognize_image_set('test_images', 'test_result', 'result.txt')
