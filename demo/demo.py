'''
预测部分
'''
from card_recognize.predict import get_predict_result


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
    model_path = '../card_recognize/weights/weights_2019_07_08_15_06_27.h5'
    # model_path = '../card_recognize/weights/weights_2019_06_05_18_16_40.h5'
    all_result_list = get_predict_result(image_set_path, result_set_path, model_path)
    # for i in range(len(all_result_list)):
    #     get_blank_space(all_result_list[i])
     # 结果写入文件
    with open(result_path, 'w') as f:
        for one in all_result_list:
            f.write(one + '\n')


if __name__ == '__main__':
    recognize_image_set('test_images', 'test_result', 'result.txt')
