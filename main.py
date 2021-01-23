from flask import Flask
from flask import request
import io
import json
from timeit import default_timer as timer
import base64
from PIL import Image
from card_recognize.predict import get_one_result, nms_inter_class
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    '''
    用于提供一个供 Android APP 调用的预测银行卡号的服务，并输出调试信息供后台记录
    '''
    start = timer()
    img_data_base64 = request.form['data']
    img_data = base64.b64decode(img_data_base64)
    end1 = timer()
    img = io.BytesIO(img_data)
    end2 = timer()
    model_path = 'G:\Bank_Card_OCR\card_recognize\weights\weights_2019_07_08_15_06_27.h5'
    image = Image.open(img)
    end3 = timer()
    predNum, predNum_blank = get_one_result(image, model_path)
    end4 = timer()
    logger.info('解码时间为：' + str(end1 - start) + 's')
    logger.info('io时间为：' + str(end2 - end1) + 's')
    logger.info('open时间为：' + str(end3 - end2) + 's')
    logger.info('预测时间为：' + str(end4 - end3) + 's')
    logger.info('总时间为：' + str(end4 - start) + 's')

    logger.info(predNum)
    ret = {'predNum': predNum, 'predNum_blank': predNum_blank}
    logger.info(json.dumps(ret))
    return json.dumps(ret)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086)
