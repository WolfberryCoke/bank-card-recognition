import os, shutil
from PIL import Image


def select(src_image_path, dest_image_path, src_xml_path, dest_xml_path):
    '''
    剔除脏数据，即分辨率小于544*544的图片
    :param src_image_path:
    :param dest_image_path:
    :param src_xml_path:
    :param dest_xml_path:
    :return:
    '''
    cnt = 0
    for file in os.listdir(src_image_path):
        file_path = os.path.join(src_image_path, file)
        image = Image.open(file_path)
        # print(image.size[0])
        # print(image.size[1])
        if (image.size[0] < 544 and image.size[1] < 544):
            cnt += 1
            continue
        image.save(dest_image_path + '\\' + file[0:7] + '.jpg')
        for file2 in os.listdir(src_xml_path):
            if file2[0:7] == file[0:7]:
                shutil.copyfile(os.path.join(src_xml_path, file2), os.path.join(dest_xml_path, file2))
                break

    print('有' + cnt + '份脏数据')


def file_rename(src_image_path, dest_image_path, src_xml_path, dest_xml_path):
    for file in os.listdir(src_image_path):
        if 'image' in file:
            index = file.find('(')
            print(index)
            index2 = file.find(')')
            print(index2)
            print(file[int(index),int(index2)])


if __name__ == "__main__":
    src_image_path = 'src_image'
    dest_image_path = 'dest_image'
    src_xml_path = 'src_xml'
    dest_xml_path = 'dest_xml'
    file_rename(src_image_path, dest_image_path, src_xml_path, dest_xml_path)
