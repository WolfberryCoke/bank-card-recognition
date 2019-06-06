'''
用kmeans聚类生成预测特征图的anchor box 集合
'''
import numpy as np

class Kmeans:

    def __init__(self, cluster_number, filename):
        '''
        填充参数之后执行方法
        :param cluster_number:
        :param filename:
        '''
        self.cluster_number = cluster_number
        self.filename = filename
        self.txt2clusters()

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        '''
        计算聚类中心与 bounding box 之间的IOU
        :param boxes:
        :param clusters:
        :return: [0,1]之间的值
        '''
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        '''
        k-means 核心算法
        :param boxes:
        :param k:
        :param dist:
        :return:
        '''
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        # 随机选取聚类中心
        np.random.seed()
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            # 聚类中心不再变化
            if (last_nearest == current_nearest).all():
                break
            # 更新聚类中心
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest
        return clusters

    def result2txt(self, data):
        '''
        将聚类中心坐标写入指定文件
        :param data:
        :return: 无
        '''
        f = open("model_data/yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        '''
        从一份数据中读出bounding box的所有参数
        :return: [w,h]的数组
        '''
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            # infos是*_label.txt的文件名和坐标信息列表
            infos = line.split(" ")
            length = len(infos)
            # 第0列是文件名，所以从1开始，算出矩形的宽度和高度
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "../dataset/label/train_label.txt"
    # 执行方法在构造函数中
    Kmeans(cluster_number, filename)