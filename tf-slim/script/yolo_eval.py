# for http://arkouji.cocolog-nifty.com/blog/2017/07/yolotensorflow-.html
import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer

from test import Detector
from utils.pascal_voc import pascal_voc
import copy
import math

class Detector_ex(Detector):
    def __init__(self, net, weight_file, data):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

        # add
        self.data = data

    def eval(self):
        # 画像毎にdetectする処理
        # resultとannotationを比較して評価する．

        test_timer = Timer()
        load_timer = Timer()

        ious = []
        for step in range(int(len(self.data.gt_labels) / cfg.BATCH_SIZE)):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()

            # print(images.shape) # (20, 448, 448, 3)
            # print(labels.shape) # (20, 7, 7, 14)

            print("step:", step)

            test_timer.tic()
            result = self.detect_from_cvmat(images)
            test_timer.toc()

            # print("results in batch :", result)
            ious.extend(self.calc_iou(result, labels))
            mean = sum(ious) / len(ious)
            # print("ious", ious)
            print("current mean iou", mean)

            # if cfg.BATCH_SIZE == 1:
            #     # image = np.reshape(images, [self.image_size, self.image_size, 3]).astype(np.uint8)
            #     image = np.squeeze(((images + 1) / 2) * 255).astype(np.uint8) # flaot64 to uint8
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     result = self.detect(image)
            #     self.draw_result(image, result)
            #     cv2.imshow('Image', image)
            #     cv2.waitKey(0)


    def calc_iou(self, results, labels):
        # result: {list} [[class_name, x, y, w, h, confidence?], ...] if there isn't any object, result = []
        # labels: (20, 7, 7, 14)
        # IoUの計算 : （共通集合の面積 / 和集合の面積）

        # ground truthを整形する
        gt_labels = []
        for label in labels:
            cell_id = np.where(label[..., 0] == 1) # confidence == 1
            adj_label = np.concatenate([np.array(cell_id).T, label[cell_id]], axis=1)
            gt_labels.append(adj_label)
            # print(adj_label)

        # print(gt_labels)

        ious_in_batch = []
        for i, result in enumerate(results):
            gt_label = gt_labels[i] # [[cell_y, cell_x, confi, cx, cy, w, h, (classes)], ...]
            num_of_pred_boxes = len(result)
            num_of_gt_boxes = len(gt_label)
            positive_ious = []
            # print("result", ":", result)
            # print("gt_label:", gt_label)

            for pred_box in result:

                pred_class = pred_box[0]
                pred_class_id = self.classes.index(pred_class)
                pred_centerx, pred_centery = pred_box[1], pred_box[2]
                # pred_cellx = int(pred_centerx / self.image_size * self.cell_size)
                # pred_celly = int(pred_centery / self.image_size * self.cell_size)
                pred_w, pred_h = pred_box[3], pred_box[4]

                target_gt_boxes = [gt_box for gt_box in gt_label if pred_class_id in np.where(gt_box[7:] == 1)]
                # print("pred_box_class:", pred_class, pred_class_id)
                # print("target_gt_boxes", target_gt_boxes)

                target_ious = []
                for target_gt_box in target_gt_boxes:

                    # gt_cellx, gt_celly = target_gt_box[1], target_gt_box[0]
                    gt_centerx, gt_centery = target_gt_box[3], target_gt_box[4]
                    gt_w, gt_h = target_gt_box[5], target_gt_box[6]
                    # gt_class_id = np.argmax(target_gt_box[7:] - 7)
                    # gt_class = self.classes[gt_class_id]

                    pred_box = calcBox(pred_centerx, pred_centery, pred_w, pred_h)
                    gt_box = calcBox(gt_centerx, gt_centery, gt_w, gt_h)
                    iou = bb_intersection_over_union(gt_box, pred_box)
                    target_ious.append(iou)

                if len(target_ious) == 0:
                    positive_ious.append(0)
                else:
                    positive_ious.append(max(target_ious))# 最も値が高いiouを格納

            # print("num_of_pred_boxes", num_of_pred_boxes)
            # print("num_of_gt_boxes", num_of_gt_boxes)
            # print("positive_ious", positive_ious)

            ious = copy.deepcopy(positive_ious)
            offset = num_of_gt_boxes + num_of_pred_boxes - len(positive_ious) * 2
            if offset > 0:
                ious.extend([0 for _ in range(offset)])

            mean = sum(ious) / len(ious)
            # print("ious", ious)
            # print("mean of IoU at single image", mean)
            ious_in_batch.append(mean)

        return ious_in_batch



def calcBox(center_x, center_y, w, h):
    # box : [lu_x, lu_y, rd_x, rd_y]
    return [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]

def bb_intersection_over_union(boxA, boxB):
    # box : [lu_x, lu_y, rd_x, rd_y]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="save.ckpt-15000", type=str)
    parser.add_argument('--weight_dir', default='dataset_shisa/output/2018_10_14_20_04/', type=str)
    # parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    # parser.add_argument('--weight_dir', default='weights', type=str)

    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--image', default='test/person.jpg', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(is_training=True)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)

    pascal = pascal_voc('test')

    detector = Detector_ex(yolo, weight_file, pascal)
    detector.eval()


if __name__ == '__main__':
    main()
