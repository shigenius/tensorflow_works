import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16, vgg_arg_scope
from nets.inception_v4 import inception_v4, inception_v4_arg_scope
import cv2
import argparse
import numpy as np
import time

from PIL import Image
csvimport
from pathlib import Path
import math

archs = {
    'inception_v4': {'fn': inception_v4, 'arg_scope': inception_v4_arg_scope, 'extract_point': 'PreLogitsFlatten'},
    # 'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': 'shigeNet_v1/vgg_16/fc7'}# shape=(?, 14, 14, 512) dtype=float32
    'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': 'vgg_16/fc7', 'extract_shape': (-1, 1, 1, 4096)}# shape=(?, 14, 14, 512)
}
# shigeNet_v1/vgg_16/fc7
# 'shigeNet_v1/vgg_16/conv5/conv5_3' # shape=(?, 14, 14, 512) dtype=float32

def get_label(path):
    return {l.split(" ")[0]: l.split(" ")[1].replace("\n", "") for l in open(path, "r")}

def general_object_recognition(input_placeholder, num_classes, extractor_name):
    logits, end_points = vgg_16(input_placeholder, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    predictions = tf.nn.softmax(logits, name='Predictions')
    feature = end_points[archs[extractor_name]['extract_point']]
    return predictions, feature

def specific_object_recognition(feature_c, feature_o, num_classes, keep_prob, extractor_name='vgg_16'):
    end_points = {}
    with tf.variable_scope('shigeNet_v1', 'shigeNet_v1', reuse=None) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):

                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

            return end_points


def calc_coordinate_from_index(indices, image_shape, window_size, stride):
    # Don't use TF module!
    w = image_shape[1]
    h = image_shape[0]
    x_points = np.arange(0, w - window_size, stride)
    y_points = np.arange(0, h - window_size, stride)
    num_xp = x_points.shape[0]
    #num_yp = y_points.shape[0]
    #print("indices:", indices, "image_shape:", image_shape, "window_size:", window_size, "stride:", stride)
    #print("w:", w, "h:",h)
    #print(x_points)
    #print(y_points)
    #print("num_of_windows_in_one_row", num_xp)
    #print("num_of_windows_in_one_col", num_yp)

    coordcates = []
    for i in indices:
        xp = i % num_xp
        yp = int(i / num_xp)
        x = x_points[xp]
        y = y_points[yp]
        # print(i, (xp, yp), (x, y))
        coordcates.append((i, (x, y), (window_size, window_size)))

    return coordcates


def sliding_window_tf(input_placeholder, window_size, stride):
    input_shape = tf.shape(input_placeholder)

    # # input_shapeが可変な場合(placeholderから計算してきた値)，for文は使えない．
    # indices = [[i, j] for i in range(0, input_shape[1] - window_size, stride)
    #                     for j in range(0, input_shape[0] - window_size, stride)]
    # indices_tensor = tf.constant(indices)

    xx, yy = tf.meshgrid(tf.range(0, input_shape[0] - window_size, stride),
                         tf.range(0, input_shape[1] - window_size, stride), indexing='ij')
    xxsq = tf.expand_dims(tf.reshape(xx, [-1]), 1)
    yysq = tf.expand_dims(tf.reshape(yy, [-1]), 1)
    indices = tf.concat([xxsq, yysq], -1)

    return tf.map_fn(
        lambda x: tf.strided_slice(input_placeholder, [x[0], x[1]], [x[0] + window_size, x[1] + window_size]),
        indices,
        dtype=tf.float32)


def selective_search_tf():
    pass

def show_variables_in_ckpt(path):
    prefix = path
    saver = tf.train.import_meta_graph("{}.meta".format(prefix))
    sess = tf.Session()
    saver.restore(sess, prefix)
    graph=tf.get_default_graph()
    for op in graph.get_operations():
        if "fc3" in op.name:
            print(op)


def draw_result(img, coordinates, labels, slabel):
    # coordinate : (index, (ulx, uly), (window_size, window_size))
    # labels : [predicted_label_number, ...]
    for i in range(len(coordinates)):
        label_name = slabel[str(labels[i])]
        ulx, uly = coordinates[i][1]
        ww, wh = coordinates[i][2]

        w = int(ww / 2)
        h = int(wh / 2)
        x = int(ulx + w)
        y = int(uly + h)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2) # drow object rectangle

        # drow label
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x + w, y - h), (125, 125, 125), -1)
        # cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)
        cv2.putText(img, label_name, (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)

def get_annotation(image_path, txtname="subwindow_log.txt"):
    img_p = Path(image_path)
    img_obj_name = img_p.parents[1].name
    cropped_dir_p = Path(str(img_p.parent)+'_cropped')
    log_p = cropped_dir_p/txtname
    assert log_p.exists(), 'Does not exist :{0}'.format(str(log_p))

    img_id = int(img_p.stem.split('_')[1])# フレーム番号

    anno = None
    with open(str(log_p), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if int(row[0]) == img_id:
                anno = row
                break

    return anno # [frame, center_x, center_y, size_x, size_y]

def get_distance(x1, y1, x2, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d

def calc_iou(predict_box, gt_box):
    # box define: [lu_x, lu_y, rd_x, rd_y]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(predict_box[0], gt_box[0])
    yA = max(predict_box[1], gt_box[1])
    xB = min(predict_box[2], gt_box[2])
    yB = min(predict_box[3], gt_box[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (predict_box[2] - predict_box[0] + 1) * (predict_box[3] - predict_box[1] + 1)
    boxBArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    precision = interArea / boxAArea

    # return the intersection over union value
    return iou, precision


def eval(args):
    extractor_name = args.extractor
    image_size = archs[extractor_name]['fn'].default_image_size

    # get labels
    glabel = get_label(args.glabel)
    slabel = get_label(args.slabel)
    num_of_gclass = len(glabel.keys())
    num_of_sclass = len(slabel.keys())

    target_category = ['seesaa', 'shisa']
    target_label = [i for i in glabel.items() if i[1] in target_category][0]

    with tf.name_scope('general'):
        # define placeholders
        with tf.name_scope('input'):
            input_placeholder = tf.placeholder(dtype="float32", shape=(None,  None,  3))
            is_training = tf.placeholder(dtype="bool")  # train flag

        stride = int(image_size/2)


        resized_input = tf.image.resize_images(input_placeholder, (image_size, image_size))
        cropps = sliding_window_tf(input_placeholder=input_placeholder,
                                   window_size=image_size,
                                   stride=stride)

        # general object detection
        predictions, feature = general_object_recognition(cropps, num_of_gclass, extractor_name)


        candidate_index = tf.reshape(tf.where(tf.equal(tf.argmax(predictions, 1), int(target_label[0]))), [-1]) #targetlabelと同じ値なpredのindexを返す
        candidate_feature = tf.reshape(tf.gather(feature, candidate_index, axis=0), archs[extractor_name]['extract_shape']) # 指定したindicesで抜き出す
        print(candidate_feature)

        _, bg_feature = general_object_recognition(resized_input[tf.newaxis, :, :, :], num_of_gclass, extractor_name)
        bg_feature_expand = tf.reshape(tf.tile(bg_feature, [tf.shape(candidate_feature)[0],1,1,1]), archs[extractor_name]['extract_shape'])# batch sizeをcand featureと合わせる．

        variables_to_restore_g = slim.get_variables_to_restore()
        restorer_g = tf.train.Saver(variables_to_restore_g)

    with tf.name_scope('specific'):
        with tf.name_scope('input'):
            candfeat_placeholder = tf.placeholder(dtype="float32", shape=(None, 1, 1, 4096))
            bgfeat_placeholder = tf.placeholder(dtype="float32", shape=(None, 1, 1, 4096))
            keep_prob = tf.placeholder(dtype="float32")

        # end_points_s = specific_object_recognition(candidate_feature, bg_feature_expand, num_of_sclass, keep_prob, extractor_name='vgg_16')
        end_points_s = specific_object_recognition(candfeat_placeholder, bgfeat_placeholder, num_of_sclass, keep_prob,
                                                   extractor_name='vgg_16') # test用
        predictions_s = end_points_s["Predictions"]
        predict_labels = tf.argmax(predictions_s, 1)

        variables_to_restore_s = set(slim.get_variables_to_restore()) - set(variables_to_restore_g)
        # variables_to_restore_s = {name_in_checkpoint(var): var for var in variables_to_restore_s}
        # print(variables_to_restore_s)

        restorer_s = tf.train.Saver(variables_to_restore_s)

    comparison = tf.not_equal(tf.size(candidate_index), 0) # 候補領域がなければFalse
    y = tf.cond(comparison, lambda: predict_labels, lambda: tf.constant([], dtype=tf.int64)) # 候補領域がなければ空tensorを返す

    # 使用するGPUメモリを80%までに制限
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.8
        )
    )

    with tf.Session(config=config) as sess:

        # initialize and restore model
        sess.run(tf.global_variables_initializer())
        restorer_g.restore(sess, args.gr)
        print("Restored general recognition model:", args.gr)
        restorer_s.restore(sess, args.sr)
        print("Restored specific recognition model:", args.sr)

        # prepare test set
        with open(args.test_txt, 'r') as f:
            f_ = [line.rstrip().split() for line in f]

        data = [[l, get_annotation(l[0])] for l in f_] # data: [[(path_str, label), [frame, center_x, center_y, size_x, size_y]],...]
        data = [l for l in data if l[1] != None] # annotationを取得できなかった画像は飛ばす

        # log
        f = open(args.log, 'w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['path', 'gt_label', 'IoU', 'AveragePrecision', 'Recall', 'detect_succeed', 'running_time'])

        # iterative run
        for gt in data:# gt: [(path_str, label), [frame, center_x, center_y, size_x, size_y]
            input_image = np.array(Image.open(gt[0][0])) / 255
            ious = []
            ap = []
            gt_box = (float(gt[1][1]) - float(gt[1][3]), float(gt[1][2]) - float(gt[1][4]), float(gt[1][1]) + float(gt[1][3]),float(gt[1][2]) + float(gt[1][4]))  # one box in one image :[lu_x, lu_y, rd_x, rd_y]
            gt_box = [int(i) for i in gt_box]
            hoge = np.zeros(shape=input_image.shape[0:2])
            hoge[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = 1  # gtの範囲を1にする

            print(gt[0][0])

            start_time = time.time()
            # detection
            cand_index, cand_feat, bg_feat = sess.run([candidate_index, candidate_feature, bg_feature_expand], feed_dict={input_placeholder: input_image})
            if len(cand_index) != 0: # 候補領域があれば
                # instance recognition
                pred = sess.run(predict_labels, feed_dict={candfeat_placeholder: cand_feat, bgfeat_placeholder: bg_feat, keep_prob: 1.0})
                elapsed_time = time.time() - start_time

                coordinates = calc_coordinate_from_index(indices=cand_index, image_shape=input_image.shape, window_size=image_size, stride=stride)

                for i, item in enumerate(coordinates):  # item: (i, (x, y), (window_size, window_size))
                    p_label = pred[i]
                    g_label = gt[0][1]
                    if p_label == g_label:
                        pred_box = (item[1][0], item[1][1], item[1][0]+item[2][0], item[1][1]+item[2][1])
                        iou, precision = calc_iou(pred_box, gt_box)
                        ious.append(iou)
                        ap.append(precision)

                        hoge[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]] = 0 # recallの計算に用いる
                    else:
                        ious.append(0)
                        ap.append(0)

                hit_gtArea = hoge[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]]
                recall = np.sum(hit_gtArea == 0) / hit_gtArea.size
                average_iou = sum(ious) / len(ious)
                average_prec = sum(ap) / len(ap)


            else: # 候補領域がない場合
                elapsed_time = time.time() - start_time
                average_iou = 0.0
                average_prec = 0.0
                recall = 0.0

            print("iou:", average_iou, ious)
            print("Average precision:", average_prec, ap)
            print("recall", recall)

            writer.writerow([gt[0][0], gt[0][1], average_iou, average_prec, recall, len(cand_index) != 0, elapsed_time])
            print("running time:", elapsed_time, "(s)")

        f.close()

            # # 中心点からの距離による評価書きかけ
            # TP_score = []
            # for i, item in enumerate(coordinates): # item: (i, (x, y), (window_size, window_size))
            #     # print("index:", item[0], "(x, y):", item[1], "predict label:", pred[i], slabel[str(pred[i])])
            #     p_label = pred[i]
            #     g_label = a[0][1]
            #     if p_label == g_label: # if prediction is TP
            #         # with normalize
            #         p_center_x = (item[1][0] - (item[2][0] / 2)) / input_image.shape[1]
            #         p_center_y = (item[1][1] - (item[2][1] / 2)) / input_image.shape[0]
            #         g_center_x = a[1][1] / input_image.shape[1]
            #         g_center_y = a[1][2] / input_image.shape[0]
            #         distance = get_distance(g_center_x, g_center_y, p_center_x, p_center_y)
            #         TP_score.append(distance)
            #
            # # calc mean score
            # for i in TP_score:
            #
            #
            # FP = len(coordinates) - TP_score





            # print("\n running time:", elapsed_time, "(sec)")

            # image = cv2.imread(args.input)
            # input_image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # draw_result(image, coordinates, pred, slabel)
            # cv2.imshow('result', image)
            # cv2.waitKey(0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('test_txt', help='File path of test_set.txt', default='/Users/shigetomi/Desktop/dataset_fit_noNegative/dataset_shisa/test_orig_IDreplaced.txt')
    parser.add_argument('glabel', help='File name of general recog label txt')
    parser.add_argument('slabel', help='File name of specific recog label txt')

    parser.add_argument('-extractor', help='extractor architecture name', default='vgg_16')
    parser.add_argument('-net', help='network name', default='shigeNet')
    parser.add_argument('-gr', default='/Users/shigetomi/workspace/tensorflow_works/tf-slim/model/general_recog/vgg16_imagenet_0718.ckpt-75')
    parser.add_argument('-sr', default='/Users/shigetomi/workspace/tensorflow_works/tf-slim/model/transl_shisa.ckpt-0')
    parser.add_argument('-log', default='specific_object_detection_log.csv')


    args = parser.parse_args()

    if args.net == 'shigeNet':
        eval(args)
        # show_variables_in_ckpt(args.sr)
