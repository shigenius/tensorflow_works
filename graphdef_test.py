# after running : models/tutorials/image/imagenet/classify_image.py

import tensorflow as tf
import numpy as np

import argparse
from datetime import datetime

from TwoInputDataset import TwoInputDataset
from tensorflow.contrib.layers.python.layers.layers import batch_norm
import cv2
#MODEL_PATH = '/Users/shigetomi/Downloads/imagenet/classify_image_graph_def.pb'
#MODEL_PATH = '/home/akalab/classify_image_graph_def.pb' # pre-trained inception v3 model
# IMAGE_PATH = '/Users/shigetomi/Desktop/samplepictures/image_0011.jpg'

#
# def sandbox():
#     model_path = '/home/akalab/classify_image_graph_def.pb'  # pre-trained inception v3 model
#     # 画像の読み込み
#     if not tf.gfile.Exists(IMAGE_PATH):
#         tf.logging.fatal('File does not exist %s', IMAGE_PATH)
#     image_data = tf.gfile.FastGFile(IMAGE_PATH, 'rb').read()
#     print(image_data)
#
#     create_graph(model_path)
#
#     with tf.Session() as sess:
#
#         # graph内のtensor名は下記で確認できる．
#         assing_ops = tf.Graph.get_operations(sess.graph)
#         for op in assing_ops:
#             print("operation name :", op.name, ", op_def.name :" ,op.op_def.name)
#             for outputname in op.outputs:
#                 print("output :", outputname) # 出力テンソルの名前
#
#
#         logits = sess.run('softmax:0', feed_dict={'DecodeJpeg/contents:0': image_data}) # 出力の取得はテンソル名で指定できる．
#         logits = np.squeeze(logits)
#
#         # print(logits.shape, logits)
#
#         # 最もスコアが高いラベルとそのスコアを出力
#         import sys
#         sys.path.append('/Users/shigetomi/workspace/models/tutorials/image/imagenet/')
#         from classify_image import NodeLookup
#
#         NUM_TOP_PREDICTIONS = 5
#
#         node_lookup = NodeLookup(label_lookup_path='/Users/shigetomi/Downloads/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt', uid_lookup_path='/Users/shigetomi/Downloads/imagenet/imagenet_synset_to_human_label_map.txt')
#         top_k = logits.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
#         for node_id in top_k:
#             human_string = node_lookup.id_to_string(node_id)
#             score = logits[node_id]
#             print('%s (score = %.5f)' % (human_string, score))
#
#
#         # extractor として(deep feartures?)
#         pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#         #print("pool3", sess.run(pool3), pool3.shape) # 重みの出力？
#
#         pool3_features = sess.run(pool3, {'DecodeJpeg/contents:0': image_data})
#         # pool3_features = np.squeeze(pool3_features)
#
#         print("pool3_features", type(pool3_features), pool3_features, pool3_features.shape) #(2048,)


def create_graph(pb_path):
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def architecture(features_placeholder, images_placeholderB, keep_prob, is_training, num_classes, image_size):
    with tf.variable_scope('SimpleCNN') as scope:
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.get_variable(name, initializer=initial, trainable=True)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.get_variable(name, initializer=initial, trainable=True)

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        def batch_norm_layer(x, train_phase, scope_bn='bn'):
            bn_train = batch_norm(x, decay=0.999, epsilon=1e-3, center=True, scale=True,
                                  updates_collections=None,
                                  is_training=True,
                                  reuse=None,  # is this right?
                                  trainable=True,
                                  scope=scope_bn)
            bn_inference = batch_norm(x, decay=0.999, epsilon=1e-3, center=True, scale=True,
                                      updates_collections=None,
                                      is_training=False,
                                      reuse=True,  # is this right?
                                      trainable=True,
                                      scope=scope_bn)
            z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
            return z

        x_image = tf.reshape(images_placeholderB, [-1, image_size, image_size, 3])

        with tf.variable_scope('conv1') as scope:
            W_conv1 = weight_variable([11, 11, 3, 64], "w")
            b_conv1 = bias_variable([64], "b")
            h_conv1 = tf.nn.relu(
                batch_norm_layer(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1,
                                 is_training))

        with tf.variable_scope('conv2') as scope:
            W_conv2 = weight_variable([3, 3, 64, 64], "w")
            b_conv2 = bias_variable([64], "b")
            h_conv2 = tf.nn.relu(
                batch_norm_layer(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2,
                                 is_training))

        with tf.variable_scope('conv3') as scope:
            W_conv3 = weight_variable([3, 3, 64, 128], "w")
            b_conv3 = bias_variable([128], "b")
            h_conv3 = tf.nn.relu(
                batch_norm_layer(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding="VALID") + b_conv3,
                                 is_training))

        with tf.variable_scope('conv4') as scope:
            W_conv4 = weight_variable([3, 3, 128, 128], "w")
            b_conv4 = bias_variable([128], "b")
            h_conv4 = tf.nn.relu(
                batch_norm_layer(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 2, 2, 1], padding="VALID") + b_conv4,
                                 is_training))

        with tf.name_scope('concat') as scope:
            h_conv4_flat = tf.reshape(h_conv4, [-1, 6 * 6 * 128])
            incep_pool3_features_flat = tf.reshape(features_placeholder, [-1, 2048])
            h_concated = tf.concat([h_conv4_flat, incep_pool3_features_flat], 1)  # (?, x, y, z)

        with tf.variable_scope('fc1') as scope:
            W_fc1 = weight_variable([6 * 6 * 128 + 2048, 256], "w")
            b_fc1 = bias_variable([256], "b")
            h_fc1 = tf.nn.relu(batch_norm_layer(tf.matmul(h_concated, W_fc1) + b_fc1, is_training))
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.variable_scope('fc2') as scope:
            W_fc2 = weight_variable([256, num_classes], "w")
            b_fc2 = bias_variable([num_classes], "b")
            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope('softmax') as scope:
            y = tf.nn.softmax(h_fc2)

        return y

def new_arch1(features_placeholderA, features_placeholderB, keep_prob, is_training, num_classes):
    with tf.variable_scope('new_arch1') as scope:
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.get_variable(name, initializer=initial, trainable=True)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.get_variable(name, initializer=initial, trainable=True)

        def batch_norm_layer(x, train_phase, scope_bn='bn'):
            bn_train = batch_norm(x, decay=0.999, epsilon=1e-3, center=True, scale=True,
                                  updates_collections=None,
                                  is_training=True,
                                  reuse=None,  # is this right?
                                  trainable=True,
                                  scope=scope_bn)
            bn_inference = batch_norm(x, decay=0.999, epsilon=1e-3, center=True, scale=True,
                                      updates_collections=None,
                                      is_training=False,
                                      reuse=True,  # is this right?
                                      trainable=True,
                                      scope=scope_bn)
            z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
            return z

        with tf.name_scope('concat') as scope:
            incep_pool3_featuresA_flat = tf.reshape(features_placeholderA, [-1, 2048])
            incep_pool3_featuresB_flat = tf.reshape(features_placeholderB, [-1, 2048])
            h_concated = tf.concat([incep_pool3_featuresA_flat, incep_pool3_featuresB_flat], 1)  # (?, x, y, z)

        with tf.variable_scope('fc1') as scope:
            W_fc1 = weight_variable([2048 * 2, 1000], "w")
            b_fc1 = bias_variable([1000], "b")
            h_fc1 = tf.nn.relu(batch_norm_layer(tf.matmul(h_concated, W_fc1) + b_fc1, is_training))
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.variable_scope('fc2') as scope:
            W_fc2 = weight_variable([1000, 256], "w")
            b_fc2 = bias_variable([256], "b")
            h_fc2 = tf.nn.relu(batch_norm_layer(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, is_training))
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        with tf.variable_scope('fc3') as scope:
            W_fc3 = weight_variable([256, num_classes], "w")
            b_fc3 = bias_variable([num_classes], "b")
            h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

        with tf.name_scope('softmax') as scope:
            y = tf.nn.softmax(h_fc3)

    return y

def train(args):
    image_size = 227
    num_classes = args.num_classes
    dataset = TwoInputDataset(train1=args.train1, train2=args.train2, test1=args.test1, test2=args.test2,  num_classes=num_classes, image_size=image_size)

    create_graph(pb_path=args.pb_path) # graphを作成する(restore)

    with tf.Session() as sess:
        # placeholders
        images_placeholderB = tf.placeholder(dtype="float32", shape=(None, image_size * image_size * 3))
        labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool") # train flag

        image_for_extractor_placeholder = tf.placeholder(dtype="float32", shape=(image_size*image_size*3))
        #features_placeholder = tf.placeholder(dtype="float32", shape=(None, 1, 1, 2048))# pool3 features
        features_placeholderA = tf.placeholder(dtype="float32", shape=(None, 1, 1, 2048))
        features_placeholderB = tf.placeholder(dtype="float32", shape=(None, 1, 1, 2048))

        # define archtecture
        #y = architecture(features_placeholder, images_placeholderB, keep_prob, is_training, num_classes, image_size)
        y = new_arch1(features_placeholderA, features_placeholderB, keep_prob, is_training, num_classes)

        # train & test operations
        loss = -tf.reduce_sum(labels_placeholder * tf.log(tf.clip_by_value(y, 1e-10, 1)))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.name_scope('train') as scope:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
            acc_summary_train = tf.summary.scalar("train_accuracy", accuracy)
            loss_summary_train = tf.summary.scalar("train_loss", loss)

        with tf.name_scope('test') as scope:
            acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)
            loss_summary_test = tf.summary.scalar("test_loss", loss)

        # restore checkpoint
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, args.restore)
            print("Model restored from : ", args.restore)

        # initialize
        sess.run(tf.global_variables_initializer())

        # summary
        if args.cbflag != None: # if running cross-valid
            summary_writer = tf.summary.FileWriter(args.logdir + '/twostep/' + args.cbflag + '/' + datetime.now().isoformat(), sess.graph)
        else:
            summary_writer = tf.summary.FileWriter(args.logdir + '/twostep/' + datetime.now().isoformat(), sess.graph)

        training_op_list = [accuracy, acc_summary_train, loss_summary_train]
        val_op_list = [accuracy, acc_summary_test, loss_summary_test]

        # feature extract operations
        npimage2tensor_op = tf.convert_to_tensor(
            tf.cast(tf.reshape(image_for_extractor_placeholder, [image_size, image_size, 3])[:, ::-1], dtype='uint8')) # uintでいいの？
        encode_op = tf.image.encode_jpeg(npimage2tensor_op)
        get_incep_pool3_op = sess.graph.get_tensor_by_name('pool_3:0')

        # *inner function
        def feature_extract(batch):
            incep_features_batch = np.zeros((len(batch), 1, 1, 2048))
            for j, image in enumerate(batch):
                # ndarray image -> tensor -> encoded image
                encoded_data = sess.run(encode_op, {image_for_extractor_placeholder: image})
                incep_features_batch[j] = sess.run(get_incep_pool3_op, {'DecodeJpeg/contents:0': encoded_data})
            return incep_features_batch

        # train cycle
        for step in range(args.max_steps):
            dataset.shuffle()
            # batch proc
            for i in range(int(len(dataset.train2_path) / args.batch_size)):  # i : batch index

                batchA, batchB, labels = dataset.getTrainBatch(args.batch_size, i)

                # get features from pre-trained inception-v3 model
                incep_features_batchA = feature_extract(batchA)
                incep_features_batchB = feature_extract(batchB)

                # batch train
                # sess.run(train_step, feed_dict={features_placeholder: incep_features_batchA,
                #                                 images_placeholderB: batchB,
                #                                 labels_placeholder: labels,
                #                                 keep_prob: args.dropout_prob,
                #                                 is_training: True})

                sess.run(train_step, feed_dict={features_placeholderA: incep_features_batchA,
                                                features_placeholderB: incep_features_batchB,
                                                labels_placeholder: labels,
                                                keep_prob: args.dropout_prob,
                                                is_training: True})

                # at final batch proc (output accuracy)
                if i >= int(len(dataset.train2_path) / args.batch_size) - 1:
                    dataset.shuffle()
                    # result = sess.run(training_op_list,
                    #                   feed_dict={features_placeholder: incep_features_batchA,
                    #                              images_placeholderB: batchB,
                    #                              labels_placeholder: labels,
                    #                              keep_prob: 1.0,
                    #                              is_training: False})

                    result = sess.run(training_op_list,
                                      feed_dict={features_placeholderA: incep_features_batchA,
                                                features_placeholderB: incep_features_batchB,
                                                 labels_placeholder: labels,
                                                 keep_prob: 1.0,
                                                 is_training: False})

                    # write summary in Tensorboard
                    for j in range(1, len(result)):
                        summary_writer.add_summary(result[j], step)

                    print("step %d : training final-batch(size=%d) accuracy: %g" % (step, args.batch_size, result[0]))


                    # validation proc
                    test_dataA, test_dataB, test_labels = dataset.getTestData() # get Full size test set

                    # get features
                    incep_features_test_dataA = feature_extract(test_dataA)
                    incep_features_test_dataB = feature_extract(test_dataB)

                    # val_result = sess.run(val_op_list,
                    #                       feed_dict={features_placeholder:incep_features_test_dataA,
                    #                                  images_placeholderB: test_dataB,
                    #                                  labels_placeholder: test_labels,
                    #                                  keep_prob: 1.0,
                    #                                  is_training: False})
                    val_result = sess.run(val_op_list,
                                          feed_dict={features_placeholderA: incep_features_test_dataA,
                                                     features_placeholderB: incep_features_test_dataB,
                                                     labels_placeholder: test_labels,
                                                     keep_prob: 1.0,
                                                     is_training: False})

                    # write valid summary
                    for j in range(1, len(val_result)):
                        summary_writer.add_summary(val_result[j], step)

                    # print(" ramdom test batch((size=%d)) accuracy %g" % (args.batch_size, val_result[0]))
                    print("  Full test set accuracy %g" % (val_result[0]))
                    # print("incep_pool3 weight? :", sess.run(get_incep_pool3_op), get_incep_pool3_op.shape)

                    # save checkpoint model
                    saver.save(sess, args.save_path)

        print("all process finished　without a hitch. save the trained model at :", args.save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train1', help='File name of train data(cropped)',   default='/Users/shigetomi/Desktop/dataset_walls/train2.txt')
    parser.add_argument('--train2', help='File name of train data (original)', default='/Users/shigetomi/Desktop/dataset_walls/train1.txt')
    parser.add_argument('--test1', help='File name of test data(cropped)',     default='/Users/shigetomi/Desktop/dataset_walls/test2.txt')
    parser.add_argument('--test2', help='File name of test data (original)',   default='/Users/shigetomi/Desktop/dataset_walls/test1.txt')

    parser.add_argument('--max_steps', '-s', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=20)
    parser.add_argument('--num_classes', '-nc', type=int, default=6)

    parser.add_argument('--pb_path', '-pb', default='/Users/shigetomi/Downloads/imagenet/classify_image_graph_def.pb', help='FullPath of inception-v3 graph (protobuffer)')
    parser.add_argument('--restore', '-r', default=None, help='FullPath of restoring model')
    parser.add_argument('--save_path', '-save', default='/Users/shigetomi/workspace/tensorflow_works/model/twostep.ckpt', help='FullPath of saving model')

    parser.add_argument('--logdir', '-log', default='/Users/shigetomi/workspace/tensorflow_works/log/', help='TensorBoard log')

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.75)

    parser.add_argument('--cbflag', '-cb', default=None) # usually, dont use this


    args = parser.parse_args()

    # sandbox()
    train(args)
