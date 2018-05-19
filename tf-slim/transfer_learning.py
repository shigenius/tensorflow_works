# after running : models/tutorials/image/imagenet/classify_image.py

import tensorflow as tf
import numpy as np

import argparse
from datetime import datetime
from TwoInputDataset import TwoInputDataset
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow.contrib.slim as slim

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from nets.inception_v4 import inception_v4, inception_v4_arg_scope

def shigeNet_v1(cropped_images, original_images, num_classes, keep_prob=1.0, is_training=True, scope='shigeNet_v1', reuse=None):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v1', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(inception_v4_arg_scope()):
                logits_c, end_points_c = inception_v4(cropped_images, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
                logits_o, end_points_o = inception_v4(original_images, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

                feature_c = end_points_c['PreLogitsFlatten']
                feature_o = end_points_o['PreLogitsFlatten']

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([feature_c, feature_o], 1)  # (?, x, y, z)

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
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points

def sandbox(args):
    MODEL_PATH = '/Users/shigetomi/Downloads/inception_v4.ckpt'
    # print_tensors_in_checkpoint_file(file_name=MODEL_PATH, tensor_name='', all_tensors=False, all_tensor_names=True)
    # print_tensors_in_checkpoint_file(file_name=MODEL_PATH, tensor_name='', all_tensors=True, all_tensor_names=True)

    image_size = inception_v4.default_image_size
    num_classes = args.num_classes # road sign

    # Define placeholders
    cropped_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size,  image_size,  3))
    original_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size, image_size, 3))
    labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes))
    keep_prob = tf.placeholder(dtype="float32")
    is_training = tf.placeholder(dtype="bool")  # train flag

    # Build the graph
    end_points = shigeNet_v1(cropped_images=cropped_images_placeholder, original_images=original_images_placeholder, num_classes=num_classes, is_training=is_training, keep_prob=keep_prob)
    predictions = end_points["Predictions"]

    # train ops
    # slim.losses.softmax_cross_entropy(predictions, labels_placeholder)
    # total_loss = slim.losses.get_total_loss()
    # tf.summary.scalar('losses/total_loss', total_loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    # train_tensor = slim.learning.create_train_op(total_loss, optimizer)


    # Get restored vars name in checkpoint
    def name_in_checkpoint(var):
        if "shigeNet_v1" in var.op.name:
            return var.op.name.replace("shigeNet_v1/", "")

    # Get vars restored
    variables_to_restore = slim.get_variables_to_restore()
    # dict of {name in checkpoint: var.op.name}
    variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if "InceptionV4" in var.op.name}
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()# save all vars

    # Train ops もっといい書き方があるかも(slimをつかったやつとか)
    loss = -tf.reduce_sum(labels_placeholder * tf.log(tf.clip_by_value(predictions, 1e-10, 1)))
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels_placeholder, 1))
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

    training_op_list = [accuracy, acc_summary_train, loss_summary_train]
    val_op_list = [accuracy, acc_summary_test, loss_summary_test]


    dataset = TwoInputDataset(train1=args.train_c, train2=args.train_o, test1=args.test_c, test2=args.test_o,
                              num_classes=num_classes, image_size=image_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, MODEL_PATH)
        print("Model restored from:", MODEL_PATH)

        # summary
        if args.cbflag != None:  # if running cross-valid
            summary_writer = tf.summary.FileWriter(
                args.logdir + '/twostep/' + args.cbflag + '/' + datetime.now().isoformat(), sess.graph)
        else:
            summary_writer = tf.summary.FileWriter(args.logdir + '/twostep/' + datetime.now().isoformat(), sess.graph)

        # Train cycle
        for step in range(args.max_steps):
            dataset.shuffle()

            # Train proc
            for i in range(int(len(dataset.train1_path) / args.batch_size)):  # i : batch index

                cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, i)

                # batch train
                sess.run(train_step, feed_dict={cropped_images_placeholder: cropped_batch,
                                                original_images_placeholder: orig_batch,
                                                labels_placeholder: labels,
                                                keep_prob: args.dropout_prob,
                                                is_training: True})
                # # freezeの確認用
                # var = slim.get_variables_by_name("shigeNet_v1/InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/weights")
                # print(var)
                # print(sess.run(var))

            # Get accuracy of train
            dataset.shuffle()
            cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, 0)

            result = sess.run(training_op_list,
                              feed_dict={cropped_images_placeholder: cropped_batch,
                                         original_images_placeholder: orig_batch,
                                         labels_placeholder: labels,
                                         keep_prob: 1.0,
                                         is_training: False})

            # Write summary in Tensorboard
            for j in range(1, len(result)):
                summary_writer.add_summary(result[j], step)

            print("step %d : training batch(size=%d) accuracy: %g" % (step, args.batch_size, result[0]))

            # Validation proc
            cropped_test_batch, orig_test_batch, test_labels = dataset.getTestData()  # get Full size test set
            val_result = sess.run(val_op_list,
                                  feed_dict={cropped_images_placeholder: cropped_test_batch,
                                             original_images_placeholder: orig_test_batch,
                                             labels_placeholder: test_labels,
                                             keep_prob: 1.0,
                                             is_training: False})

            # Write valid summary
            for j in range(1, len(val_result)):
                summary_writer.add_summary(val_result[j], step)
            print("  Full test set accuracy %g" % (val_result[0]))

            # Save checkpoint model
            saver.save(sess, args.save_path)

        print("all process finished　without a hitch. save the trained model at :", args.save_path)




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
    parser.add_argument('--train_c', help='File name of train data(cropped)',   default='/Users/shigetomi/Desktop/dataset_walls/train2.txt')
    parser.add_argument('--train_o', help='File name of train data (original)', default='/Users/shigetomi/Desktop/dataset_walls/train1.txt')
    parser.add_argument('--test_c', help='File name of test data(cropped)',     default='/Users/shigetomi/Desktop/dataset_walls/test2.txt')
    parser.add_argument('--test_o', help='File name of test data (original)',   default='/Users/shigetomi/Desktop/dataset_walls/test1.txt')

    parser.add_argument('--max_steps', '-s', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=20)
    parser.add_argument('--num_classes', '-nc', type=int, default=6)

    parser.add_argument('--pb_path', '-pb', default='/Users/shigetomi/Downloads/imagenet/classify_image_graph_def.pb', help='FullPath of inception-v3 graph (protobuffer)')
    parser.add_argument('--restore', '-r', default=None, help='FullPath of restoring model')
    parser.add_argument('--save_path', '-save', default='/Users/shigetomi/workspace/tensorflow_works/model/twostep.ckpt', help='FullPath of saving model')

    parser.add_argument('--logdir', '-log', default='/Users/shigetomi/workspace/tensorflow_works/log/', help='TensorBoard log')

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.8)

    parser.add_argument('--cbflag', '-cb', default=None) # usually, dont use this


    args = parser.parse_args()

    sandbox(args)
    # train(args)