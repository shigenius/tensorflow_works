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
from nets.vgg import vgg_16, vgg_arg_scope

archs = {
    'inception_v4': {'fn': inception_v4, 'arg_scope': inception_v4_arg_scope, 'extract_point': 'PreLogitsFlatten'},
    'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': 'shigeNet_v1/vgg_16/fc7'}# shape=(?, 14, 14, 512) dtype=float32
}
# shigeNet_v1/vgg_16/fc7
# 'shigeNet_v1/vgg_16/conv5/conv5_3' # shape=(?, 14, 14, 512) dtype=float32


def shigeNet_v1(cropped_images, original_images, num_classes, keep_prob=1.0, is_training=True, scope='shigeNet_v1', reuse=None, extractor_name='inception_v4'):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v1', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, is_training=False, reuse=True)

                feature_c = end_points_c[archs[extractor_name]['extract_point']]
                feature_o = end_points_o[archs[extractor_name]['extract_point']]

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

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

def shigeNet_v2(cropped_images, original_images, num_classes, keep_prob=1.0, is_training=True, scope='shigeNet_v2', reuse=None, extractor_name='inception_v4'):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v2', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, is_training=True, reuse=None) # no freeze param

                feature_c = end_points_c[archs[extractor_name]['extract_point']]
                feature_o = end_points_o[archs[extractor_name]['extract_point']]

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

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


def train(args):
    extractor_name = args.extractor

    model_path = args.model_path
    image_size = archs[extractor_name]['fn'].default_image_size
    num_classes = args.num_classes # road sign
    val_fre = 5# Nstep毎にvalidate

    # Define placeholders
    with tf.name_scope('input'):
        with tf.name_scope('cropped_images'):
            cropped_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size,  image_size,  3))
        with tf.name_scope('original_images'):
            original_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size, image_size, 3))
        with tf.name_scope('labels'):
            labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool")  # train flag

        # メモreshapeはずしてみる
        tf.summary.image('cropped_images', tf.reshape(cropped_images_placeholder, [-1, image_size, image_size, 3]), max_outputs=args.batch_size)
        tf.summary.image('original_images', tf.reshape(original_images_placeholder, [-1, image_size, image_size, 3]), max_outputs=args.batch_size)

    # Build the graph
    end_points = shigeNet_v1(cropped_images=cropped_images_placeholder, original_images=original_images_placeholder, extractor_name=extractor_name, num_classes=num_classes, is_training=is_training, keep_prob=keep_prob)
    # logits = tf.squeeze(end_points["Logits"], [1, 2])
    logits = end_points["Logits"]
    predictions = end_points["Predictions"]

    # Get restored vars name in checkpoint
    def name_in_checkpoint(var):
        if "shigeNet_v1" in var.op.name:
            return var.op.name.replace("shigeNet_v1/", "")

    # Get vars restored
    variables_to_restore = slim.get_variables_to_restore()
    # dict of {name in checkpoint: var.op.name}
    variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if extractor_name in var.op.name}
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=None)# save all vars

    # Train ops
    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(labels_placeholder, logits)
        tf.summary.scalar("loss", loss)

    with tf.name_scope('train') as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = slim.optimize_loss(loss, tf.train.get_or_create_global_step(), learning_rate=args.learning_rate, optimizer='Adam')

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)

    dataset = TwoInputDataset(train_c=args.train_c, train_o=args.train_o, test_c=args.test_c, test_o=args.test_o,
                              num_classes=num_classes, image_size=image_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, model_path)
        print("Model restored from:", model_path)

        # summary
        if args.cvflag != None:  # if running cross-valid
            train_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/recorded_at_' + args.cvflag + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/recorded_at_' + args.cvflag + '/test')
        else:
            train_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/test')

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        merged = tf.summary.merge_all()

        num_batch = int(len(dataset.train_path_c) / args.batch_size)

        # Train cycle
        for step in range(args.max_steps):
            dataset.shuffle()

            # Train proc
            for i in range(num_batch-1):  # i : batch index
                # print('step%g, batch%g' % (step, i))
                cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, i)
                sess.run(train_step,
                         feed_dict={cropped_images_placeholder: cropped_batch['batch'],
                                    original_images_placeholder: orig_batch['batch'],
                                    labels_placeholder: labels,
                                    keep_prob: args.dropout_prob,
                                    is_training: True})

            # Final batch proc: get summary and train_trace
            cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, num_batch-1)
            summary, train_accuracy, train_loss, _ = sess.run([merged, accuracy, loss, train_step],
                                                              feed_dict={cropped_images_placeholder: cropped_batch['batch'],
                                                                         original_images_placeholder: orig_batch['batch'],
                                                                         labels_placeholder: labels,
                                                                         keep_prob: args.dropout_prob,
                                                                         is_training: True},
                                                              options=run_options,
                                                              run_metadata=run_metadata)
            # Write summary
            train_summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_summary_writer.add_summary(summary, step)
            print('step %d: training accuracy %g,\t loss %g' % (step, train_accuracy, train_loss))

            # Validation proc
            if step % val_fre == 0:
                num_test_batch = int(len(dataset.test_path_c) / args.batch_size)
                acc_list = []
                loss_list = []
                for i in range(num_test_batch - 1):
                    test_cropped_batch, test_orig_batch, test_labels = dataset.getTestData(args.batch_size, i)
                    summary_test, test_accuracy, test_loss = sess.run([merged, accuracy, loss],
                                                                      feed_dict={cropped_images_placeholder: test_cropped_batch['batch'],
                                                                                 original_images_placeholder: test_orig_batch['batch'],
                                                                                 labels_placeholder: test_labels,
                                                                                 keep_prob: 1.0,
                                                                                 is_training: False})
                    acc_list.append(test_accuracy)
                    loss_list.append(test_loss)

                mean_acc = sum(acc_list)/len(acc_list)
                mean_loss = sum(loss_list) / len(loss_list)


                # Write valid summary
                test_summary_writer.add_summary(summary_test, step)
                test_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="Valid/accuracy", simple_value=mean_acc)
                ]), step)
                test_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="Valid/loss", simple_value=mean_loss)
                ]), step)

                print('step %d: test mean accuracy %g,\t mean loss %g' % (step, mean_acc, mean_loss))
                # Save checkpoint model
                saver.save(sess, args.save_path, global_step=step)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_c', help='File name of train data(cropped)',   default='/Users/shigetomi/Desktop/dataset_walls/train2.txt')
    parser.add_argument('--train_o', help='File name of train data (original)', default='/Users/shigetomi/Desktop/dataset_walls/train1.txt')
    parser.add_argument('--test_c', help='File name of test data(cropped)',     default='/Users/shigetomi/Desktop/dataset_walls/test2.txt')
    parser.add_argument('--test_o', help='File name of test data (original)',   default='/Users/shigetomi/Desktop/dataset_walls/test1.txt')
    parser.add_argument('-extractor', help='extractor architecture name', default='vgg_16')

    parser.add_argument('--max_steps', '-s', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=20)
    parser.add_argument('--num_classes', '-nc', type=int, default=6)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.8)

    parser.add_argument('--model_path', '-model', default='/Users/shigetomi/Downloads/vgg_16.ckpt', help='FullPath of inception-v4 model(ckpt)')
    parser.add_argument('--save_path', '-save', default='/Users/shigetomi/workspace/tensorflow_works/model/transl.ckpt', help='FullPath of saving model')
    parser.add_argument('--summary_dir', '-summary', default='/Users/shigetomi/workspace/tensorflow_works/log/', help='TensorBoard log')

    parser.add_argument('--cvflag', '-cv', default=None) # usually, dont use this

    args = parser.parse_args()

    train(args)

    print("all process finished　without a hitch. save the trained model at :", args.save_path)
