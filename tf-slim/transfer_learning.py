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
from nets.resnet_v2 import resnet_v2_50, resnet_arg_scope
from shigeNet import *
import utils
import cv2
import time

def calc_loss(pred, supervisor_t):
  cross_entropy = -tf.reduce_sum(supervisor_t * tf.log(y_pred))
  # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=supervisor_t)) # with_logitsは内部でソフトマックスも計算してくれる
  return cross_entropy

def training(loss):
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

def train(args):
    extractor_name = args.extractor

    model_path = args.model_path
    image_size = archs[extractor_name]['fn'].default_image_size
    num_classes_s = args.num_classes_s
    num_classes_g = args.num_classes_g
    val_freq = 1# Nstep毎にvalidate
    store_freq = 10

    arch_name = "shigeNet_v1"
    if arch_name == "shigeNet_v1":
        model = shigeNet_v1
    elif arch_name == "shigeNet_v2":
        model = shigeNet_v2
    elif arch_name == "shigeNet_v3":
        model = shigeNet_v3
    elif arch_name == "shigeNet_v4":
        model = shigeNet_v4
    elif arch_name == "shigeNet_v5":
        model = shigeNet_v5
    elif arch_name == "shigeNet_v6":
        model = shigeNet_v6
    elif arch_name == "shigeNet_v7":
        model = shigeNet_v7
    elif arch_name == "shigeNet_v8":
        model = shigeNet_v8

    # Define placeholders
    with tf.name_scope('input'):
        with tf.name_scope('cropped_images'):
            x_c = tf.placeholder(dtype="float32", shape=(None, image_size,  image_size,  3))
        with tf.name_scope('original_images'):
            x_o = tf.placeholder(dtype="float32", shape=(None, image_size, image_size, 3))
        with tf.name_scope('labels'):
            t = tf.placeholder(dtype="float32", shape=(None, num_classes_s))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool")  # train flag

        # # メモreshapeはずしてみる
        # tf.summary.image('cropped_images', tf.reshape(x_c, [-1, image_size, image_size, 3]), max_outputs=args.batch_size)
        # tf.summary.image('original_images', tf.reshape(x_o, [-1, image_size, image_size, 3]), max_outputs=args.batch_size)

    # Build the graph
    y = model(cropped_images=x_c, original_images=x_o, extractor_name=extractor_name, num_classes_s=num_classes_s, num_classes_g=num_classes_g, is_training=is_training, keep_prob=keep_prob)
    # logits = tf.squeeze(end_points["Logits"], [1, 2])
    y_logit = y["Logits"]
    y_pred = y["Predictions"]

    # Get restored vars name in checkpoint
    #def name_in_checkpoint(var):
    #    if arch_name+"/orig" in var.op.name:
    #        return var.op.name.replace(arch_name+"/orig/", "")
    #    if arch_name+"/crop" in var.op.name:
    #        return var.op.name.replace(arch_name+"/crop/", "")

    def name_in_checkpoint(var):
        if arch_name in var.op.name:
            return var.op.name.replace(arch_name+"/", "")

    # Get vars restored
    variables_to_restore = slim.get_variables_to_restore()
    # print(variables_to_restore)
    # dict of {name in checkpoint: var.op.name}
    variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if extractor_name in var.op.name}
    #print(variables_to_restore)
    #print([key for key in variables_to_restore.keys() if key == None])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=None)# save all vars

    # Train ops
    with tf.name_scope('loss'):
        # loss = tf.losses.softmax_cross_entropy(t, logits)
        # loss = calc_loss(y_logit, t)
        loss = calc_loss(y_pred, t)
        tf.summary.scalar("loss", loss)

    with tf.name_scope('train') as scope:
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_step = slim.optimize_loss(loss, tf.train.get_or_create_global_step(), learning_rate=args.learning_rate, optimizer='Adam')
        train_step = training(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)

    # # GradCAM
    # y_c = tf.reduce_sum(tf.multiply(y_logit, t), axis=1)
    #
    # target_conv_layer_c = y["feature_c"]
    # target_conv_layer_grad_c = tf.gradients(y_c, target_conv_layer_c)[0]
    # gb_grad_c = tf.gradients(loss, x_c)[0]
    #
    # target_conv_layer_o = y["feature_o"]
    # target_conv_layer_grad_o = tf.gradients(y_c, target_conv_layer_o)[0]
    # gb_grad_o = tf.gradients(loss, x_o)[0]


    dataset = TwoInputDataset(train_c=args.train_c, train_o=args.train_o, test_c=args.test_c, test_o=args.test_o,
                              num_classes=num_classes_s, image_size=image_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    with tf.Session(config=config) as sess:
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
        sum_iter = 0

        # Train cycle
        for step in range(args.max_steps):
            dataset.shuffle()

            train_acc_l = []
            train_loss_l = []
            step_start_time = time.time()

            # Train proc
            for i in range(num_batch-1):  # i : batch index
                cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, i)
                start_time = time.time()
                train_acc, train_loss, _ = sess.run([accuracy, loss, train_step],
                                         feed_dict={x_c: cropped_batch['batch'],
                                                    x_o: orig_batch['batch'],
                                                    t: labels,
                                                    keep_prob: args.dropout_prob,
                                                    is_training: True})
                elapsed_time = time.time() - start_time

                train_acc_l.append(train_acc)
                train_loss_l.append(train_loss)
                print("step%03d" % step, i, "of", num_batch, "train acc:",  train_acc, ", train loss:", train_loss, "elappsed time:", elapsed_time)
                train_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="train/acc_by_1iter", simple_value=train_acc)
                ]), sum_iter)
                train_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="train/loss_by_1iter", simple_value=train_loss)
                ]), sum_iter)
                sum_iter += 1



            # Final batch proc: get summary and train_trace
            cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, num_batch-1)
            summary, train_acc, train_loss, _ = sess.run([merged, accuracy, loss, train_step],
                              feed_dict={x_c: cropped_batch['batch'],
                                         x_o: orig_batch['batch'],
                                         t: labels,
                                         keep_prob: args.dropout_prob,
                                         is_training: True},
                              options=run_options,
                              run_metadata=run_metadata)

            train_acc_l.append(train_acc)
            train_loss_l.append(train_loss)
            print("step%03d" % step, i, "of", num_batch, "train acc:", train_acc, ", train loss:", train_loss,
                  "elappsed time:", elapsed_time)
            mean_train_accuracy = sum(train_acc_l) / len(train_acc_l)
            mean_train_loss = sum(train_loss_l) / len(train_loss_l)
            print('step %d: training accuracy %g,\t loss %g' % (step, mean_train_accuracy, mean_train_loss), "elapsed time:", time.time() - step_start_time)

            # Write summary
            train_summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            # train_summary_writer.add_summary(summary, step)
            train_summary_writer.add_summary(summary, step)
            train_summary_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="train/mean_accuracy", simple_value=mean_train_accuracy)
            ]), step)
            train_summary_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="train/mean_loss", simple_value=mean_train_loss)
            ]), step)

            # Validation proc
            if step % val_freq == 0:

                # #Grad-CAM
                # # if i % 100 == 0:
                # prob_np = sess.run(y_pred, feed_dict={x_c: cropped_batch['batch'],
                #                                     x_o: orig_batch['batch'],
                #                                     t: labels,
                #                                     keep_prob: args.dropout_prob,
                #                                     is_training: True})
                # print('prob_np:', prob_np)
                # net_np, y_c_np, gb_grad_value_c, target_conv_layer_value_c, target_conv_layer_grad_value_c,  gb_grad_value_o, target_conv_layer_value_o, target_conv_layer_grad_value_o = sess.run(
                #     [y_logit, y_c, gb_grad_c, target_conv_layer_c, target_conv_layer_grad_c, gb_grad_o, target_conv_layer_o, target_conv_layer_grad_o],
                #     feed_dict={x_c: cropped_batch['batch'],
                #                x_o: orig_batch['batch'],
                #                t: labels,
                #                keep_prob: args.dropout_prob,
                #                is_training: True})
                #
                # for i in range(args.batch_size):
                #     # print('See visualization of below category')
                #     # utils.print_prob(batch_label[i], './synset.txt')
                #     utils.print_prob(prob_np[i], './synset.txt')
                #     # print('gb_grad_value[i]:', gb_grad_value[i])
                #     # print('gb_grad_value[i] shape:', gb_grad_value[i].shape)
                #     utils.visualize(cropped_batch['batch'][i], target_conv_layer_value_c[i], target_conv_layer_grad_value_c[i],
                #                     gb_grad_value_c[i])
                #     utils.visualize(orig_batch['batch'][i], target_conv_layer_value_o[i],
                #                     target_conv_layer_grad_value_o[i],
                #                     gb_grad_value_o[i])

                num_test_batch = int(len(dataset.test_path_c) / args.batch_size)
                test_acc_l = []
                test_loss_l = []
                for i in range(num_test_batch):
                    test_cropped_batch, test_orig_batch, test_labels = dataset.getTestData(args.batch_size, i)
                    summary_test, test_accuracy, test_loss = sess.run([merged, accuracy, loss],
                                                                      feed_dict={x_c: test_cropped_batch['batch'],
                                                                                 x_o: test_orig_batch['batch'],
                                                                                 t: test_labels,
                                                                                 keep_prob: 1.0,
                                                                                 is_training: False})

                    test_acc_l.append(test_accuracy)
                    test_loss_l.append(test_loss)
                    print("Valid step%03d" % step, i, "of", num_test_batch, "acc:", test_accuracy, ", loss:", test_loss)

                print("test acc_list", test_acc_l)
                mean_test_acc = sum(test_acc_l) / len(test_acc_l)
                mean_test_loss = sum(test_loss_l) / len(test_loss_l)


                # Write valid summary
                # test_summary_writer.add_summary(summary_test, step)
                test_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid/accuracy", simple_value=mean_test_acc)
                ]), step)
                test_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid/loss", simple_value=mean_test_loss)
                ]), step)

                print('step %d: test mean accuracy %g,\t mean loss %g' % (step, mean_test_acc, mean_test_loss))

            if step % store_freq == 0:
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
    parser.add_argument('--num_classes_s', '-ns', type=int, default=6)
    parser.add_argument('--num_classes_g', '-ng', type=int, default=9)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.5)

    parser.add_argument('--model_path', '-model', default='/Users/shigetomi/Downloads/vgg_16.ckpt', help='FullPath of inception-v4 model(ckpt)')
    parser.add_argument('--save_path', '-save', default='/Users/shigetomi/workspace/tensorflow_works/model/transl.ckpt', help='FullPath of saving model')
    parser.add_argument('--summary_dir', '-summary', default='/Users/shigetomi/workspace/tensorflow_works/log/', help='TensorBoard log')

    parser.add_argument('--cvflag', '-cv', default=None) # usually, dont use this

    args = parser.parse_args()

    train(args)

    print("all process finished　without a hitch. save the trained model at :", args.save_path)
