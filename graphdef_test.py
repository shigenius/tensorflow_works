# after running : models/tutorials/image/imagenet/classify_image.py

# Q. これはなに？
#  .pb形式で保存されたgraph構造とパラメータを呼び出してなんやかんやするサンプルコードです

import tensorflow as tf
import numpy as np

import argparse
from datetime import datetime

#MODEL_PATH = '/Users/shigetomi/Downloads/imagenet/classify_image_graph_def.pb'
MODEL_PATH = '/home/akalab/classify_image_graph_def.pb' # pre-trained inception v3 model
IMAGE_PATH = '/Users/shigetomi/Desktop/samplepictures/image_0011.jpg'


def create_graph():
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def sandbox():
    # 画像の読み込み
    if not tf.gfile.Exists(IMAGE_PATH):
        tf.logging.fatal('File does not exist %s', IMAGE_PATH)
    image_data = tf.gfile.FastGFile(IMAGE_PATH, 'rb').read()
    print(image_data)

    create_graph()

    with tf.Session() as sess:

        # graph内のtensor名は下記で確認できる．
        assing_ops = tf.Graph.get_operations(sess.graph)
        for op in assing_ops:
            print("operation name :", op.name, ", op_def.name :" ,op.op_def.name)
            for outputname in op.outputs:
                print("output :", outputname) # 出力テンソルの名前


        logits = sess.run('softmax:0', feed_dict={'DecodeJpeg/contents:0': image_data}) # 出力の取得はテンソル名で指定できる．
        logits = np.squeeze(logits)

        # print(logits.shape, logits)

        # 最もスコアが高いラベルとそのスコアを出力
        import sys
        sys.path.append('/Users/shigetomi/workspace/models/tutorials/image/imagenet/')
        from classify_image import NodeLookup

        NUM_TOP_PREDICTIONS = 5

        node_lookup = NodeLookup(label_lookup_path='/Users/shigetomi/Downloads/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt', uid_lookup_path='/Users/shigetomi/Downloads/imagenet/imagenet_synset_to_human_label_map.txt')
        top_k = logits.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = logits[node_id]
            print('%s (score = %.5f)' % (human_string, score))


        # extractor として(deep feartures?)
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        #print("pool3", sess.run(pool3), pool3.shape) # 重みの出力？

        pool3_features = sess.run(pool3, {'DecodeJpeg/contents:0': image_data})
        # pool3_features = np.squeeze(pool3_features)

        print("pool3_features", type(pool3_features), pool3_features, pool3_features.shape) #(2048,)

@profile
def train(args):
    from TwoInputDataset import TwoInputDataset

    image_size = 227
    num_classes = 4

    # arch = SimpleCNN()
    dataset = TwoInputDataset(train1=args.train1, train2=args.train2, test1=args.test1, test2=args.test2,  num_classes=num_classes, image_size=image_size)

    create_graph() # graphを作成する(restore)

    # with tf.Graph().as_default():
    with tf.Session() as sess:
        # images_placeholderA = tf.placeholder(dtype="float", shape=(None, image_size * image_size * 3))
        images_placeholderB = tf.placeholder(dtype="float32", shape=(None, image_size * image_size * 3))
        labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes))
        features_placeholder = tf.placeholder(dtype="float32", shape=(None, 1, 1, 2048)) # pool3 features. *shape & type is not nconfirmed yet
        keep_prob = tf.placeholder(dtype="float32")

        #--- graph内のtensor名の確認．
        # assing_ops = tf.Graph.get_operations(sess.graph)
        # for op in assing_ops:
        #     print("operation name :", op.name, op.op_def.name)
        #     for outputname in op.outputs:
        #         print("  output :", outputname)
        #---

        incep_pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        # print("incep_pool3 weight? :", sess.run(incep_pool3), incep_pool3.shape) # 重みの出力？

        # incep_features = sess.run(incep_pool3, {'DecodeJpeg/contents:0': image_data})
        # incep_features = np.squeeze(incep_features)
        # print(incep_features)

        # base archtecture
        with tf.variable_scope('SimpleCNN') as scope:
            def weight_variable(shape, name):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.get_variable(name, initializer=initial, trainable=True)

            def bias_variable(shape, name):
                initial = tf.constant(0.1, shape=shape)
                return tf.get_variable(name, initializer=initial, trainable=True)

            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            x_image = tf.reshape(images_placeholderB, [-1, image_size, image_size, 3])

            with tf.variable_scope('conv1') as scope:
                W_conv1 = weight_variable([11, 11, 3, 64], "w")
                b_conv1 = bias_variable([64], "b")
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)

            with tf.name_scope('pool1') as scope:
                h_pool1 = max_pool_2x2(h_conv1)

            with tf.variable_scope('conv2') as scope:
                W_conv2 = weight_variable([3, 3, 64, 128], "w")
                b_conv2 = bias_variable([128], "b")
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)

            with tf.name_scope('pool2') as scope:
                h_pool2 = max_pool_2x2(h_conv2)

            with tf.name_scope('concat') as scope:
                h_pool2_flat = tf.reshape(h_pool2, [-1,  6 * 6 * 128])
                incep_pool3_features_flat = tf.reshape(features_placeholder, [-1, 2048])
                h_concated = tf.concat([h_pool2_flat, incep_pool3_features_flat], 1)  # (?, x, y, z)

            with tf.variable_scope('fc1') as scope:
                W_fc1 = weight_variable([6 * 6 * 128 + 2048, 1024], "w")
                b_fc1 = bias_variable([1024], "b")
                h_fc1 = tf.nn.relu(tf.matmul(h_concated, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            with tf.variable_scope('fc2') as scope:
                W_fc2 = weight_variable([1024, num_classes], "w")
                b_fc2 = bias_variable([num_classes], "b")
                h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            with tf.name_scope('softmax') as scope:
                y = tf.nn.softmax(h_fc2)

        # logits = arch.inference(images_placeholder, keep_prob)
        loss = -tf.reduce_sum(labels_placeholder * tf.log(tf.clip_by_value(y, 1e-10, 1)))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.name_scope('train') as scope:
            train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
            acc_summary_train = tf.summary.scalar("train_accuracy", accuracy)
            loss_summary_train = tf.summary.scalar("train_loss", loss)

        with tf.name_scope('test') as scope:
            acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)

        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, args.restore)
            print("Model restored from : ", args.restore)

        # 変数の初期化
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(args.logdir + "/twostep/" + datetime.now().isoformat(), sess.graph)

        # 学習の処理
        for step in range(args.max_steps):
            dataset.shuffle()  # バッチで取る前にデータセットをshuffleする
            # batch処理
            for i in range(int(len(dataset.train2_path) / args.batch_size)):  # iがbatchのindexになる #バッチのあまりが出る

                batchA, batchB, labels = dataset.getTrainBatch(args.batch_size, i)

                # get features
                incep_features_batch = np.zeros((len(batchA), 1, 1, 2048))
                for j, image in enumerate(batchA):
                    # ndarray image -> tensor -> encoded image
                    image_tensor = tf.convert_to_tensor(np.uint8(np.reshape(image, [image_size, image_size, 3])[:, :, ::-1].copy()))
                    encoded = tf.image.encode_jpeg(image_tensor)
                    encoded_data = sess.run(encoded)
                    incep_features_batch[j] = sess.run(incep_pool3, {'DecodeJpeg/contents:0': encoded_data})

                # print(incep_features_batch.shape) # (batchsize, 1, 1, 2048)

                # バッチを学習
                sess.run(train_step, feed_dict={features_placeholder: incep_features_batch,
                                                images_placeholderB: batchB,
                                                labels_placeholder: labels,
                                                keep_prob: args.dropout_prob})

                # print("i:", i, "int(len(dataset.train2_path) / args.batch_size): ", int(len(dataset.train2_path) / args.batch_size))
                # 最終バッチの処理
                if i >= int(len(dataset.train2_path) / args.batch_size) - 1:

                    training_op_list = [accuracy, acc_summary_train, loss_summary_train]

                    # 最終バッチの学習のあと，そのバッチを使って評価．
                    result = sess.run(training_op_list,
                                      feed_dict={features_placeholder: incep_features_batch, images_placeholderB: batchB,
                                                 labels_placeholder: labels,
                                                 keep_prob: 1.0})

                    # 必要なサマリーを追記
                    for l in range(1, len(result)):
                        summary_writer.add_summary(result[l], step)

                    print("step %d  training final-batch accuracy: %g" % (step, result[0]))

                    # validation
                    test_dataA, test_dataB, test_labels = dataset.getTestData(args.batch_size)

                    # print("test_dataA length :", len(test_dataA))
                    # get features
                    incep_features_test_batch = np.zeros((len(test_dataA), 1, 1, 2048))

                    for k, image in enumerate(test_dataA):
                        # ndarray image -> tensor -> encoded image
                        image_tensor = tf.convert_to_tensor(
                            np.uint8(np.reshape(image, [image_size, image_size, 3])[:, :, ::-1].copy()))
                        encoded = tf.image.encode_jpeg(image_tensor)
                        encoded_data = sess.run(encoded)
                        incep_features_test_batch[k] = sess.run(incep_pool3, {'DecodeJpeg/contents:0': encoded_data})
                    
                    val_op_list = [accuracy, acc_summary_test]
                    val_result = sess.run(val_op_list,
                                          feed_dict={features_placeholder:incep_features_test_batch,
                                                     images_placeholderB: test_dataB,
                                                     labels_placeholder: test_labels,
                                                     keep_prob: 1.0})

                    summary_writer.add_summary(val_result[1], step)

                    print(" ramdom test batch accuracy %g" % val_result[0])

                    # print("incep_pool3 weight? :", sess.run(incep_pool3), incep_pool3.shape)

                    # 最終的なモデルを保存
                    save_path = saver.save(sess, args.save_path)
                    #print("save the trained model at :", save_path)

        print("all process finished.")

class tmpparse:
    def __init__(self):
        self.train1 = '/home/akalab/dataset_walls/train2.txt'
        self.train2 = '/home/akalab/dataset_walls/train1.txt'
        self.test1 = '/home/akalab/dataset_walls/test2.txt'
        self.test2 = '/home/akalab/dataset_walls/test1.txt'
        self.max_steps = 200
        self.batch_size = 10
        self.save_path = '/home/akalab/tensorflow_works/model/twostep.ckpt'
        self.logdir = '/home/akalab/tensorflow_works/log/'
        self.learning_rate = 1e-4
        self.dropout_prob = 0.5
        self.restore = '/home/akalab/tensorflow_works/model/twostep.ckpt'

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train1', help='File name of train data', default='~/dataset_walls/train2.txt ')
    parser.add_argument('--train2', help='File name of train data (subset)', default='~/dataset_walls/train1.txt')
    parser.add_argument('--test1', help='File name of train data', default='~/dataset_walls/test2.txt')
    parser.add_argument('--test2', help='File name of train data (subset)', default='~/dataset_walls/test1.txt')

    parser.add_argument('--max_steps', '-s', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=10)

    #parser.add_argument('--save_path', '-save', default='/home/akalab/tensorflow_works/model/twostep.ckpt', help='FullPath of output model')
    #parser.add_argument('--logdir', '-log', default='/home/akalab/tensorflow_works/log/', help='Directory to put the training data. (TensorBoard)')
    parser.add_argument('--save_path', '-save', default='/Users/shigetomi/workspace/tensorflow_works/model/twostep.ckpt', help='FullPath of output model')
    parser.add_argument('--logdir', '-log', default='/Users/shigetomi/workspace/tensorflow_works/tensorflow_works/log/', help='Directory to put the training data. (TensorBoard)')

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.5)

    # parser.add_argument('--resotre', '-r', default=None, help='FullPath of loading model')

    args = parser.parse_args()
    """
    args = tmpparse()
    # sandbox()
    train(args)
