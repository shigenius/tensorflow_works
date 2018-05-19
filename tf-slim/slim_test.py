import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(x_image):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.max_pool2d], padding='SAME'):
            # Layer01
            h_conv1 = slim.conv2d(x_image, 32, [5, 5])
            h_pool1 = slim.max_pool2d(h_conv1, [2, 2])
            # Layer02
            h_conv2 = slim.conv2d(h_pool1, 64, [5, 5])
            h_pool2 = slim.max_pool2d(h_conv2, [2, 2])
            # Layer03
            h_pool2_flat = slim.flatten(h_pool2)
            h_fc1 = slim.fully_connected(h_pool2_flat, 1024)
            # Layer04
            y_conv = slim.fully_connected(h_fc1 , 10, activation_fn=None)
    return y_conv

if __name__ == '__main__':
    inference()