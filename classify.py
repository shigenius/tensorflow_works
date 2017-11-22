import pandas as pd
import tensorflow as tf
from sklearn import cross_validation

# get data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', sep = "\t")
data = data.ix[:,1:]
train_data, test_data, train_target, test_target = cross_validation.train_test_split(data.ix[:,['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']], data.ix[:,['Species']], test_size = 0.4, random_state = 0)


# 設計図作成
# プレースホルダー設定
X = tf.placeholder(tf.float32, shape = [None, 4])
Y = tf.placeholder(tf.float32, shape = [None, 3])

# パラメーター設定
W = tf.Variable(tf.random_normal([4, 3], stddev=0.35))

# 活性化関数
y_ = tf.nn.softmax(tf.matmul(X, W))

# ビルド
# 損失関数
cross_entropy = -tf.reduce_sum(Y * tf.log(y_))

# 学習
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(cross_entropy)

# 実行
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        x = train_data
        y = pd.get_dummies(train_target)
        print(sess.run(W))

        sess.run(train, feed_dict = {X: x, Y: y})

    test = sess.run(W)