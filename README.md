# tensorflow_works

## 環境
* tensorflow 1.4.0
* python 3.5.x

## classify.py

* mnistデータセットの分類問題を解く．公式のTutorial

## read_csv_images.py

* CSVファイルから画像を読み込む
* データセットを読み込むフレームワークはtensorflowで色々用意してあるらしい

## train_tf.py

* 自作データセットの読み込み，学習，評価，モデルの保存
* cross_entropy の部分でlog(0) = NaN になる可能性がある
* 

## train.py

* 自作データセットの読み込み，学習，評価，モデルの保存．
* train_tf.pyで，一度にデータセット全体を読み込んでout-of-memoryになったりしてたのを直した．(tensorflowはメモリを2GBまでしか占有できないらしい)
  * batch処理の際に必要な分だけ画像を読み込む
* NNアーキテクチャを柔軟に変更できる．
* train accuracyがtrainデータセット全体のaccuracyじゃなくて，一部分での評価になってるところに注意．

基本的な使い方
~~~
% train.py <TrainDataTextPath> <TestDataTextPath> -a SimpleCNN -save <FullPATH> -b <batch size> -s <num step>
~~~

## Dataset.py

* train,pyから移植．
* 引数の数やらでtrainデータセットとtestデータセットの両方，もしくはtestデータセットのみを受け取る
* イニシャライザの時点では，各画像のパスとラベルデータのみを保持．
* getTrainBatch()で指定したbatch分ndarrayに変換して返す．getTestData()はすべてのtestデータをndarrayにして返す．
* shuffle()はtrainデータの順番をシャッフルする．対応するラベルとのindexの関係は破壊されない．batch毎に呼ぶといいかも

## evaluate.py 

* 保存したmodelの評価を行う
* saver.restore()のサンプル
~~~
% python evaluate.py <TestDataTXTPath> <ModelPath(i.e. ./model.ckpt)>
~~~

## PeripheralCNN.py

* PeripheralCNN : 画面全体を学習するモデル
* FovealCNN : PeripheralCNNのfeature mapと，特定物体の局所的な画像情報によるfeature mapをconcatしたものを学習するモデル

* まずPeripheralCNNを普通に学習させて，得たmodelのweights(fixed)を，FovealCNNで転移学習する感じ．
* 特定物体認識 <-> 一般物体認識
* 未完成
