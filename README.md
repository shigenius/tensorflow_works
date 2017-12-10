# tensorflow_works

## Table of Contents

   * [tensorflow_works](#tensorflow_works)
      * [環境](#環境)
      * [classify.py](#classifypy)
      * [read_csv_images.py](#read_csv_imagespy)
      * [train_tf.py](#train_tfpy)
      * [train.py](#trainpy)
      * [Dataset.py](#datasetpy)
      * [evaluate.py](#evaluatepy)
      * [SpecificObjectRecognition.py](#specificobjectrecognitionpy)
      * [TwoInputDataset.py](#twoinputdatasetpy)

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
* `train_tf.py`で，一度にデータセット全体を読み込んでout-of-memoryになったりしてたのを直した．(tensorflowはメモリを2GBまでしか占有できないらしい)
  * batch処理の際に必要な分だけ画像を読み込む
* NNアーキテクチャを柔軟に変更できる．
* train accuracyがtrainデータセット全体のaccuracyじゃなくて，一部分での評価になってるところに注意．

基本的な使い方
~~~
% train.py <TrainDataTextPath> <TestDataTextPath> -a SimpleCNN -save <FullPATH> -b <batch size> -s <num step>
~~~

## Dataset.py

* `train.py`から移植．
* 引数の数やらでtrainデータセットとtestデータセットの両方，もしくはtestデータセットのみを受け取る
* イニシャライザの時点では，各画像のパスとラベルデータのみを保持．
* `getTrainBatch()`で指定したbatch分ndarrayに変換して返す．`getTestData()`はすべてのtestデータをndarrayにして返す．
* `shuffle()`はtrainデータの順番をシャッフルする．対応するラベルとのindexの関係は破壊されない．batch毎に呼ぶといいかも

## evaluate.py 

* 保存したmodelの評価を行う
* `saver.restore()`のサンプル
~~~
% python evaluate.py <TestDataTXTPath> <ModelPath(i.e. ./model.ckpt)>
~~~

## SpecificObjectRecognition.py

* PrimaryCNN : 画面全体を学習するモデル
* SecondaryCNN : PrimaryCNNのfeature mapと，特定物体の局所的な画像情報によるfeature mapをconcatしたものを学習するモデル

* まずPrimaryCNNを普通に学習させて，得たmodelのweights(fixed)を，SecondaryCNNで転移学習する感じ．
* 特定物体認識 <-> 一般物体認識

## TwoInputDataset.py

* `Dataset.py`の派生クラス
* 2入力に対応
* 今は本データセットとサブデータセットのindex関係が正常ではない
    * サブデータセットのtrackingの際にフレームが抜ける可能性があるため
    * フレーム数の情報がファイル名に保持されるように，データセットのファイル名を記述しているtxtファイルを新たに作成するスクリプトが必要
    * とりあえずサブデータセットがフレーム抜けしていないデータを用いる．
        * あとで`shuffle()`, `getTrainBatch()`を修正する必要がある
* サブデータセットの作成 : [opencv_tracking.py](https://github.com/shigenius/python_sources)を用いる．
