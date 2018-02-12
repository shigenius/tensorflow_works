# tensorflow_works

## Table of Contents

   * [tensorflow_works](#tensorflow_works)
      * [Table of Contents](#table-of-contents)
      * [環境](#環境)
      * [実験手順の再現](#実験手順の再現)
         * [環境構築](#環境構築)
         * [データセットの用意](#データセットの用意)
            * [動画を画像にサンプリング](#動画を画像にサンプリング)
            * [cropping](#cropping)
            * [negative-sampleを作成する.](#negative-sampleを作成する)
            * [データセットを作成(train1,2とtest1,2のtxtファイルが作成される)](#データセットを作成train12とtest12のtxtファイルが作成される)
         * [学習](#学習)
         * [学習の経過を見る](#学習の経過を見る)
      * [classify.py](#classifypy)
      * [read_csv_images.py](#read_csv_imagespy)
      * [train_tf.py](#train_tfpy)
      * [train.py](#trainpy)
      * [Dataset.py](#datasetpy)
      * [evaluate.py](#evaluatepy)
      * [SpecificObjectRecognition.py](#specificobjectrecognitionpy)
      * [TwoInputDataset.py](#twoinputdatasetpy)
      * [makeDataset_forSpecificObjRecog.py](#makedataset_forspecificobjrecogpy)
      * [graphdef_test.py](#graphdef_testpy)


下に書いてあるコードが新しいやつ

## 環境
* tensorflow 1.4.0
* python 3.5.x

## 実験手順の再現
コードが肥大化してきたため，ある程度まとめようと思う．

### 環境構築

### データセットの用意
動画を用意する.(スマートフォンなどで撮影したものも可)

#### 動画を画像にサンプリング
[convert_to_image_each_labels.sh](https://github.com/shigenius/python_sources#convert_to_image_each_labelssh)を用いて動画を画像郡に変換する．
11行目の後に以下を挿入すると後が楽
~~~
if [ ! -d $label${dir}_cropped ]; 
  then mkdir ./$label/${dir}_cropped
fi;
~~~

このようなディレクトリ構造を作成する．

~~~
dataset/
 + convert_to_image_each_labels.sh
 + class1/
    + class1_video1.mp4
    + class1_video2.mov
 + class2/
    + class2_video1.mp4
    + class2_video2.mov
~~~

~~~
% cd <dataset>
% sh convert_to_image_each_labels.sh
~~~
* ffmpegが必要

#### cropping
[opencv_tracking.py](https://github.com/shigenius/python_sources) を用いてサブデータセットを作成する．
~~~
% python opencv_tracking.py <movie_path> <output_dir_path> -s <skipflame_value>
~~~
* 30fpsの動画のみ対応

30fpsの動画に変換する場合
~~~
動画の情報を取得
% ffmpeg -i <videopath>
フレームを間引く
% ffmpeg -i <input>-r 30 <output>
フレームを補完する場合(とても重たい)
% ffmpeg -i <input> -vf "minterpolate=30:2:0:1:8:16:32:0:1:5" -c:v mpeg4 -q:v 1 <output>
~~~

datasetのディレクトリ構造を以下のようにしておく．

~~~
dataset/
 + class1/
 + class2/
    + class2_video1.mp4
    + class2_video2.mov
    + class2_video1/
        + image_0001.jpg
    + class2_video1_cropped/
        + image_0001.jpg
~~~

#### negative-sampleを作成する.

~~~
% python makeRandomCropping-NegativeData.py <path of dataset directory>
~~~

#### データセットを作成(train1,2とtest1,2のtxtファイルが作成される)

~~~
% python makeDataset_forSpecificObjRecog.py <path of dataset directory>  -r <test set rate(default=0.1)>
~~~

### 学習
linuxの場合screenコマンドを用いれば，sshを切ってもプロセスはkillされない．

~~~
% python graphdef_test.py --train1 ~/dataset_walls/train2.txt --test1 ~/dataset_walls/test2.txt --train2 ~/dataset_walls/train1.txt --test2 ~/dataset_walls/test1.txt -pb /home/akalab/classify_image_graph_def.pb -save /home/akalab/tensorflow_works/model/twostep.ckpt -log /home/akalab/tensorflow_works/log -b 20 -s 1000
~~~
* 一例

### 学習の経過を見る
ローカル環境とリモート環境が同じネットワーク上にいる前提

ローカル環境下で
~~~
% ssh -L 8888:localhost:6006 <username@address>
~~~

リモート環境下で
~~~
% tensorboard --logdir tensorflow_works/log/twostep
~~~

ローカル環境のブラウザで`localhost:8888`を開く．

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

* PrimaryCNN : 普通のCNN
* SecondaryCNN : PrimaryCNNのfeature mapと，特定物体の局所的な画像情報によるfeature mapをconcatしたものを学習するモデル

* まずPrimaryCNNを学習させて，得たmodelのweights(fixed)を，SecondaryCNNで転移学習する．
* おそらく，特定物体の認識について大きく精度が向上すると思われる．
  * 特定物体認識 <-> 一般物体認識

## TwoInputDataset.py

* `Dataset.py`の派生クラス
* 2入力に対応
* train1とtrain2，またtest1とtest2は元ファイルの行番号によって対応する．
  * そのためそれぞれ行数が一致していなければならない

##  makeDataset_forSpecificObjRecog.py

* `SpecificObjectRecognition.py` のデータセットを作成する．
* testセットもランダムにサンプリングする．
* 前提として，[opencv_tracking.py](https://github.com/shigenius/python_sources) を用いてサブデータセットを作成しておく必要がある．
datasetのディレクトリ構造を以下のようにしておく．
~~~
dataset/
 + class1/
 + class2/
    + class2_video1.mp4
    + class2_video2.mov
    + class2_video1/
        + image_0001.jpg
    + class2_video1_cropped/
        + image_0001.jpg
~~~
* サブデータセットの各ディレクトリのsuffixとして`_cropped`をつける
* 上記の例を用いると，class2_video1/内の画像数 >= class2_video1_cropped/内の画像数 となるようにしなければならない．
  * trackingでフレームが抜ける可能性があるため
 
~~~
% python makeDataset_forSpecificObjRecog.py <dataset path> -r <test set rate(default=0.1)>
~~~

##  graphdef_test.py
* graphdefのテストで書いていたのにいつのまにか実験コードになってしまった
* SpecificObjectRecognition.pyで実装しようとした"特定"物体に対して有効な物体検出器アーキテクチャの実装．
* 1段目の画像認識器にはpretrained inception-v3を用いる
  * classify_image_graph_def.pbはtensorflow公式の[classify_image.py](https://github.com/tensorflow/models/tree/master/tutorials/image/imagenet)を実行する際にダウンロードされるやつを用いるとよい

使い方
~~~
% python graphdef_test.py --train1 ~/dataset_walls/train2.txt --test1 ~/dataset_walls/test2.txt --train2 ~/dataset_walls/train1.txt --test2 ~/dataset_walls/test1.txt -pb /home/akalab/classify_image_graph_def.pb -save /home/akalab/tensorflow_works/model/twostep.ckpt -log /home/akalab/tensorflow_works/log -b 20 -s 1000
~~~
* 一例
