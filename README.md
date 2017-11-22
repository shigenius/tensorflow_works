# tensorflow_works

## 環境
* tensorflow 1.4.0
* python 3.5.x

## classify.py

* mnistデータセットの分類問題を解く公式のサンプルコード

## read_csv_images.py

* CSVファイルから画像を読み込む
* データセットを読み込むフレームワークはtensorflowで色々用意してあるらしい

## train_tf.py

* 自作データセットの読み込み，学習，モデルの保存
* cross_entropy の部分でlog(0) = NaN になる可能性がある
* 

## train.py

* train_tf.pyを元にNNアーキテクチャを代えたりしたやつ
