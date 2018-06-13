# tf-slim

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [tf-slim](#tf-slim)
	- [TF-slimについて](#tf-slim)
	- [transfer_learning.py](#transferlearningpy)

<!-- /TOC -->

## TF-slimについて
ソース
* https://github.com/tensorflow/models
* https://github.com/tensorflow/models/tree/master/research/slim
	- pre-trained checkpointファイルやそれぞれのnetworkの定義(コード)なども載っています．

参考記事

* TensorFlow-Slim
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/README.md

* TF-Slim : TensorFlow 軽量ライブラリ Slim
http://tensorflow.classcat.com/2017/04/16/tensorflow-slim/
http://tensorflow.classcat.com/2017/04/27/tensorflow-slim-2/
	- ↑の日本語訳です

* TensorFlow(TF-Slim)で簡単に転移学習を試す
http://workpiles.com/2016/12/tensorflow-slim-transfer_learning/

* TF-Slimを使ってTensorFlowを簡潔に書く
http://workpiles.com/2016/12/tensorflow-tf-slim/

---
## transfer_learning.py
* 転移学習を用いて特定物体認識(shigeNet-v1)

実行例
~~~
% python transfer_learning.py --train_c ~/dataset_roadsign/train2.txt --test_c ~/dataset_roadsign/test2.txt --train_o ~/dataset_roadsign/train1.txt --test_o ~/dataset_roadsign/test1.txt --model_path ~/inception_v4.ckpt -save /home/akalab/tensorflow_works/tf-slim/model/twostep_roadsign.ckpt --summary_dir /home/akalab/tensorflow_works/tf-slim/log -b 10 -s 1 -nc 6
~~~

## comparison_learning.py
* single imageを入力としてvgg16を学習


実行例
~~~
% python comparison_learning.py --train_c ~/Desktop/minitrain2.txt --test_c ~/Desktop/dataset_roadsign/test1.txt -b 5 -s 3 -model /Users/shigetomi/Downloads/vgg_16.ckpt -nc 6 -save ./model/comparison.ckpt
~~~

## crossValidation.py
* 動画の撮影日毎に交差検証を行う．
* transfer_learning.pyかcomparison_learning.pyのどちらかを実行するコードとして指定する．
* 予め[make_crossValidDataset.py](https://github.com/shigenius/tensorflow_works#makecrossvaliddatasetpy)でデータセットを作成しておく必要がある．
	- ffmpeg依存

実行例
~~~
% python crossValidation.py ~/dataset_roadsign/ -extractor 'vgg_16' --model_path ~/vgg_16.ckpt -b 25 -s 500 --log_name 0612_roadsign --summary_dir /home/shige/tensorflow_works/tf-slim/log -execfile 'transfer_learning.py'
~~~

## data_augmentation.py
* 事前にデータセットを拡張するスクリプト．
* 指定したdatasetをaugmentして各ディレクトリに保存する．
* augmentationは，random contrast, gaussian noise, random brightness, random rotate など
	- だいたい標準正規乱数に従っている，はず


実行例
~~~
% python data_augmentation.py ~/Desktop/dataset_roadsign/ -r 4
~~~
* -r : データセットをn倍に拡張する
