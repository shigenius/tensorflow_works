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

## transfer_learning.py
* 転移学習を用いて特定物体認識(shigeNet-v1)

実行例
~~~
% python transfer_learning.py --train_c ~/dataset_roadsign/train2.txt --test_c ~/dataset_roadsign/test2.txt --train_o ~/dataset_roadsign/train1.txt --test_o ~/dataset_roadsign/test1.txt --model_path ~/inception_v4.ckpt -save /home/akalab/tensorflow_works/tf-slim/model/twostep_roadsign.ckpt --summary_dir /home/akalab/tensorflow_works/tf-slim/log -b 10 -s 1 -nc 6
~~~
