# tf-slim
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [tf-slim](#tf-slim)
	- [環境](#環境)
	- [transfer_learning.py](#transferlearningpy)

<!-- /TOC -->
## 環境
* tensorflow 1.4.0
* python 3.5.x

## transfer_learning.py
* 転移学習を用いて特定物体認識(shigeNet-v1)

実行例
~~~
% python transfer_learning.py --train_c ~/dataset_roadsign/train2.txt --test_c ~/dataset_roadsign/test2.txt --train_o ~/dataset_roadsign/train1.txt --test_o ~/dataset_roadsign/test1.txt --model_path ~/inception_v4.ckpt -save /home/akalab/tensorflow_works/tf-slim/model/twostep_roadsign.ckpt --summary_dir /home/akalab/tensorflow_works/tf-slim/log -b 10 -s 1 -nc 6
~~~
