# tf-slim
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [tf-slim](#tf-slim)
	- [環境](#環境)
	- [transfer_learning.py](#transferlearningpy)

<!-- /TOC -->
## 環境
* tensorflow 1.4.<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [tf-slim](#tf-slim)
	- [環境](#環境)
	- [transfer_learning.py](#transferlearningpy)

<!-- /TOC -->
* python 3.5.x

## transfer_learning.py
* 転移学習を用いて特定物体認識(shigeNet-v1)

実行例
~~~
% python transfer_learning.py --train1 ~/Desktop/minitrain2.txt --test1 ~/Desktop/dataset_roadsign/test2.txt --train2 ~/Desktop/dataset_roadsign/train1.txt --test2 ~/Desktop/dataset_roadsign/test1.txt -b 5 -s 3 -pb '/Users/shigetomi/Downloads/imagenet/classify_image_graph_def.pb' -nc 6 -save ./model/twostep.ckpt
~~~
