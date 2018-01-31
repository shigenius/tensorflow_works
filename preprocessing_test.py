import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.datasets import cifar10
import matplotlib.pyplot as plt

def draw_images(datagen, x, result_images):
    # 出力先ディレクトリを作成
    temp_dir = "temp"
    if os.path.exists(temp_dir) is False:
        os.mkdir(temp_dir)

    # generatorから9個の画像を生成
    # xは1サンプルのみなのでbatch_sizeは1で固定
    g = datagen.flow(x, batch_size=1, save_to_dir=temp_dir, save_prefix='img', save_format='jpg')
    for i in range(9):
        batch = g.next()

if __name__ == '__main__':
    IMAGE_FILE = '/Users/shigetomi/Desktop/samplepictures/0000065595.jpg'
    # 画像をロード（PIL形式画像）
    img = load_img(IMAGE_FILE)

    # numpy arrayに変換（row, col, channel)
    x = img_to_array(img)
    # print(x.shape)

    # 4次元テンソルに変換（sample, row, col, channel)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)

    # パラメータを一つだけ指定して残りはデフォルト
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, zca_whitening=True)

    # 生成した画像をファイルに保存
    draw_images(datagen, x, "result_rotation.jpg")