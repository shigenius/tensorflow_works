import cv2
import numpy as np

import csv
import argparse

import os
import re
import sys

# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
# r キー
R_KEY = 0x72
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

class Motion:
    def __init__(self, args):
        cv2.namedWindow("motion")
        cv2.setMouseCallback("motion", self.onMouse) # マウスイベントのコールバック登録

        self.target_dir_path = args.target
        self.output_path = args.output
        pattern = r"\.(jpg|png|jpeg)$"
        self.target_files = sorted([os.path.join(self.target_dir_path, file.name) for file in os.scandir(path=self.target_dir_path) if file.is_file() and re.search(pattern, file.name)])
        self.frame = None # 現在のフレーム（カラー)
        self.gray = None # 現在のフレーム（グレー）
        self.gray_prev = None # 前回のフレーム（グレー）
        self.features = None # 特徴点
        self.status = None # 特徴点のステータス

        self.rectsize_w = args.width
        self.rectsize_h = args.height

        # 特徴点の最大数
        self.MAX_FEATURE_NUM = 500
        self.startidx = args.startidx
        self.skip_frame = args.skip
        self.interval = 1

        # 初期フレームをread
        self.frame = cv2.imread(self.target_files[0])
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("motion", self.frame)
        self.waitkeyControl(0)

    def run(self):
        log = open(os.path.join(self.output_path, "subwindow_log.txt"), 'w')
        writer = csv.writer(log, lineterminator='\n')
        writer.writerow(("id", "center_x", "center_y", "size_x", "size_y")) # write header

        # main loop
        for i, path in enumerate(self.target_files):
            self.gray_prev = np.copy(self.gray)
            self.frame = cv2.imread(path)
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            screen_w = self.frame.shape[1]
            screen_h = self.frame.shape[0]

            if self.features is not None:
                features_prev = np.copy(self.features)
                # オプティカルフローの計算
                self.features, self.status, err = cv2.calcOpticalFlowPyrLK(self.gray_prev, self.gray, features_prev, None, winSize=(10, 10), maxLevel=3, criteria=CRITERIA, flags=0)
                self.refreshFeatures()

                # skip frame
                if i % (self.skip_frame + 1) != 0:
                    continue

                if self.features is not None:
                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        rect_x = int(feature[0][0] - self.rectsize_w / 2) # 矩形の始点(ひだりうえ)のx座標
                        rect_y = int(feature[0][1] - self.rectsize_h / 2)

                        # 矩形がscreenからはみ出ている場合にzero-paddingを行う．
                        if (rect_x < 0) or (rect_y < 0) or ((rect_x + self.rectsize_w) > screen_w) or ((rect_y + self.rectsize_h) > screen_h):
                            bg = np.zeros((screen_h+(self.rectsize_h*2), screen_w+(self.rectsize_w*2), 3)) # 元画像の(w*3, h*3)のサイズの黒画像
                            bg[self.rectsize_h:self.rectsize_h+screen_h,
                               self.rectsize_w:self.rectsize_w+screen_w] = self.frame
                            crop = bg[self.rectsize_h+rect_y:self.rectsize_h*2+rect_y,
                                      self.rectsize_w+rect_x:self.rectsize_w*2+rect_x]
                        else:
                            crop = self.frame[rect_y:rect_y+self.rectsize_h, rect_x:rect_x+self.rectsize_w]

                        print(i + args.startidx, path)
                        cv2.imwrite(os.path.join(self.output_path, "image_" + '%04d' % (i+args.startidx) + ".jpg"), crop)
                        writer.writerow((i+args.startidx, feature[0][0], feature[0][1], self.rectsize_w, self.rectsize_h))
                        cv2.circle(self.frame, (feature[0][0], feature[0][1]), 4, (15, 241, 255), -1, 8, 0)
                        cv2.rectangle(self.frame, (rect_x, rect_y), (rect_x + self.rectsize_w, rect_y + self.rectsize_h), (255, 0, 0,), 3 , 4)
                else:
                    cv2.imshow("motion", self.frame)
                    self.waitkeyControl(0)
            else:
                cv2.imshow("motion", self.frame)
                self.waitkeyControl(0)
            # 表示
            cv2.imshow("motion", self.frame)
            self.waitkeyControl(self.interval)

        # 終了処理
        cv2.destroyAllWindows()
        log.close


    def waitkeyControl(self, interval):
        # インターバル
        key = cv2.waitKey(interval)
        # "Esc"キー押下で終了
        if key == ESC_KEY:
            sys.exit()
        # "s"キー押下で一時停止
        elif key == S_KEY:
            self.interval = 0
        # Run
        elif key == R_KEY:
            self.interval = self.interval


    # マウスクリックで特徴点を指定する
    #     クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
    #     クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
    def onMouse(self, event, x, y, flags, param):
        # 左クリック以外
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 最初の特徴点追加
        if self.features is None:
            self.addFeature(x, y)
            return

        radius = 200
        # 既存の特徴点が近傍にあるか探索
        index = self.getFeatureIndex(x, y, radius)

        # クリックされた近傍に既存の特徴点があるので既存の特徴点を削除する
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)

        return


    # 指定した半径内にある既存の特徴点のインデックスを１つ取得する
    #     指定した半径内に特徴点がない場合 index = -1 を返す
    def getFeatureIndex(self, x, y, radius):
        # 特徴点が１つも登録されていない
        if self.features is None:
            return -1

        max_r2 = radius ** 2
        for i, point in enumerate(self.features):
            dx = x - point[0][0]
            dy = y - point[0][1]
            r2 = dx ** 2 + dy ** 2
            if r2 <= max_r2:  # 指定した半径内にある場合
                return i

        # 全ての特徴点が指定した半径の外側にある場合
        return -1


    # 特徴点を新規に追加する
    def addFeature(self, x, y):

        # 特徴点が未登録
        if self.features is None:
            # ndarrayの作成し特徴点の座標を登録
            self.features = np.array([[[x, y]]], np.float32)
            self.status = np.array([1])
            # 特徴点を高精度化
            cv2.cornerSubPix(self.gray, self.features, (10, 10), (-1, -1), CRITERIA)

        # 特徴点の最大登録個数をオーバー
        elif len(self.features) >= self.MAX_FEATURE_NUM:
            print("max feature num over: ", self.MAX_FEATURE_NUM)

        # 特徴点を追加登録
        else:
            # 既存のndarrayの最後に特徴点の座標を追加
            self.features = np.append(self.features, [[[x, y]]], axis=0).astype(np.float32)
            self.status = np.append(self.status, 1)
            # 特徴点を高精度化
            cv2.cornerSubPix(self.gray, self.features, (10, 10), (-1, -1), CRITERIA)


    # 有効な特徴点のみ残す
    def refreshFeatures(self):
        # 特徴点が未登録
        if self.features is None:
            return

        # 全statusをチェックする
        i = 0
        while i < len(self.features):
            if self.status[i] == 0: # 特徴点として認識できない時
                # 既存のndarrayから削除
                self.features = np.delete(self.features, i, 0)
                self.status = np.delete(self.status, i, 0)
                i -= 1

            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature point tracking and cropping rectangle')
    parser.add_argument('target', help='Path to images directory')
    parser.add_argument('output', help='Path to output directory')
    parser.add_argument('-W', '--width',  type=int, default=256, help='rectangle width size (default = 256)')
    parser.add_argument('-H', '--height', type=int, default=256, help='rectangle height size (default = 256)') # '-h'rだとなぜかconflictする...?
    parser.add_argument('-s', '--skip', type=int, default=0, help='skip frame')
    parser.add_argument('-i', '--startidx', type=int, default=1, help='starting index')

    args = parser.parse_args()

    Motion(args).run()
