''' CSV file content
c:\work\image0\image1.png, 0
c:\work\image0\image2.png, 0
c:\work\image1\image3.png, 1
'''
# あとread_csv()で何ファイルか毎に取得するように修正する，あとshuffleもさせる
# augumentationも

import sys
import tensorflow as tf
import cv2

def read_csv(csvfile): #csvfileは読み込むCSVのファイルパス
    fname_queue = tf.train.string_input_producer([csvfile]) #
    reader = tf.TextLineReader() #tf.TextLineReader() : 1行単位でデータを読み取る．データを読む際はファイル姪のリストを保持するQueueを与える．
    key, val = reader.read(fname_queue) #Queueが処理される毎にKeyとvalueの値が更新されていく．key : ファイル名の何行目かという文字列, value : 行のデータそのもの
    fname, label = tf.decode_csv(val, [["aa"],[1]]) #valueの内容をパースする. >>1列目を文字列、2列目を数値として解釈するために、record_defaults引数[["aa"], [1]]を与えています。ここはデータ型そのものを指定するのではなく、代表値をいれるようです。
    return read_img(fname) #画像を返す．

def read_img(fname): #filenameを引数にとる
    img_r = tf.read_file(fname) #ファイルを読み込む．日本語が入っている場合は、UTF-8にｓうる．
    return tf.image.decode_image(img_r, channels=3) #decode. PNGとJPEGに対応．戻り値は(y, x, ch)のtensor

def main(): #使用例
    argv = sys.argv
    argc = len(argv)
    if (argc < 2):
        print('Usage: python %s csvfile' %argv[0])
        quit()
    
    image = read_csv(argv[1])
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess)
    x = sess.run(image)
    
    cv2.imshow("example", x)
    key = cv2.waitKey(0)

if __name__ == "__main__":
    # execute only if run as a script
    main()