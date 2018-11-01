# http://arkouji.cocolog-nifty.com/blog/2017/07/yolotensorflow-.html

import argparse
from pathlib import Path
import os
import re
import csv
from xml.dom.minidom import parseString
import cv2
import codecs
import shutil
import xml.etree.ElementTree as ET

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def get_image_and_annotation(image_path, txtname="subwindow_log.txt"):
    img_p = Path(image_path)
    img_obj_name = img_p.parents[1].name
    cropped_dir_p = Path(str(img_p.parent)+'_cropped')
    log_p = cropped_dir_p/txtname
    assert log_p.exists(), 'Does not exist dir :{0}'.format(str(log_p))

    img_id = int(img_p.stem.split('_')[1])


    anno = None
    with open(str(log_p), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if int(row[0]) == img_id:
                anno = row
                break

    return image_path, img_obj_name, anno

def convert_xml_format(imgpath, fname, objname, bb_xmin, bb_xmax, bb_ymin, bb_ymax):
    im = cv2.imread(imgpath)

    xml_template = "<annotation>\
                        <folder>VOC2007</folder>\
                    </annotation>"
    dom = parseString(xml_template)

    annotation = dom.getElementsByTagName("annotation")[0]

    filename = dom.createElement('filename')
    filename.appendChild(dom.createTextNode(fname))
    annotation.appendChild(filename)

    size = dom.createElement('size')
    annotation.appendChild(size)

    width = dom.createElement('width')
    width.appendChild(dom.createTextNode(str(im.shape[1])))
    size.appendChild(width)

    height = dom.createElement('height')
    height.appendChild(dom.createTextNode(str(im.shape[0])))
    size.appendChild(height)

    depth = dom.createElement('depth')
    depth.appendChild(dom.createTextNode(str(im.shape[2])))
    size.appendChild(depth)

    object = dom.createElement('object')
    annotation.appendChild(object)

    name = dom.createElement('name')
    name.appendChild(dom.createTextNode(objname))
    object.appendChild(name)

    bndbox = dom.createElement('bndbox')
    object.appendChild(bndbox)

    xmin = dom.createElement('xmin')
    xmin.appendChild(dom.createTextNode(str(bb_xmin)))
    bndbox.appendChild(xmin)

    xmax = dom.createElement('xmax')
    xmax.appendChild(dom.createTextNode(str(bb_xmax)))
    bndbox.appendChild(xmax)

    ymin = dom.createElement('ymin')
    ymin.appendChild(dom.createTextNode(str(bb_ymin)))
    bndbox.appendChild(ymin)

    ymax = dom.createElement('ymax')
    ymax.appendChild(dom.createTextNode(str(bb_ymax)))
    bndbox.appendChild(ymax)

    # domをxmlに変換して整形
    # print(dom.toprettyxml())
    return dom

def main(args):
    # make dataset of VOC format
    orig_p = Path(args.dataset_path)
    voc_p = Path(args.output_dir)
    print(orig_p.name)
    if not orig_p.is_dir():
        print("orig dir does not exist.")
        return

    voc_ch_p = voc_p/orig_p.name/'VOCdevkit'/'VOC2007'/'ImageSets'/'Main' # trainval.txtとtest.txtをこのdirに入れる
    if not voc_ch_p.exists():
        voc_ch_p.mkdir(parents=True)

    voc_im_dir_p = voc_ch_p.parents[1]/'JPEGImages'
    if not voc_im_dir_p.exists():
        voc_im_dir_p.mkdir(parents=True)
    voc_an_dir_p = voc_ch_p.parents[1]/'Annotations'
    if not voc_an_dir_p.exists():
        voc_an_dir_p.mkdir(parents=True)

    pattern1 = r"\.(jpg|png|jpeg)$"
    pattern2 = r"_cropped$"
    all_images = [f for f in find_all_files(str(orig_p)) if re.search(pattern1, f) and not re.search(pattern2, Path(f).parent.name)] # 親dirが_croppedなものは除外

    for f in all_images:
        assert Path(f).exists()

    data = [get_image_and_annotation(f) for f in all_images]
    data = [tup for tup in data if tup[2] is not None]
    print(data)

    train_text_p = orig_p/"train_orig.txt"
    val_text_p = orig_p/"valid_orig.txt"
    test_text_p = orig_p /"test_orig.txt"
    train_p = voc_ch_p /"train.txt"
    val_p = voc_ch_p / "val.txt"
    trainval_p = voc_ch_p/"trainval.txt"
    test_p = voc_ch_p /"test.txt"

    train_text_l = [f.split(' ')[0] for f in open(str(train_text_p), 'r')]
    val_text_l = [f.split(' ')[0] for f in open(str(val_text_p), 'r')]
    test_text_l = [f.split(' ')[0] for f in open(str(test_text_p), 'r')]
    train_f = open(str(train_p), 'w')
    val_f = open(str(val_p), 'w')
    trainval_f = open(str(trainval_p), 'w')
    test_f = open(str(test_p), 'w')

    for n, i in enumerate(data):
        centerx = float(i[2][1])
        centery = float(i[2][2])
        bbw = float(i[2][3])
        bbh = float(i[2][4])

        xmin = centerx - (bbw/2)
        xmax = xmin + bbw
        ymin = centery - (bbh/2)
        ymax = ymin + bbh
        image_name = '%06d.jpg' % n
        doc = convert_xml_format(i[0], image_name, i[1], xmin, xmax, ymin, ymax)

        save_anno_name = '%06d.xml' % n
        save_anno_path =  voc_an_dir_p/save_anno_name
        f = codecs.open(str(save_anno_path), 'wb', encoding='utf-8')
        doc.writexml(f, '', ' ' * 4, '\n', encoding='UTF-8')
        f.close

        save_image_path = voc_im_dir_p/image_name
        shutil.copy(i[0], str(save_image_path))
        print(n, ":", save_anno_path, save_image_path)

        index = '%06d' % n
        # trainval.txtとtest.txtの書き込み．
        print(i[0])
        if i[0] in train_text_l:
            print("in train list")
            train_f.write(index+"\n")
            trainval_f.write(index+"\n")
        if i[0] in val_text_l:
            print("in val list")
            val_f.write(index+"\n")
            trainval_f.write(index+"\n")
        if i[0] in test_text_l:
            print("in test list")
            test_f.write(index+"\n")

    print("finished,")
    train_f.close()
    trainval_f.close()
    test_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path', type=str, default='/Users/shigetomi/Desktop/dataset_fit_noNegative/dataset_shisa')
    parser.add_argument('output_dir', type=str, default='/Users/shigetomi/workspace/yolo_tensorflow/data')
    args = parser.parse_args()

    main(args)
