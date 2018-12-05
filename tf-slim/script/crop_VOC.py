import os
import argparse
import xml.etree.ElementTree as ET
import cv2
import shutil
import csv

def get_anno_info(xml_path, target_class):
    # 入力のxmlと対応するimage pathとsizeとobjectのBoundaryBoxのサイズを返す．
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    image_size = root.find('size')

    d = {'xmlpath': xml_path,
         'filename': filename,
         'size': {'w': int(image_size.find('width').text),
                  'h': int(image_size.find('height').text),
                  'd': int(image_size.find('depth').text)}}

    d['bbs'] = [{"xmin": int(obj.find('bndbox').find('xmin').text),
                 "xmax": int(obj.find('bndbox').find('xmax').text),
                 "ymin": int(obj.find('bndbox').find('ymin').text),
                 "ymax": int(obj.find('bndbox').find('ymax').text)} for obj in root.findall('object') if obj.find('name').text == target_class]
    return d

def is_target_class(xml_path, target_class):
    # 入力のxmlの中にtarget classなobjectがあるかどうか返す
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        for obj_name in obj.findall('name'):
            if obj_name.text == target_class:
                return True

    return False

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('voc_dir_path', type=str)
    parser.add_argument('output_dir_path', type=str)
    # parser.add_argument('class_name', type=str)
    args = parser.parse_args()

    yyyy = [i.name for i in os.scandir(path=args.voc_dir_path) if i.is_dir()]
    d = {y: [os.path.join(os.path.join(args.voc_dir_path, y), i.name) for i in os.scandir(path=os.path.join(args.voc_dir_path, y)) if i.is_dir()] for y in yyyy}
    # print(d)
    annotations_dir = [i for v in d.values() for i in v if 'Annotations' in i]
    xmls = sorted([os.path.join(i, annotation.name) for i in annotations_dir for annotation in os.scandir(path=i) if os.path.splitext(annotation.name)[1] == '.xml'])

    # existence check
    for f in xmls:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

    print(xmls)

    class_list = ['aeroplane','bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    for class_name in class_list:
        target_infos = [get_anno_info(xml, args.class_name) for xml in xmls if is_target_class(xml, args.class_name)]
        print(target_infos)

        # prepare output dir
        org_save_dir = os.path.join(args.output_dir_path, args.class_name)
        crp_save_dir = os.path.join(args.output_dir_path, args.class_name+'_cropped')
        os.makedirs(org_save_dir, exist_ok=True)
        os.makedirs(crp_save_dir, exist_ok=True)

        # prepare logs
        log = open(os.path.join(crp_save_dir, "subwindow_log.txt"), 'w')
        writer = csv.writer(log, lineterminator='\n')
        writer.writerow(("id", "center_x", "center_y", "size_x", "size_y"))  # write header

        count = 1
        # crop boundary box
        for i in target_infos:
            image_path = os.path.join(os.path.join(os.path.split(os.path.split(i['xmlpath'])[0])[0], 'JPEGImages'), i['filename']) # もうちょっといい書き方を
            print(image_path)
            assert os.path.exists(image_path), "file:{0}".format(f.split(" ")[0])
            image = cv2.imread(image_path)
            # print(image.shape) # (h, w d)

            bbs = i['bbs']
            for bb in bbs:
                idx = '%04d' % count
                crop = image[bb['ymin']:bb['ymax'], bb['xmin']:bb['xmax'], :]
                shutil.copyfile(image_path, os.path.join(org_save_dir, 'image_'+idx+'.jpg'))
                cv2.imwrite(os.path.join(crp_save_dir, 'image_'+idx+'.jpg'), crop)

                w = bb['xmax'] - bb['xmin']
                h = bb['ymax'] - bb['ymin']
                writer.writerow((count, bb['xmin'] + w/2, bb['ymin'] + h/2, w, h))
    
                count += 1



                # cv2.imshow("hoge", crop)
                # cv2.waitKey(0)
