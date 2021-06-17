# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str, default="651.jpg",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with new_size, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/voc_names.txt",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/model-epoch_990_step_89189_loss_0.3299_lr_1e-05",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
print(args.input_image)
img_ori = cv2.imread(args.input_image)
if args.letterbox_resize:
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
else:
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(args.new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
    print(boxes_)

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    #返回中间物体类别信息
    result = []
    for i in range(len(boxes_), ):
        if boxes_[i][0] < 960 and boxes_[i][2] < 960:
            result1 = 960 - boxes_[i][2]
            result.append(result1)
            print(result1)
        elif boxes_[i][0] < 960 and boxes_[i][2] > 960:
            result2 = 0
            result.append(result2)
        elif boxes_[i][0] > 960 and boxes_[i][2] > 960:
            result3 = boxes_[i][0] - 960
            result.append(result3)
            print(result3)
    position = result.index(min(result))
    object = {0: 'Object1', 1: 'Object2', 2: 'Object3', 3: 'Object4', 4: 'Object5', 5: 'Object6', 6: 'Object7',
              7: 'Object8', 8: 'Object9'}
    classes = object[labels_[position]]
    objecr_x1,objecr_y1,objecr_x2,objecr_y2 = boxes_[position][0],boxes_[position][1],boxes_[position][2],boxes_[position][3]
    object_coordinate = [(objecr_x1,objecr_y1),(objecr_x2,objecr_y2)]
    object_center = [int((objecr_x2+objecr_x1)/2),int((objecr_y2+objecr_y1)/2)]
    print('中间物体的中心坐标：',object_center)
    img_ori[int(object_center[1]),int(object_center[0])] = (255,0,0)
    if object_center[0] > 960:
        x_offset = object_center[0] - 960
        y_offset = object_center[1] - 540
        print('横向偏差：', x_offset,'纵向偏差',y_offset,'镜头偏右 请左移')
        img_ori = cv2.putText(img_ori, '横向偏差：'+str(x_offset), (50, 300), font, 1.2, (255, 255, 255), 3)
        img_ori = cv2.putText(img_ori, '纵向偏差：' + str(y_offset), (50, 350), font, 1.2, (255, 255, 255), 3)
    elif object_center[0] < 960:
        x_offset = 960 - object_center[0]
        y_offset = 540 - object_center[1]
        print('横向偏差：', x_offset, '纵向偏差', y_offset, '镜头偏左 请右移')
        img_ori = cv2.putText(img_ori, '横向偏差：' + str(x_offset), (50, 300), font, 1.2, (255, 255, 255), 3)
        img_ori = cv2.putText(img_ori, '纵向偏差：' + str(y_offset), (50, 350), font, 1.2, (255, 255, 255), 3)
    else:
        print('镜头已经正向')
    img_ori = cv2.putText(img_ori, '000', (50, 300), font, 1.2, (255, 255, 255), 3)
    print(position,classes)
    print(boxes_[position])

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(-1)
