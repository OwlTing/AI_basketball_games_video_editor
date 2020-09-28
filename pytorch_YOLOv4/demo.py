# Notice
# To modify the source code for AI_basketball_games_video_editor project:
# 1. Add class Yolov4DarknetEngine for object detection

# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
import cv2
from pytorch_YOLOv4.tool.utils import *
from pytorch_YOLOv4.tool.torch_utils import *
from pytorch_YOLOv4.tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args

class Yolov4DarknetEngine(object):
    
    def __init__(self, weight_path, cfg_path, namesfile_path, inference_size, num_classes):
        
        m = Darknet(cfg_path)
#         m.print_network()
        m.load_weights(weight_path)
        print('Loading weights from %s... Done!' % (weight_path))

        if use_cuda:
            m.cuda()

        self.num_classes = m.num_classes
        self.class_names = load_class_names(namesfile_path)
        self.engine = m
        self.image_size = inference_size

    def detect(self, model, img, image_size):
        model.eval()
        
        IN_IMAGE_H, IN_IMAGE_W = image_size
        
        sized = cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)        
        
        t0 = time.time()

        if type(sized) == np.ndarray and len(sized.shape) == 3:  # cv2 image
            sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(sized) == np.ndarray and len(sized.shape) == 4:
            sized = torch.from_numpy(sized.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)
        
        use_cuda = 1
        if use_cuda:
            sized = sized.cuda()
        sized = torch.autograd.Variable(sized)

        t1 = time.time()
        
        with torch.no_grad():
            output = model(sized)

        t2 = time.time()

#         print('-----------------------------------')
#         print('           Preprocess : %f' % (t1 - t0))
#         print('      Model Inference : %f' % (t2 - t1))
#         print('-----------------------------------')

        boxes = post_processing(img, 0.4, 0.6, output)

        return boxes
    
    def detect_image(self, image_src):

        # Inference input size is 416*416 does not mean training size is the same
        # Training size could be 608*608 or even other sizes
        # Optional inference sizes:
        #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
        boxes = self.detect(self.engine, image_src, self.image_size)

        width = image_src.shape[1]
        height = image_src.shape[0]

        output_boxes = []
        for box in boxes[0]:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            cls_conf = box[5]
            cls_id = box[6]
            output_boxes.append([
                max(x1, 0), 
                max(y1, 0), 
                max(x2, 0), 
                max(y2, 0), 
                cls_conf, 
                cls_id])

        return output_boxes

if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
