import os
import sys
import traceback
import re
from datetime import datetime,timedelta
import argparse
from ast import literal_eval

import numpy as np
import pandas as pd
import tqdm

import cv2

from tool.utils import read_config, save_log, read_log
from tool.utils_basketball import catch_video_highlight_indexs


draw_text_color = (0, 0, 255)
thickness = 2
drawColor = (255, 255, 255)
boundColor = (0, 255, 255)

def color_map(index):
    color = [
        (205, 79, 57), #tomato3
        (46, 139, 87), #SeaGreen
        (106, 90, 205), #SlateBlue
        (218, 112, 214), #Orchid
        (139, 137, 137), #Snow4
        (238, 207, 161), #NavajoWhite2
        (0, 100, 0), #DarkGreen
        (238, 238, 0), #Yellow2
        (139, 0, 0), #DarkRed
        (139, 0, 139), #DarkMagenta
        (124, 252, 0), #LawnGreen
        (100, 149, 237), #CornflowerBlue
        (0, 0, 0), #Black
        (20, 0, 238), #Blue2
        (0, 139, 139) #DarkCyan
    ]

    return color[int(index)%15]

def draw_box(frame_cv2_array, obj_class_index, obj_class, x1, y1, x2, y2):
    
    select_color = color_map(obj_class_index)
    cv2.rectangle(frame_cv2_array, (x1, y1), (x2, y2), select_color, 5)
    cv2.putText(
        frame_cv2_array, f'{obj_class}', (x1, y1-30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, select_color, 
        thickness, cv2.LINE_AA)      
    
def draw_frame_information(video_capture, frame_cv2_array, frame_ch):
    
    cv2.putText(
        frame_cv2_array, 
        'ch:{} f_idx:{} f_time:{}'.format(
            str(frame_ch), str(int(video_capture.get(1))), 
            str(timedelta(seconds = int(video_capture.get(0))/1000))[:-5]), 
        (0, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, 
        draw_text_color, 
        thickness, 
        cv2.LINE_AA)


class VideoEditor(object):
    
    def __init__(
            self, 
            video_path, output_path, output_video_name, highlight_flag, 
            output_mode, process_frame_init, process_frame_final, 
            obj_log_name, save_step, weight_path, cfg_path, num_classes, 
            namesfile_path, inference_detector, inference_size, cut_frame):
        
        # based settings 
        self.video_path = video_path
        self.output_path = output_path
        self.output_video_name = output_video_name
        self.highlight_flag = highlight_flag
        self.output_mode = output_mode
        self.process_frame_init = process_frame_init
        self.process_frame_final = process_frame_final
        self.obj_log_name = obj_log_name
        self.save_step = save_step
        self.weight_path = weight_path
        self.cfg_path = cfg_path
        self.num_classes = num_classes
        self.namesfile_path = namesfile_path
        self.inference_detector = inference_detector
        self.inference_size = inference_size
        self.cut_frame = cut_frame
        
        # video information
        self.frame_w = None
        self.frame_h = None
        self.frame_final = None
        self.frame_fps = None
        
        # build detector
        self.detector = None
        self.obj_log = None
        self.columns_obj_box = (
            'obj_box_id',
            'frame_ch', 
            'frame_time', 
            'frame_index', 
            'frame_w', 
            'frame_h',
            'bound_l', 
            'bound_r', 
            'bound_t', 
            'bound_d', 
            'box_x1', 
            'box_y1', 
            'box_x2', 
            'box_y2', 
            'score_obj', 
            'obj_train_dataset', 
            'obj_class_index', 
            'obj_class'
        )
        
    def build_detector(self):
        
        if self.inference_detector == 'pytorch':
            from pytorch_YOLOv4.demo import Yolov4DarknetEngine        
            self.detector = Yolov4DarknetEngine(
                self.weight_path, self.cfg_path, self.namesfile_path, 
                self.inference_size, self.num_classes)              
            
        elif self.inference_detector == 'tensorrt':
            from pytorch_YOLOv4.demo_trt import Yolov4TrtEngine
            self.detector = Yolov4TrtEngine(
                self.weight_path, self.namesfile_path, 
                self.inference_size, self.num_classes)

    def box_to_in_bound(self, box, frame_w, frame_h):
        (x1, y1, x2, y2) = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        return (x1, y1, x2, y2)  

    def extract_features(self):
        
        video_capture = cv2.VideoCapture(self.video_path)
        video_capture.set(1,self.process_frame_init) 
        
        self.frame_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_final = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        frame_msec =  video_capture.get(cv2.CAP_PROP_POS_MSEC)
        
        if self.process_frame_final < 0:
            frame_init = self.process_frame_init
            frame_final = self.frame_final
        elif self.process_frame_final <= self.frame_final:
            frame_init = self.process_frame_init
            frame_final = self.process_frame_final
        else:
            frame_init = self.process_frame_init
            frame_final = self.frame_final
        
        frame_length = frame_final - frame_init
        
        print('-----------------------------------')
        print('      Extracting features start    ')
        print('-----------------------------------')
        print('setting frame initial: ', self.process_frame_init)
        print('setting frame final: ', self.process_frame_final)
        print('video frame final: ', self.frame_final)
        print('process frame length: ', frame_length)

        # function for extract feature by deep learning

        obj_log = []
        person_emb_log = []

        u_box_id = 1

        frame_ch = 1
        obj_train_dataset = 'basketball dataset'
        pbar = tqdm.tqdm(total=frame_length, mininterval=0.05)
        
        # set frame boundary

        shrink_l = 0
        shrink_r = 0
        shrink_t = 0
        shrink_d = 0

        bound_l = int(self.frame_w*shrink_l)
        bound_r = int(self.frame_w*shrink_r)
        bound_t = int(self.frame_h*shrink_t)
        bound_d = int(self.frame_h*shrink_d)         
        
        try:
            while True:
                ret, frame_cv2_array = video_capture.read()
                pbar.update(1)
                if ret != True:
                    print("Read video error")
                    print('Extracting features finish')
                    break;

                frame_index = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                frame_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)

                if frame_index >= frame_final:
                    print("Reach setting final frame")
                    print('Extracting features finish')
                    break;

                # ---- obj layer ----
                obj_box_list = self.detector.detect_image(frame_cv2_array)

                # ---- extract feature by box ----
                for box in obj_box_list:
                    (x1, y1, x2, y2, score_obj, obj_class_index) = box
                    (x1, y1, x2, y2) = self.box_to_in_bound((x1, y1, x2, y2), self.frame_w, self.frame_h)

                    obj_class = self.detector.class_names[obj_class_index]
                    obj_log.append((
                        u_box_id, 
                        frame_ch, 
                        frame_time, 
                        frame_index,
                        self.frame_w,
                        self.frame_h,
                        bound_l,
                        bound_r, 
                        bound_t, 
                        bound_d,
                        x1, y1, 
                        x2, y2, 
                        score_obj, 
                        obj_train_dataset, 
                        obj_class_index, 
                        obj_class))

                    u_box_id += 1

                if frame_index % self.save_step == 0:
                    save_log(obj_log, self.output_path, self.obj_log_name + '_step_' +str(frame_index))
            
            self.obj_log = obj_log
            
        except Exception as e:

                error_class = e.__class__.__name__ #取得錯誤類型
                detail = e.args[0] #取得詳細內容
                cl, exc, tb = sys.exc_info() #取得Call Stack
                lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
                line_num = lastCallStack[1] #取得發生的行號
                func_name = lastCallStack[2] #取得發生的函數名稱
                error_infor = "line {}, in {}: [{}] {}".format(line_num, func_name, error_class, detail)
                print(error_infor)

        finally:
            pbar.close()
            video_capture.release()
            
    def save_log(self):
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path) 
            
        save_log(self.obj_log, self.output_path, self.obj_log_name)
        
    def read_log(self):
        
        self.obj_log = read_log(self.output_path, self.obj_log_name)
        
    def draw_result(self):
        
        video_capture = cv2.VideoCapture(self.video_path)
        video_capture.set(1,self.process_frame_init) 

        self.frame_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_final = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_fps = video_capture.get(cv2.CAP_PROP_FPS)

        frame_msec =  video_capture.get(cv2.CAP_PROP_POS_MSEC)
        
        # write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(self.output_path, self.output_video_name)
        label_video_writer = cv2.VideoWriter(out_video_path, fourcc, 30, (self.frame_w, self.frame_h))        
        
        if self.process_frame_final < 0:
            frame_init = self.process_frame_init
            frame_final = self.frame_final
        elif self.process_frame_final <= self.frame_final:
            frame_init = self.process_frame_init
            frame_final = self.process_frame_final
        else:
            frame_init = self.process_frame_init
            frame_final = self.frame_final

        frame_length = frame_final - frame_init
        
        print('-----------------------------------')
        print('      Video processing start       ')
        print('-----------------------------------')
        print('setting frame initial: ', self.process_frame_init)
        print('setting frame final: ', self.process_frame_final)
        print('video frame final: ', self.frame_final)
        print('process frame length: ', frame_length)
        
        df_obj_log = pd.DataFrame(self.obj_log, columns=self.columns_obj_box)
        
        shot_indexs, shot_boxes, highlight_indexs = catch_video_highlight_indexs(df_obj_log, self.cut_frame)
        
        # draw box for checking result
        frame_ch = 1
        pbar = tqdm.tqdm(total=frame_length, mininterval=0.001)    

        try:
            while True:
                ret, frame_cv2_array = video_capture.read()
                
                pbar.update(1)
                if ret != True:
                    print("Read video error")
                    print("Video processing finish")
                    break;

                frame_index = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                frame_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)

                if frame_index >= frame_final:
                    print("Reach setting final frame")
                    print("Video processing finish")
                    break;

                # ---- highlight check ----
                if self.highlight_flag and (frame_index not in highlight_indexs):
                    continue                

                # ---- catch box by frame_index ----
                for row in df_obj_log[df_obj_log['frame_index'] == frame_index].iterrows():
                    row_item = row[1]
                    x1 = row_item['box_x1']
                    y1 = row_item['box_y1']
                    x2 = row_item['box_x2']
                    y2 = row_item['box_y2']
                    score_obj = round(row_item['score_obj'], 2)
                    obj_class_index = row_item['obj_class_index']
                    obj_class = row_item['obj_class']
                    obj_box_id = row_item['obj_box_id']

                    # ---- draw all box for analysis
                    if self.output_mode == 'full':
                        draw_box(frame_cv2_array, obj_class_index, obj_class, x1, y1, x2, y2)

                    elif self.output_mode == 'basketball' and obj_class_index != 0:
                        draw_box(frame_cv2_array, obj_class_index, obj_class, x1, y1, x2, y2)
                        
                    elif self.output_mode == 'shot' and (obj_box_id in shot_boxes):
                        scale = 0.5
                        hoop_w = abs(x2-x1)
                        hoop_h = abs(y2-y1)

                        x1 = int(x1 - hoop_w * scale/2)
                        y1 = int(y1 - hoop_h * scale/2)
                        x2 = int(x2 + hoop_w * scale/2)
                        y2 = int(y2 + hoop_h * scale/2)
                        
                        (x1, y1, x2, y2) = self.box_to_in_bound((x1, y1, x2, y2), self.frame_w, self.frame_h)
                        draw_box(frame_cv2_array, obj_class_index, 'shot', x1, y1, x2, y2)

                # ---- draw channel information ----
                if self.output_mode != 'clean':
                    draw_frame_information(video_capture, frame_cv2_array, frame_ch)       

                label_video_writer.write(frame_cv2_array)

        except Exception as e:

                error_class = e.__class__.__name__ #取得錯誤類型
                detail = e.args[0] #取得詳細內容
                cl, exc, tb = sys.exc_info() #取得Call Stack
                lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
                line_num = lastCallStack[1] #取得發生的行號
                func_name = lastCallStack[2] #取得發生的函數名稱
                error_infor = "line {}, in {}: [{}] {}".format(line_num, func_name, error_class, detail)
                print(detail)
                print(error_infor)

        finally:
            pbar.close()
            video_capture.release()
            label_video_writer.release()        
            
    def __str__(self):
        return ''.join(f'{k}: {v}\n' for k, v in self.__dict__.items())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='AI_basketball_video_editor', description='AI tool for basketball video editor', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--video_path', type=str, required=True, help='input video path')
    
    parser.add_argument('--output_path', type=str, required=True, help='output folder path')
    
    parser.add_argument('--output_video_name', type=str, required=True, help='output video name')
    
    parser.add_argument(
        '--highlight_flag', type=int, required=False, default=1, 
        help='select 1 with auto-generated highlight or 0 without auto-generated highlight')
    
    parser.add_argument(
        '--output_mode', type=str, required=False, default='shot', 
        help='''output video mode
full            show person basketball basketball_hoop frame_information
basketball      show basketball basketball_hoop frame_information
shot            show basketball shot frame_information
standard        show frame_information
clean           only cutting video''')
    
    parser.add_argument(
        '--process_frame_init', type=int, required=False, default=0, 
        help='start processing frame')
    
    parser.add_argument(
        '--process_frame_final', type=int, required=False, default=-1, 
        help='end processing frame. If process_frame_final < 0, use video final frame')
    
    parser.add_argument(
        '--obj_log_name', type=str, required=False, default='obj_log_name.data', 
        help='save frame information and obj detect result')
    
    parser.add_argument(
        '--save_step', type=int, required=False, default=2000, 
        help='save obj log for each frame step')
    
    parser.add_argument(
        '--weight_path', type=str, required=False, 
        default='pytorch_YOLOv4/weights/yolov4-basketball.weights', 
        help='Yolov4 weight path')
    
    parser.add_argument(
        '--cfg_path', type=str, required=False, 
        default='pytorch_YOLOv4/cfg/yolov4-basketball.cfg', 
        help='Yolov4 cfg path')
    
    parser.add_argument(
        '--num_classes', type=int, required=False, default=3, 
        help='num classes = 3 (person/basketball/basketball_hoop)')
    
    parser.add_argument(
        '--namesfile_path', type=str, required=False, 
        default='pytorch_YOLOv4/data/basketball_obj.names', 
        help='Yolov4 class names path')
    
    parser.add_argument(
        '--inference_detector', type=str, required=False, default='pytorch', 
        help='object detector engine. You can select pytorch or tensorrt')
    
    parser.add_argument(
        '--inference_size', type=str, required=False, default='(1184, 1184)', 
        help='''Image input size for inference
If you use pytorch yolov4 object detector engine
height = 320 + 96 * n, n in {0, 1, 2, 3, ...}
width = 320 + 96 * m, m in {0, 1, 2, 3, ...}
inference_size= (height, width)

If you use tensorrt yolov4 object detector engine
Image input size for inference only with inference_size = (1184, 1184)''')

    parser.add_argument(
        '--read_flag', type=int, required=False, default=0, 
        help='''read log mode flag
If you had finished extracting features. You can use select 1 to 
read log for different output video mode.''')
    
    parser.add_argument(
        '--cut_frame', type=int, required=False, default=50, 
        help='cut frame range around shot frame index for highlight video')    
    
    args = parser.parse_args()
        
    if args.inference_detector == 'pytorch':
        args.weight_path = 'pytorch_YOLOv4/weights/yolov4-basketball.weights'
    
    elif args.inference_detector == 'tensorrt':
        args.weight_path = 'pytorch_YOLOv4/weights/yolov4-basketball.trt'
    
    videoeditor = VideoEditor(
        video_path = args.video_path, 
        output_path = args.output_path, 
        output_video_name = args.output_video_name,
        highlight_flag = args.highlight_flag, 
        output_mode = args.output_mode, 
        process_frame_init = args.process_frame_init, 
        process_frame_final = args.process_frame_final, 
        obj_log_name = args.obj_log_name, 
        save_step = args.save_step, 
        weight_path = args.weight_path, 
        cfg_path = args.cfg_path, 
        num_classes = args.num_classes, 
        namesfile_path = args.namesfile_path, 
        inference_detector = args.inference_detector, 
        inference_size = literal_eval(args.inference_size), 
        cut_frame = args.cut_frame)
        
    
    if args.read_flag:
        videoeditor.read_log()
        videoeditor.draw_result()
    
    else:
        videoeditor.build_detector()
    #     print(videoeditor)
        videoeditor.extract_features()
        videoeditor.save_log()
        videoeditor.draw_result()
