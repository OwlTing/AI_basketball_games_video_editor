from __future__ import unicode_literals
import argparse
from ast import literal_eval

import youtube_dl

from video_editor import VideoEditor


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='AI_basketball_video_editor', description='AI tool for basketball video editor', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dlurl', type=str, required=True, help='Youtube video url')
    
    parser.add_argument('--dlpath', type=str, required=True, help='Youtube video save path')    
    
#     parser.add_argument('--video_path', type=str, required=True, help='input video path')
    
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
        
    dl_url = args.dlurl
    dl_path = args.dlpath    
    
    ydl_opts = {
        'format':'bestvideo+bestaudio[ext=m4a]',
        'outtmpl':dl_path
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([dl_url])
    
    videoeditor = VideoEditor(
        video_path = args.dlpath, 
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
