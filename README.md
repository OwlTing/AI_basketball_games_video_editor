# AI Basketball Games Video Editor

![](https://img.shields.io/static/v1?label=python&message=3.6&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.3&color=<COLOR>)
![](https://img.shields.io/static/v1?label=tensorrt&message=7.0.0&color=%3CCOLOR%3E)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

AI Basketball Games Video Editor is a command-line program to get basketball highlight video by PyTorch YOLOv4 object  detection.
Analyze basketball and basketball hoop locations collected from  object detection.
It can get shot frame index and cut video frame to merge highlight video.

```
├── README.md
├── video_editor.py                   demo to get basketball highlight video
├── pytorch_YOLOv4                    pytorch-YOLOv4 source code
│   ├── weights                       need to download weights
│   └── ...
├── tool
│   ├── utils_basketball.py           detect basketball shots algorithm
│   └── utils.py                  
├── dataset
│   └── your_video_name.mp4
├── result
│   ├── obj_log_name.data             save frame information and object detect result
│   └── your_output_video_name.mp4   
```
<p float="left">
  <img src="https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/introduction.gif" width="267" height="225"/>
  <img src="https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/introduction.jpg" width="267" height="225"/>
  <img src="https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/gif_highlight.gif" width="267" height="225"/>
</p>

# 0. Environments

## 0.1 Get a copy
```sh
git clone https://github.com/OwlTing/AI_basketball_games_video_editor.git
```

## 0.2 Create virtual environments
```sh
conda create --name py36_env python=3.6
conda activate py36_env
cd AI_basketball_games_video_editor
```

## 0.3 Requirements
Debian 10  
python 3.6  
numpy  
pandas  
tqdm  
cv2  
pytorch 1.3.0  
Please refer to the official documentation for installing pytorch https://pytorch.org/get-started/locally/  
More details for different cuda version https://pytorch.org/get-started/previous-versions/  
Example:  
conda install pytorch==1.3.0 torchvision==0.4.1 cudatoolkit=10.0 -c pytorch  

Optional (For tensorrt yolov4 object detector engine):  
tensorrt 7.0.0  
Please refer to the official documentation for installing tensorrt with different cuda version  
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html  
Example: (For Debian 10 cuda 10.0)  
1. mkdir tensorrt  
2. From https://developer.nvidia.com/tensorrt, to download 
   `TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz`  
   (select TensorRT 7.0) in the directory `tensorrt/`  
3. tar xzvf `TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz`  
4. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<path_your_tensorrt>/TensorRT-7.0.0.11/lib  
5. cd TensorRT-7.0.0.11/python/  
6. pip install `tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl`   
7. 
```
sudo cp /<path_your_tensorrt>/TensorRT-7.0.0.11/lib/libnvinfer.so.7 /usr/lib/ ;  
sudo cp /<path_your_tensorrt>/TensorRT-7.0.0.11/lib/libnvonnxparser.so.7 /usr/lib/ ;  
sudo cp /<path_your_tensorrt>/TensorRT-7.0.0.11/lib/libnvparsers.so.7 /usr/lib/ ;  
sudo cp /<path_your_tensorrt>/TensorRT-7.0.0.11/lib/libnvinfer_plugin.so.7 /usr/lib/ ;  
sudo cp /<path_your_tensorrt>/TensorRT-7.0.0.11/lib/libmyelin.so.1 /usr/lib/  
```    
8. pip install pycuda  

# 1. Weights Download

## 1.1 darknet2pytorch
- google(https://drive.google.com/file/d/15waE6I1odd_cR3hKKpm1uXXE41s5q1ax)
- `mkdir pytorch_YOLOv4/weights/`
- download file `yolov4-basketball.weights` in the directory `pytorch_YOLOv4/weights/`

## 1.2 tensorrt
- google(https://drive.google.com/file/d/1_c8uhyi47Krs5gAbRR66zzYKaxGNnzEs)
- `mkdir pytorch_YOLOv4/weights/`
- download file `yolov4-basketball.trt` in the directory `pytorch_YOLOv4/weights/`


# 2. Use AI Basketball Games Video Editor

## 2.1 Prepare your basketball video
- download your basketball video in the directory `dataset/`

## 2.2 Prepare output folder
- `mkdir result`

## 2.3 Run the demo
```sh
python video_editor.py --video_path VIDEO_PATH --output_path OUTPUT_PATH --output_video_name OUTPUT_VIDEO_NAME [OPTIONS]

# example
python video_editor.py --video_path dataset/basketball_demo.mp4 --output_path result/demo --output_video_name out_demo.mp4
```

- It will generate `your_output_video_name.mp4 obj_log_name.data` in the directory `result/`

- If you had finished extracting features. You can use `--read_flag 1` to read log for different output video mode. 

- If you use pytorch yolov4 object detector engine `--inference_detector pytorch`.  
  For image input size, you can select any inference_size = (height, width) in   
  height = 320 + 96 * n, n in {0, 1, 2, 3, ...}  
  width = 320 + 96 * m, m in {0, 1, 2, 3, ...}  
  Exmaple `--inference_size (1184, 1184)` or `--inference_size (704, 704)`  
  Default inference_size is (1184, 1184)
  
- If you use tensorrt yolov4 object detector engine `--inference_detector tensorrt`.  
  For image input size, you only can select `--inference_size (1184, 1184)`.  
  Tensorrt engine 3x faster than pytorch engine fps.

- You can use `--output_mode shot` to select different output video mode.
  ```
  output video mode  
  full            show person basketball basketball_hoop frame_information  
  basketball      show basketball basketball_hoop frame_information  
  shot            show basketball shot frame_information  
  standard        show frame_information  
  clean           only cutting video
  ```
![image](https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/output_mode_clean.jpg)
![image](https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/output_mode_full.jpg)
![image](https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/output_mode_basketball.jpg)
![image](https://github.com/OwlTing/AI_basketball_games_video_editor/blob/master/pic/output_mode_shot.jpg)

- You can refer the command-line options.
  ```
  optional arguments:
  -h, --help                                       show this help message and exit
  
  --video_path VIDEO_PATH                          input video path (default: None)
                                                   
  --output_path OUTPUT_PATH                        output folder path (default: None)
                                                   
  --output_video_name OUTPUT_VIDEO_NAME            output video name (default: None)
                                                   
  --highlight_flag HIGHLIGHT_FLAG                  select 1 with auto-generated highlight or 
                                                   0 without auto-generated highlight (default: 1)
                                                   
  --output_mode OUTPUT_MODE                        output video mode 
                                                   full       show person basketball basketball_hoop frame_information 
                                                   basketball show basketball basketball_hoop frame_information 
                                                   shot       show basketball shot frame_information 
                                                   standard   show frame_information 
                                                   clean      only cutting video (default: shot)
                                                   
  --process_frame_init PROCESS_FRAME_INIT          start processing frame (default: 0)
                                                   
  --process_frame_final PROCESS_FRAME_FINAL        end processing frame. If process_frame_final < 0, 
                                                   use video final frame (default: -1)
                                                   
  --obj_log_name OBJ_LOG_NAME                      save frame information and obj detect result 
                                                   (default: obj_log_name.data)
                                                   
  --save_step SAVE_STEP                            save obj log for each frame step (default: 2000)
                                                   
  --weight_path WEIGHT_PATH                        Yolov4 weight path (default: pytorch_YOLOv4/weights/yolov4-basketball.weights)
                                                   
  --cfg_path CFG_PATH                              Yolov4 cfg path (default: pytorch_YOLOv4/cfg/yolov4-basketball.cfg)
  
  --num_classes NUM_CLASSES                        num classes = 3 (person/basketball/basketball_hoop) (default: 3)
                                                   
  --namesfile_path NAMESFILE_PATH                  Yolov4 class names path (default: pytorch_YOLOv4/data/basketball_obj.names)
                                                   
  --inference_detector INFERENCE_DETECTOR          object detector engine. You can select pytorch or tensorrt (default: pytorch)
                                                   
  --inference_size INFERENCE_SIZE                  Image input size for inference 
                                                   If you use pytorch yolov4 object detector engine 
                                                   height = 320 + 96 * n, n in {0, 1, 2, 3, ...} 
                                                   width = 320 + 96 * m, m in {0, 1, 2, 3, ...} 
                                                   inference_size= (height, width) 
                                                   
                                                   If you use tensorrt yolov4 object detector engine Image input size for
                                                   inference only with inference_size = (1184, 1184) (default: (1184, 1184))
                                                   
  --read_flag READ_FLAG                            read log mode flag If you had finished extracting features. You can use 
                                                   select 1 to read log for different output video mode. (default: 0)
                                                                                                    
  --cut_frame CUT_FRAME                            cut frame range around shot frame index for highlight video (default: 50)  
  ```

Reference:
- https://github.com/Tianxiaomo/pytorch-YOLOv4
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code Yolo v4:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```

Contact:  
Issues should be raised directly in the repository.  
If you are very interested in this project, please feel free to contact me (george_chen@owlting.com).  
