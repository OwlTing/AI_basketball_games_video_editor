#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import numpy as np
import pandas as pd

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    return interArea

# get_shot_index
def detect_shot(
        df_obj_log_basketball, df_obj_log_basketball_hoop, 
        search_frame_range=2, search_mode='full', scale=0.5):
    
    shot_indexs = []
    shot_boxes = []
    for row in df_obj_log_basketball.iterrows():
        row_item = row[1]
        basketball_index = row_item['frame_index']
        x1 = row_item['box_x1']
        y1 = row_item['box_y1']
        x2 = row_item['box_x2']
        y2 = row_item['box_y2']
        basketball_box = (x1, y1, x2, y2)
        if search_mode == 'backward':
            search_mask = (
                (df_obj_log_basketball_hoop['frame_index'] <= basketball_index) 
                & (df_obj_log_basketball_hoop['frame_index'] >= (basketball_index - search_frame_range)))

        elif search_mode == 'forward':
            search_mask = (
                (df_obj_log_basketball_hoop['frame_index'] <= (basketball_index + search_frame_range)) 
                & (df_obj_log_basketball_hoop['frame_index'] >= basketball_index))

        elif search_mode == 'full':
            search_mask = (
                (df_obj_log_basketball_hoop['frame_index'] <= (basketball_index + search_frame_range)) 
                & (df_obj_log_basketball_hoop['frame_index'] >= (basketball_index - search_frame_range)))

        for row in df_obj_log_basketball_hoop[search_mask].iterrows():
            row_item = row[1]
            x1 = row_item['box_x1']
            y1 = row_item['box_y1']
            x2 = row_item['box_x2']
            y2 = row_item['box_y2']
            obj_box_id = row_item['obj_box_id']
            hoop_w = abs(x2-x1)
            hoop_h = abs(y2-y1)

            x1 = x1 - hoop_w * scale/2
            y1 = y1 - hoop_h * scale/2
            x2 = x2 + hoop_w * scale/2
            y2 = y2 + hoop_h * scale/2

            basketball_hoop_box = (x1, y1, x2, y2)

            if bb_intersection_over_union(basketball_box, basketball_hoop_box) > 0:
                shot_indexs.append(basketball_index)
                shot_boxes.append(obj_box_id)

    shot_indexs = set(shot_indexs)
    shot_boxes = set(shot_boxes)
    
    return shot_indexs, shot_boxes

# get highlight indexs
def get_highlight_indexs(shot_indexs, shot_index_range=50):
    highlight_indexs = set([])
    for shot_index in shot_indexs:
        current_indexs = set([i for i in range(shot_index - shot_index_range, shot_index + shot_index_range + 1)])
        highlight_indexs = highlight_indexs.union(current_indexs)
        
    return highlight_indexs

def catch_video_highlight_indexs(df_obj_log, shot_index_range):
    df_obj_log_basketball = df_obj_log[df_obj_log['obj_class_index'] == 1]
    df_obj_log_basketball_hoop = df_obj_log[df_obj_log['obj_class_index'] == 2]
    
    t0 = datetime.now()

    shot_indexs, shot_boxes = detect_shot(
        df_obj_log_basketball, df_obj_log_basketball_hoop, 
        search_frame_range=2, search_mode='full', scale=0.5)

    highlight_indexs = get_highlight_indexs(shot_indexs, shot_index_range)

    t1 = datetime.now()

    spend_time = t1 - t0
    print('spend time: ', spend_time)
    
    return shot_indexs, shot_boxes, highlight_indexs