#!/usr/bin/env python
# coding: utf-8

import configparser
import pickle

def read_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config

def save_log(data, output_dir, file_name):
    output_path = '{}/{}'.format(output_dir, file_name)
    
    mydata = data
    with open(output_path, 'wb') as f:
        pickle.dump(mydata, f)
        
def read_log(output_dir, file_name):
    read_path = '{}/{}'.format(output_dir, file_name)
    
    with open(read_path, 'rb') as f:
        mydata = pickle.load(f)
        
    return mydata
