#!/usr/bin/env python
# Created on 2023/06
# Author: HUA
# 处理新的数据This file is used to convert the raw mat data to wav file

from convert2wav import write_wav_files
from convert2wav import write_spec_files
from convert2wav import write_csv_files
import os


sourcePath='F:\\mat_data\\data_202401'
outputPath='D:\\csvProcessNew_100000'

segment_length = 100000
#segment_length = 64000
wav_length = 4  # WAV文件长度（单位：秒）
sample_rate = 16000  # 采样率（例如：44100 Hz）
fft_size = 128

def get_subdirectories(folder):
    subdirectories = []
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
            subdirectories.extend(get_subdirectories(item_path))
    return subdirectories

    
    

if __name__=='__main__':
    # 获取文件夹下的所有文件夹
    sourceSubdirectories = get_subdirectories(sourcePath)

    
    for directory in sourceSubdirectories:
    # 获取相对路径，创建对应文件夹
        relative_path = os.path.relpath(directory,sourcePath )
        new_output_path = os.path.join(outputPath, relative_path)
        os.makedirs(new_output_path, exist_ok=True)
        #write_wav_files(directory,new_output_path,segment_length,wav_length,sample_rate)
        write_csv_files(directory,new_output_path,segment_length,fft_size)
