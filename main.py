# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:44:22 2023

@author: Naveed
"""
from feature_extractor_functions import FeatureExtractor
from multiprocess import Pool
import os
import multiprocessing
print('No.of cores available in system is:', multiprocessing.cpu_count()) # checking no.of cores available in system 

if __name__ == '__main__':
    files_path = "test" # path where input pdf files are stored
    img_path = "temp_img" # path to temporarily store images will be deleted after programme complition
    result_path = 'extracted_result' # path where results need to be stored
    block_model_path = "yolov5_models/block.pt" # block extractor yolo model path
    feature_model_path = "yolov5_models/feature.pt" # feature extractor yolo model path
    extractor = FeatureExtractor(files_path, img_path, result_path, block_model_path, feature_model_path)
    files_list = os.listdir(files_path) # list of pdf files
    filenames = [(sleep_value, file) for sleep_value, file in zip(range(0, len(files_list)*3, 3), files_list) 
                 if os.path.splitext(file)[-1].lower() == '.pdf']
    # starting multiprocessing 
    pool_process = Pool(12) # the available cores in my system is 12
    pool_process.map(extractor.data_extractor, filenames)
    pool_process.close()
    pool_process.join()
