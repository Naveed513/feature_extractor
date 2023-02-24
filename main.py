# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:44:22 2023

@author: SPSOFT
"""

from feature_extractor_functions import FeatureExtractor 




if __name__ == '__main__':
    files_path = 'D:/poll/finall_poll/test1' # path where input pdf files are stored
    img_path = "D:/poll/finall_poll/temp_image" # path to temporarily store images will be deleted after programme complition
    result_path = 'D:/poll/finall_poll/extracted_result' # path where results need to be stored
    block_model_path = "D:/poll/finall_poll/block.pt" # block extractor yolo model path
    feature_model_path = "D:/poll/finall_poll/feature.pt" # feature extractor yolo model path
    extractor = FeatureExtractor(files_path, img_path, result_path, block_model_path, feature_model_path)
    extractor.model_init() # initializing yolov5 models
    extractor.data_extractor() # runing extractor