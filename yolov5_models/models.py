# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:31:45 2023

@author: Naveed
"""
import torch
import pickle
import gzip
# yolo models to pickle

# paths of yolo block and feature models
block_model_path = "D:/poll/finall_poll/block.pt" # block extractor yolo model path
feature_model_path = "D:/poll/finall_poll/feature.pt" # feature extractor yolo model path

# # trained yolov5 model to extract blocks from the image
# print('loading block extractor........')
# yolo_block = {'path':block_model_path, 'force_reload':True}
# model_block = torch.hub.load('ultralytics/yolov5', 'custom', **yolo_block) 


# # Save model_feature to a file
# with open('model_block.pkl', 'wb') as f:
#     pickle.dump(model_block, f)
    

# # trained yolov5 model to extract features from the cell
# print('loading feature extractor')
# yolo_feature = {'path':feature_model_path, 'force_reload':True}
# model_feature = torch.hub.load('ultralytics/yolov5', 'custom', **yolo_feature) 

# # Save model_feature to a file
# with open('model_feature.pkl', 'wb') as f:
#     pickle.dump(model_feature, f)
    
# # Load model_feature from a file
# with open('model_feature.pkl', 'rb') as f:
#     model_feature = pickle.load(f)


# yolo models to zip

model_block = torch.hub.load('ultralytics/yolov5', 'custom', path = block_model_path, force_reload = True)
model_feature = torch.hub.load('ultralytics/yolov5', 'custom', path = feature_model_path, force_reload = True)

# converting to pickle
model_block_pkl = pickle.dumps(model_block)
model_feature_pkl = pickle.dumps(model_feature)

# compressing pickle using gzip library
block_zip = gzip.compress(model_block_pkl)
feature_zip = gzip.compress(model_feature_pkl)

# saving compressed files
with open('model_block.zip', 'wb') as file:
    file.write(block_zip)
    
with open('feature.zip', 'wb') as file:
    file.write(feature_zip)
    
# opening zip/compressed files
with open('model_block.zip', 'rb') as file:
    block_zip = file.read()

# decompressing zip file to pickle file
block_pickle = gzip.decompress(block_zip)

# loading data from pickle file
model_block = pickle.loads(block_zip)


    
    
