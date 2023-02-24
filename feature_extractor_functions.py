# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:36:22 2023

@author: Naveed
"""

# importing required libraries

import cv2
import xlsxwriter
import fitz
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\poll\finall_poll\Tesseract-OCR\tesseract.exe" # path of tesseract.exe


class FeatureExtractor:
    
    def __init__(self, files_path, img_path, result_path, block_model_path, feature_model_path):
        self.files_path = files_path # path were input pdf files are stored
        self.img_path = img_path # path where images are stored temporarily
        self.result_path = result_path # path where the result excel file is stored
        self.block_model_path = block_model_path # path for yolo block extractor model
        self.feature_model_path = feature_model_path # path for yolo feature extractor model
        
    def model_init(self):
        # trained yolov5 model to extract individual cell/block from the cells group image
        print('loading block extractor........')
        yolo_block = {'path':self.block_model_path, 'force_reload':True}
        self.model_block = torch.hub.load('ultralytics/yolov5', 'custom', **yolo_block) 

        # trained yolov5 model to extract features from the cell
        print('loading feature extractor')
        yolo_feature = {'path':self.feature_model_path, 'force_reload':True}
        self.model_feature = torch.hub.load('ultralytics/yolov5', 'custom', **yolo_feature) 
    
    #converting pdf to image using PyMuPDF
    def pdf_image(self, path):
        '''parameter path contains single_pdf_file location'''
        dir_name, file_name = os.path.split(path)
        img_name = os.path.splitext(file_name)[0]
        zoom_x = 2.0  # horizontal zoom
        zoom_y = 2.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)
        #patht="/code/./coreservice/pdf"
        doc = fitz.open(path)
        #os.chdir(patht)
    #     image_path = 'D:/poll/18022023/temp_image'
        image_pages=[]
        order = {}
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            photo = "{}_{}.png".format(img_name, page.number)
            photo_path = os.path.join(self.img_path, photo).replace('\\', '/')
            order[photo_path] = page.number
            pix.save(photo_path)
            image_pages.append(photo_path)
#             print('image name:', photo_path)
        image_lst = sorted(image_pages, key = lambda x:order[x])
        return image_lst
    
    def text_process(self, text):
        text = text.replace('\u200c','')
        text = text.replace("'", "")
        text = text.replace("\n", "")
        text = text.replace('౦','')
        text = text.replace('స్తీ', 'స్త్రీ')
        return text
    
    def list_remover(self, lst):
        [os.remove(i) for i in lst if os.path.exists(i)]
        return 'files are cleaned'
    
    def data_extractor(self):
        try:
            photo_path = []
            self.output_file_name = f'{self.result_path}/extracted_data.xlsx'
            while os.path.exists(self.output_file_name):
                self.output_file_name = os.path.splitext(self.output_file_name)[0]+f'{np.random.randint(3000)}.xlsx'
            workbook = xlsxwriter.Workbook(self.output_file_name)
            for direct, _, filenames in os.walk(self.files_path):
                for filename in filenames:
                    name_file, ext = os.path.splitext(filename)
                    print('name:', name_file)
                    if ext.lower() == '.pdf':   
                        pdf_path = os.path.join(direct, filename).replace('\\', '/')
                        print('pdf to image conversion started.......')
                        image_data = self.pdf_image(pdf_path)
                        print('............... image conversion completed')
                        if len(image_data) < 4:
                            print('no image data')
                            [os.remove(i) for i in image_data]
                            no_pages = pd.DataFrame([{'instruction':'Please check the given file it has to be atleast 4 pages'}])
                            no_pages.to_excel(f'{self.result_path}/{name_file}.xlsx', index = False)
                            continue
                        else:
                            rmv_lst = image_data[:2] + [image_data[-1]]
                            [os.remove(i) for i in rmv_lst]
                            image_data = image_data[2:-1]   
                        cnt = 0
                        worksheet = workbook.add_worksheet(name_file)
                        for image in image_data:                
                            img = cv2.imread(image)
                            print(f'started ----> blocks extraction from page:{image_data.index(image)+1}')
                            blocks_image = self.model_block(img)
                            df = blocks_image.pandas().xyxy[0]
                            df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
                            for xmin,ymin,xmax,ymax,thresh, annot_b in zip(df["xmin"],df["ymin"],df["xmax"],df["ymax"], df['confidence'], df['name']):
                                block_image = img[ymin:ymax, xmin:xmax]
                                features_image = self.model_feature(block_image)
                                feat_df = features_image.pandas().xyxy[0].sort_values(by = 'confidence', ascending = False)
                                feat_df = feat_df.drop_duplicates(subset = 'name', keep = 'first')
                                feat_df[['xmin', 'ymin', 'xmax', 'ymax']] = feat_df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
                                for xmin_f, ymin_f, xmax_f,\
                                ymax_f,thresh_f, annot_f in zip(feat_df['xmin'], feat_df['ymin'],\
                                                           feat_df['xmax'], feat_df['ymax'],feat_df['confidence'], feat_df['name']):
                                    feat_img = block_image[ymin_f:ymax_f, xmin_f:xmax_f]
                                    result_text_data = self.text_process(pytesseract.image_to_string(feat_img, lang = 'Telugu', config = '--psm 10'))
                                    if cnt == 0:
                                        worksheet.write('H1','Serial Number')
                                        worksheet.write('B1', 'Registration Number')
                                        worksheet.write('C1','Name')
                                        worksheet.write('D1','Gender')
                                        worksheet.write('E1','Age')
                                        worksheet.write('F1','House Number')
                                        worksheet.write('G1','Gaurdian')
                                        worksheet.write('A1','Photo')
                                        cnt += 1
                                    if annot_f == 'HouseNo':
                                        worksheet.write(f'F{cnt+1}', result_text_data)
                                    elif annot_f == 'sno':
                                        worksheet.write(f'H{cnt+1}', result_text_data)
                                    elif annot_f == 'Age':
                                        worksheet.write(f'E{cnt+1}', result_text_data)
                                    elif annot_f == 'Gaurdian':
                                        worksheet.write(f'G{cnt+1}', result_text_data)
                                    elif annot_f == 'Name':
                                        worksheet.write(f'C{cnt+1}', result_text_data)
                                    elif annot_f == 'Gender':
                                        worksheet.write(f'D{cnt+1}', result_text_data)
                                    else:
                                        worksheet.write(f'B{cnt+1}', result_text_data)
                                        feat_photo = block_image[ymax_f:, xmin_f:xmax_f]
                                        feat_photo_path = f'{self.img_path}/{name_file}_img{np.random.randint(100, 50000)}.png'
                                        while os.path.exists(feat_photo_path):
                                            feat_photo_path = f'{self.img_path}/{name_file}_img{np.random.randint(100, 50000)}.png'
                                        photo_path.append(feat_photo_path)
                                        cv2.imwrite(feat_photo_path, feat_photo)
                                        worksheet.set_row_pixels(cnt, feat_photo.shape[0])
                                        worksheet.set_column_pixels('A:A', feat_photo.shape[1])
                                        worksheet.insert_image(f'A{cnt+1}',
                                                               feat_photo_path, {'x_scale':0.9, 'y_scale':0.9,
                                                                                 'object_position':1})
                                cnt += 1
                            self.list_remover([image])
#                             os.remove(image)
                            worksheet.autofit()
            workbook.close()
            self.list_remover(photo_path)
            return f'Data extraction completed please check the result in path: {self.output_file_name}'

        except Exception as err:
            print('The code is stopped because of this error:', err)
            try:
                workbook.close()
            except:
                pass
            return err
    def __str__(self):
        return 'This object is used to extract the features from the scanned images'