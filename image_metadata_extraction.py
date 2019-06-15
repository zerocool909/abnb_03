# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 01:24:56 2019

@author: adraj
"""

import cv2
import csv
import requests
import pandas as pd
import numpy as np
import cv2
from scipy.interpolate import interp1d
import shutil


df = pd.read_csv('train_airbnb_2905.csv', header = 0, encoding='latin1') 
df2 = df.loc[::700]
df2=df2[pd.notnull(df2['thumbnail_url'])]

def getImage(url):      
        try:
            response = requests.get(url, stream=True)
            with open('img.png', 'wb') as out_file:
                 shutil.copyfileobj(response.raw, out_file)
            del response
                #Load Image
            img1 = cv2.imread('img.png',3) #READ BGR
            width, height, depth = img1.shape
            maxValue = width * height * depth * 255
            imageValue = np.sum(img1)
            #Map Value between 0 and 1
            m = interp1d([0,maxValue],[0,100])
            return int(m(imageValue))
        except AttributeError:
            return 0
    
df2['rgb_value']=df2.apply(lambda x: getImage(x.thumbnail_url), axis=1)
