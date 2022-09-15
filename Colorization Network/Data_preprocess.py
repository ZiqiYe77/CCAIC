#coding:utf-8
import os 
import glob
from imageio import imread,imsave
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt


path='./dataset/'#image dataset path
cName = ['mango','watermelon','honeydew','cantaloupe','grapefruit','strawberry','raspberry','blueberry','avocado','orange','lime','lemon']#image concept
dataFile=open('./train.txt','a')#Image path list
for i in range(len(cName)): 
    file_name = path + cName[i]
    trainFiles = glob.glob(file_name+"/*.png")

    for j in range(len(trainFiles)):
        print(trainFiles[j])
        ImgName=trainFiles[j].split('/')[3]
        dataFile.write(cName[i]+'/'+ImgName+' '+str(i)+'\n')