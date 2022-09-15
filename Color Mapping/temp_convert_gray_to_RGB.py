#-*- coding: UTF-8 -*-
### There are some images of data sets that are gray charts, we should turn them into RGB and save them in place
### The original grayscale images and lists are saved in the Gray-Scale Images folder

import os
import cv2
from skimage.io import imread, imsave
from utils.utils import list_dir

# folder = '/mnt/d/ziqi/vis/newzone/Color-Concept-Associations-using-Google-Images-master/downloads/test/'
folder = '/mnt/d/ziqi/vis/vis_color_concept/dataset/baseline_12fruit/top/'
subfolders = list_dir(folder)
for f in subfolders:
	if f.endswith('.txt'): continue
	# subsubfolders = list_dir(folder + f + '/')
	# for ff in subsubfolders:
	#     if ff.endswith('.txt'): continue
	fig_names = list_dir(folder + f)
	for fig_name in fig_names:
		fig_file = folder + f + '/' + fig_name
		img = imread(fig_file)
		if len(img.shape) == 2:
			print(fig_file)
			imsave('./gray_' + fig_name, img)
			img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
			imsave(fig_file, img_RGB)