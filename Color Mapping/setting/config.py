#-*- coding: UTF-8 -*-
from skimage.color import lab2rgb, rgb2lab
import pandas as pd
import numpy as np
import math
import sys
import os
sys.path.append('.')
sys.path.append('..')

### 知识点：python 的类变量、类方法

def normalize(l):
	return l/np.sum(l)

def lab2lch(color):
	l, a, b = color
	c = math.sqrt(a**2 + b**2)
	h = (math.atan2(b, a) * 180 / math.pi + 360) % 360
	h = h * math.pi / 180
	return [l, c, h]


def all_settings_for_an_dataset(dataset):
	"""
		dataset: fruits12 / recycle6 / fruits5 / vegetables5 / (fruits12_cartoon / fruits12_photo)
		returns: dict
	"""
	if dataset == 'fruits12':
		all_settings = {
			### basic settings
			'concepts':			CONFIG.FRUITS_12,
			'mask_path':		CONFIG.MASK_PATH_FOR_FRUITS_12,
			'ratings_space':	58,
			'GT_ratings':		CONFIG.get_FRUITS_12_RATING_GT_58(),
			'lab_colors':		CONFIG.get_LAB_58(),
			'rgb_colors':		CONFIG.get_RGB_58(),
			'ratings_sort_idx':	CONFIG.get_sorting_idx_58_to_visualize()
			### exps
			# 'baseline':
		}


class CONFIG():

	### concepts
	FRUITS_12		= ['mango', 'watermelon', 'honeydew', 'cantaloupe', 'grapefruit', 'strawberry', 'raspberry', 'blueberry', 'avocado', 'orange', 'lime', 'lemon']
	VEGETABLES_5	= ['corn', 'carrot', 'eggplant', 'mushroom', 'celery']
	FRUITS_5		= ['peach', 'cherry', 'grape', 'banana', 'apple']
	RECYCLE_6		= ['Paper', 'Plastic', 'Glass', 'Metal', 'Compost','Trash']
	# RECYCLE_6		= ['Compost', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
	# RECYCLE_num		= [77, 69, 77, 74, 77, 70]
	RECYCLE_num		= [77, 69, 76, 73, 77, 70]

	### path for masks
	MASK_PATH_FOR_FRUITS_12			= './datasets/fruits12_all_masks.pkl'
	MASK_PATH_FOR_FRUITS_12_cartoon	= './datasets/fruits12_cartoon_all_masks.pkl'
	MASK_PATH_FOR_FRUITS_12_photo	= './datasets/fruits12_photo_all_masks.pkl'
	MASK_PATH_FOR_VEGETABLES_5		= './datasets/vegetables5_all_masks.pkl'
	MASK_PATH_FOR_FRUITS_5			= './datasets/fruits5_all_masks.pkl'
	MASK_PATH_FOR_RECYCLE_6			= './datasets/recycle6_all_masks.pkl'
	MASK_PATH_dict = {
		'fruits12':			MASK_PATH_FOR_FRUITS_12,
		'fruits12_cartoon':	MASK_PATH_FOR_FRUITS_12_cartoon,
		'fruits12_photo':	MASK_PATH_FOR_FRUITS_12_photo,
		'recycle6':			MASK_PATH_FOR_RECYCLE_6,
		'vegetables5':		MASK_PATH_FOR_VEGETABLES_5,
		'fruits5':			MASK_PATH_FOR_FRUITS_5,
	}


	### mapping from img file -> net result
	### not 100% sure right
	RESULT_ORDER = [12, 32, 31, 46, 27, 38, 29, 36, 28, 48, 
					33, 19, 18,  7, 24, 47, 37, 10, 49,  1, 
					 4, 15,  3, 13, 11, 39,  8,  0, 44,  2, 
					14,  6,  5, 26, 42, 34, 45,  9, 20, 40, 
					41, 21, 22, 25, 16, 23, 30, 35, 17, 43]

	RESULT_ORDER2 = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 
					18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 
					27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 
					36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 
					45, 46, 47, 48, 49, 5, 6, 7, 8, 9]

	RESULT_ORDER3 = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 
					18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 
					27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 
					36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 
					45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 
					54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 
					63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 
					72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 
					81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 
					90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

	### color ratings
	LAB_58	= None
	RGB_58	= None
	LAB_71	= None
	RGB_71	= None
	LAB_37	= None
	RGB_37	= None
	LAB_Tableau20 = None
	RGB_Tableau20 = None
	LAB_EXP30 = None
	RGB_EXP30 = None
	LAB_EXP27 = None
	RGB_EXP27 = None
	LAB_EXP28 = None
	RGB_EXP28 = None
	LAB_UWE65 = None
	RGB_UWE65 = None
	LAB_UWE66 = None
	RGB_UWE66 = None
	FRUITS_12_RATING_GT_58		= None
	FRUITS_5_RATING_GT_71		= None
	VEGETABLES_5_RATING_GT_71	= None
	RECYCLE_6_RATING_GT_37		= None
	
	DISTANCE_MATRIX_L2_LAB	= {}

	### 313 ab space
	AB_PTS_IN_HULL	= None
	GAMUT_MASK		= None

	@classmethod
	def build_gamut_mask(cls):
		pts_in_hull = cls.get_AB_PTS_IN_HULL()
		pts = pd.DataFrame({'a':pts_in_hull[:,0], 'b':pts_in_hull[:,1]})
		mask = np.zeros((221, 221, 1)).astype(int)
		smallest_a = -90
		biggest_a = 100
		for A in range(smallest_a, 10, 10):
			pts_row = pts[pts['a'] == A]
			min_row_b = pts_row['b'].min()
			max_row_b = pts_row['b'].max()
			for a in range(A, A+10):
				for b in range(min_row_b, max_row_b+1):
					mask[a+110, b+110] = 1
		for A in reversed(range(-10, biggest_a+1, 10)):
			pts_row = pts[pts['a'] == A]
			min_row_b = pts_row['b'].min()
			max_row_b = pts_row['b'].max()
			for a in range(A-9, A+1):
				for b in range(min_row_b, max_row_b+1):
					mask[a+110, b+110] = 1
		cls.GAMUT_MASK = mask

	@classmethod
	def check_ab_in_gamut(cls, a, b):
		if cls.GAMUT_MASK is None:
			cls.build_gamut_mask()
		return (cls.GAMUT_MASK[a+110, b+110] == 1)

	@classmethod
	def get_LAB_58(cls):
		if cls.LAB_58 is None:
			cls.LAB_58 = np.loadtxt('./settings/58lab.csv', delimiter=',')
		return cls.LAB_58
	
	@classmethod
	def get_RGB_Tableau_20(cls):
		if cls.RGB_Tableau20 is None:
			cls.RGB_Tableau20 = np.loadtxt('./settings/Tableau20_rgb.csv', delimiter=',')
			cls.RGB_Tableau20 = cls.RGB_Tableau20 / 255.0
		return cls.RGB_Tableau20

	@classmethod
	def get_LAB_Tableau_20(cls):
		if cls.LAB_Tableau20 is None:
			rgb = cls.get_RGB_Tableau_20()
			cls.LAB_Tableau20 = rgb2lab(rgb)
		return cls.LAB_Tableau20

	@classmethod
	def get_RGB_EXP_30(cls):
		if cls.RGB_EXP30 is None:
			cls.RGB_EXP30 = np.loadtxt('./settings/Tableau20_plus_10_rgb.csv', delimiter=',')
			cls.RGB_EXP30 = cls.RGB_EXP30 / 255.0
		return cls.RGB_EXP30

	@classmethod
	def get_LAB_EXP_30(cls):
		if cls.LAB_EXP30 is None:
			rgb = cls.get_RGB_EXP_30()
			cls.LAB_EXP30 = rgb2lab(rgb)
		return cls.LAB_EXP30

	@classmethod
	def get_RGB_EXP_27(cls):
		if cls.RGB_EXP27 is None:
			cls.RGB_EXP27 = np.loadtxt('./settings/Tableau20_plus_7_rgb.csv', delimiter=',')
			cls.RGB_EXP27 = cls.RGB_EXP27 / 255.0
		return cls.RGB_EXP27

	@classmethod
	def get_LAB_EXP_27(cls):
		if cls.LAB_EXP27 is None:
			rgb = cls.get_RGB_EXP_27()
			cls.LAB_EXP27 = rgb2lab(rgb)
		return cls.LAB_EXP27

	@classmethod
	def get_RGB_EXP_28(cls):
		if cls.RGB_EXP28 is None:
			cls.RGB_EXP28 = np.loadtxt('./settings/Tableau20_plus_8_rgb_drinks.csv', delimiter=',')
			cls.RGB_EXP28 = cls.RGB_EXP28 / 255.0
		return cls.RGB_EXP28

	@classmethod
	def get_LAB_EXP_28(cls):
		if cls.LAB_EXP28 is None:
			rgb = cls.get_RGB_EXP_28()
			cls.LAB_EXP28 = rgb2lab(rgb)
		return cls.LAB_EXP28

	@classmethod
	def get_RGB_UWE_65(cls):
		if cls.RGB_UWE65 is None:
			cls.RGB_UWE65 = np.loadtxt('./settings/UW58_plus_7_rgb.csv', delimiter=',')
		return cls.RGB_UWE65

	@classmethod
	def get_LAB_UWE_65(cls):
		if cls.LAB_UWE65 is None:
			rgb = cls.get_RGB_UWE_65()
			cls.LAB_UWE65 = rgb2lab(rgb)
		return cls.LAB_UWE65

	@classmethod
	def get_RGB_UWE_66(cls):
		if cls.RGB_UWE66 is None:
			cls.RGB_UWE66 = np.loadtxt('./settings/UW58_plus_8_rgb_drinks.csv', delimiter=',')
		return cls.RGB_UWE66

	@classmethod
	def get_LAB_UWE_66(cls):
		if cls.LAB_UWE66 is None:
			rgb = cls.get_RGB_UWE_66()
			cls.LAB_UWE66 = rgb2lab(rgb)
		return cls.LAB_UWE66

	@classmethod
	def get_RGB_58(cls):
		if cls.RGB_58 is None:
			cls.RGB_58 = np.loadtxt('./settings/58rgb.csv', delimiter=',')
		return cls.RGB_58

	@classmethod
	def get_LAB_71(cls):
		if cls.LAB_71 is None:
			cls.LAB_71 = np.loadtxt('./settings/71lab.csv', delimiter=',')
		return cls.LAB_71
	
	@classmethod
	def get_RGB_71(cls):
		if cls.RGB_71 is None:
			cls.RGB_71 = np.loadtxt('./settings/71rgb.csv', delimiter=',')
		return cls.RGB_71

	@classmethod
	def get_LAB_37(cls):
		if cls.LAB_37 is None:
			cls.LAB_37 = np.loadtxt('./settings/37lab.csv', delimiter=',')
		return cls.LAB_37
	
	@classmethod
	def get_RGB_37(cls):
		if cls.RGB_37 is None:
			cls.RGB_37 = np.loadtxt('./settings/37rgb.csv', delimiter=',')
		return cls.RGB_37

	@classmethod
	def get_FRUITS_12_RATING_GT_58(cls):
		if cls.FRUITS_12_RATING_GT_58 is None:
			ratings_gt_58 = np.loadtxt('./settings/12fruits_58GT_ratings.csv', delimiter=',')
			for i in range(len(ratings_gt_58)):
				ratings_gt_58[i] = normalize(ratings_gt_58[i])
			cls.FRUITS_12_RATING_GT_58 = dict(zip(cls.FRUITS_12, ratings_gt_58))
		return cls.FRUITS_12_RATING_GT_58

	@classmethod
	def get_FRUITS_5_RATING_GT_71(cls):
		if cls.FRUITS_5_RATING_GT_71 is None:
			ratings_gt_71 = np.loadtxt('./settings/5fruits_71GT_ratings.csv', delimiter=',')
			for i in range(len(ratings_gt_71)):
				ratings_gt_71[i] = normalize(ratings_gt_71[i])
			cls.FRUITS_5_RATING_GT_71 = dict(zip(cls.FRUITS_5, ratings_gt_71))
		return cls.FRUITS_5_RATING_GT_71

	@classmethod
	def get_VEGETABLES_5_RATING_GT_71(cls):
		if cls.VEGETABLES_5_RATING_GT_71 is None:
			ratings_gt_71 = np.loadtxt('./settings/5vegetable_71GT_ratings.csv', delimiter=',')
			for i in range(len(ratings_gt_71)):
				ratings_gt_71[i] = normalize(ratings_gt_71[i])
			cls.VEGETABLES_5_RATING_GT_71 = dict(zip(cls.VEGETABLES_5, ratings_gt_71))
		return cls.VEGETABLES_5_RATING_GT_71

	@classmethod
	def get_RECYCLE_6_RATING_GT_37(cls):
		if cls.RECYCLE_6_RATING_GT_37 is None:
			ratings_gt_37 = np.loadtxt('./settings/6recycle_37GT_ratings.csv', delimiter=',')
			for i in range(len(ratings_gt_37)):
				ratings_gt_37[i] = normalize(ratings_gt_37[i])
			cls.RECYCLE_6_RATING_GT_37 = dict(zip(cls.RECYCLE_6, ratings_gt_37))
		return cls.RECYCLE_6_RATING_GT_37

	@classmethod
	def get_AB_PTS_IN_HULL(cls):
		if cls.AB_PTS_IN_HULL is None:
			pts_in_hull = np.load('./settings/pts_in_hull.npy').tolist()
			pts_in_hull.sort(key = lambda x:(x[0], x[1]))
			cls.AB_PTS_IN_HULL = np.array(pts_in_hull)
		return cls.AB_PTS_IN_HULL

	@classmethod
	def get_sorting_idx_58_to_visualize(cls):
		lch_58_colors = []
		for lab in cls.get_LAB_58():
			lch_58_colors.append(lab2lch(lab))
		h_58_colors = np.array(lch_58_colors)[:, 2]
		sorting_idx = np.argsort(h_58_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_58_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_58_to_visualize()
		return sorting_idx, cls.get_RGB_58()[sorting_idx]

	@classmethod
	def get_sorting_idx_71_to_visualize(cls):
		lch_71_colors = []
		for lab in cls.get_LAB_71():
			lch_71_colors.append(lab2lch(lab))
		h_71_colors = np.array(lch_71_colors)[:, 2]
		sorting_idx = np.argsort(h_71_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_71_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_71_to_visualize()
		return sorting_idx, cls.get_RGB_71()[sorting_idx]

	@classmethod
	def get_sorting_idx_37_to_visualize(cls):
		lch_37_colors = []
		for lab in cls.get_LAB_37():
			lch_37_colors.append(lab2lch(lab))
		h_37_colors = np.array(lch_37_colors)[:, 2]
		sorting_idx = np.argsort(h_37_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_37_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_37_to_visualize()
		return sorting_idx, cls.get_RGB_37()[sorting_idx]

	@classmethod
	def get_sorting_idx_Tableau20_to_visualize(cls):
		lch_Tableau20_colors = []
		for lab in cls.get_LAB_Tableau_20():
			lch_Tableau20_colors.append(lab2lch(lab))
		h_colors = np.array(lch_Tableau20_colors)[:, 2]
		sorting_idx = np.argsort(h_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_Tableau20_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_Tableau20_to_visualize()
		return sorting_idx, cls.get_RGB_Tableau_20()[sorting_idx]

	@classmethod
	def get_sorting_idx_EXP30_to_visualize(cls):
		lch_EXP30_colors = []
		for lab in cls.get_LAB_EXP_30():
			lch_EXP30_colors.append(lab2lch(lab))
		h_colors = np.array(lch_EXP30_colors)[:, 2]
		sorting_idx = np.argsort(h_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_EXP30_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_EXP30_to_visualize()
		return sorting_idx, cls.get_RGB_EXP_30()[sorting_idx]

	@classmethod
	def get_sorting_idx_EXP27_to_visualize(cls):
		lch_EXP27_colors = []
		for lab in cls.get_LAB_EXP_27():
			lch_EXP27_colors.append(lab2lch(lab))
		h_colors = np.array(lch_EXP27_colors)[:, 2]
		sorting_idx = np.argsort(h_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_EXP27_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_EXP27_to_visualize()
		return sorting_idx, cls.get_RGB_EXP_27()[sorting_idx]

	@classmethod
	def get_sorting_idx_EXP28_to_visualize(cls):
		lch_EXP28_colors = []
		for lab in cls.get_LAB_EXP_28():
			lch_EXP28_colors.append(lab2lch(lab))
		h_colors = np.array(lch_EXP28_colors)[:, 2]
		sorting_idx = np.argsort(h_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_EXP28_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_EXP28_to_visualize()
		return sorting_idx, cls.get_RGB_EXP_28()[sorting_idx]

	@classmethod
	def get_sorting_idx_UWE65_to_visualize(cls):
		lch_UWE65_colors = []
		for lab in cls.get_LAB_UWE_65():
			lch_UWE65_colors.append(lab2lch(lab))
		h_colors = np.array(lch_UWE65_colors)[:, 2]
		sorting_idx = np.argsort(h_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_UWE65_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_UWE65_to_visualize()
		return sorting_idx, cls.get_RGB_UWE_65()[sorting_idx]

	@classmethod
	def get_sorting_idx_UWE66_to_visualize(cls):
		lch_UWE66_colors = []
		for lab in cls.get_LAB_UWE_66():
			lch_UWE66_colors.append(lab2lch(lab))
		h_colors = np.array(lch_UWE66_colors)[:, 2]
		sorting_idx = np.argsort(h_colors)
		return sorting_idx

	@classmethod
	def get_sorted_RGB_UWE66_to_visulize(cls):
		sorting_idx = cls.get_sorting_idx_UWE66_to_visualize()
		return sorting_idx, cls.get_RGB_UWE_66()[sorting_idx]

	""""""""

	@classmethod
	def cal_lab_color_L2_distance_matrix(cls, color_space):
		if   color_space == 58: labs = cls.get_LAB_58()
		elif color_space == 71: labs = cls.get_LAB_71()
		elif color_space == 37: labs = cls.get_LAB_37()
		distance_matrix = np.zeros((color_space, color_space))
		for i in range(color_space):
			for j in range(color_space):
				d = np.sum((labs[i] - labs[j])**2)
				distance_matrix[i, j] = d**0.5
		return distance_matrix

	@classmethod
	def get_lab_color_L2_distance_matrix(cls, color_space):
		file_name = './settings/LAB_distance_matrix_L2_%d.npy' % color_space
		if color_space in cls.DISTANCE_MATRIX_L2_LAB.keys():
			return cls.DISTANCE_MATRIX_L2_LAB[color_space]
		elif os.path.exists(file_name):
			cls.DISTANCE_MATRIX_L2_LAB[color_space] = np.load(file_name)	
		else:
			distance_matrix = cls.cal_lab_color_L2_distance_matrix(color_space)
			np.save(file_name, distance_matrix)
			cls.DISTANCE_MATRIX_L2_LAB[color_space] = distance_matrix
		return cls.DISTANCE_MATRIX_L2_LAB[color_space]

	@classmethod
	def get_lab_color_L2_distance_matrix_58(cls):
		if cls.DISTANCE_MATRIX_L2_LAB_58 is None:
			cls.DISTANCE_MATRIX_L2_LAB_58 = cls.get_lab_color_L2_distance_matrix(58)
		return cls.DISTANCE_MATRIX_L2_LAB_58

	@classmethod
	def cal_lab_L2_distances_for_pts_in_hull(cls):
		pts = cls.get_AB_PTS_IN_HULL()
		pts_distances = np.sum(pts**2, axis=1)**0.5
		return pts_distances

	@classmethod
	def get_weights_for_pts_in_hull(cls, sigma=1.0):
		pts_distances = cls.cal_lab_L2_distances_for_pts_in_hull()
		d = pts_distances / pts_distances.max()		### TODO: normalization method
		weights = np.exp( (d / sigma) **2 )			### TODO: sigma
		return weights


