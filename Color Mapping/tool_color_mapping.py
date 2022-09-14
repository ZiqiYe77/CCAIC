#-*- coding: UTF-8 -*-
import os
import itertools
import pickle
import pandas as pd
import numpy as np
# from time import time
from tqdm import tqdm
from scipy.special import softmax

from settings.config import CONFIG
from utils.utils import count_lines
# from cal_metrics import normalize

"""
	Tools
"""

def normalize(l):
	return l/np.sum(l)


def get_mask(dataset, concept=None, idx=None):
	"""
	read image masks, which is saved with pickle
	Input:
		dataset:	str	|	fruits12 / fruits12_cartoon / fruits12_photo / recycle6 / vegetables5 / fruits5
		concept:	str or None
		idx:		int or None
	Output:
		if concept is None:
			all_masks:	dict, where concepts are keys
		if idx is None:
			masks_map:	(num_img, num_pixels) bool array
		else:
			mask:		(num_pixels,) bool array
	"""
	mask_path = CONFIG.MASK_PATH_dict[dataset]
	### the dataset's all concepts' masks in a dict
	with open(mask_path, 'rb') as f:
		all_masks = pickle.load(f)
	if concept is None: return all_masks
	### the concept's all masks in a (num_img, num_pixels) array
	masks_map = all_masks[concept]
	if idx is None: return masks_map
	### mask for an image (num_pixels,)
	mask = masks_map[idx]
	return mask


################## tools for LAB distances ##################
### distance计算后保存在setting文件夹里，每次使用都直接读文件而没有重新算
### 如果要修改distance的计算方式，需要注意修改这点

def cal_LAB_L2_distance(Lab1, Lab2):
	"""
	calculate L2 distance for Lab colors
	Input:
		Lab1: (x, 3) array
		Lab2: (3,) array
	Output:
		distance (x,) array
	"""
	distance = np.sum( (Lab1 - Lab2) **2 , axis=1)
	distance = distance**0.5
	return distance


def cal_LAB_CIEDE2000_distance(Lab1, Lab2):
	"""
	calculate L2 distance for Lab colors
	Input:
		Lab1: (x, 3) array
		Lab2: (3,) array
	Output:
		distance (x,) array
	"""
	from pyciede2000 import ciede2000
	distance = np.zeros(Lab1.shape[0])
	for idx, lab1 in enumerate(Lab1):
		distance[idx] = ciede2000(lab1, Lab2)['delta_E_00']
	return distance


### quantize L into 25/50/100 groups
def get_quantized_L(num_groups):
	""" get L for each group """
	first_right_edge = 100.0 / num_groups
	bin_half = first_right_edge / 2
	L_quantized = np.linspace(bin_half, 100-bin_half, num_groups)
	return L_quantized


def get_all_lab_combinations(num_groups):
	"""get all Lab combinations"""
	L_quantized = get_quantized_L(num_groups)
	ab_channel = CONFIG.get_AB_PTS_IN_HULL()
	all_l	= [[l] for l in L_quantized]
	all_ab	= ab_channel.tolist()
	itr_lab	= itertools.product(all_l, all_ab)
	all_lab	= [np.concatenate(lab) for lab in itr_lab]
	return np.array(all_lab)


def get_distance_map(num_groups, ratings_space):
	"""
		num_groups: 25 / 50 / 100
		ratings_space: 58 / 71 / 37 / 20 / 30 / 27
	"""
	distance_file = './settings/LAB_distance_to%d_L%d.csv' % (ratings_space, num_groups)
	if os.path.exists(distance_file):
		distance_df = pd.read_csv(distance_file)
	else:
		if ratings_space == 58: target_LAB = CONFIG.get_LAB_58()
		if ratings_space == 71: target_LAB = CONFIG.get_LAB_71()
		if ratings_space == 37: target_LAB = CONFIG.get_LAB_37()
		if ratings_space == 20: target_LAB = CONFIG.get_LAB_Tableau_20()
		if ratings_space == 30: target_LAB = CONFIG.get_LAB_EXP_30()
		if ratings_space == 27: target_LAB = CONFIG.get_LAB_EXP_27()
		if ratings_space == 28: target_LAB = CONFIG.get_LAB_EXP_28()
		# if ratings_space == 65: target_LAB = CONFIG.get_LAB_UWE_65()
		if ratings_space == 65: target_LAB = CONFIG.get_LAB_UWE_66() #TODO temp
		# if ratings_space == 66: target_LAB = CONFIG.get_LAB_UWE_66()
		# LAB_58	= CONFIG.get_LAB_58()
		all_lab	= get_all_lab_combinations(num_groups)
		distance_dict = {}
		for idx, target_lab in enumerate(target_LAB):
			dis = cal_LAB_L2_distance(all_lab, target_lab)
			distance_dict['LAB_%d'%idx] = dis
		distance_df = pd.DataFrame(distance_dict)
		distance_df.to_csv(distance_file, index=False)
	return distance_df


################## tools for weights ##################


def Z_Score(data):
	# faster than scipy.stats.zscore on small-scale data
	lenth = len(data)
	total = sum(data)
	ave = float(total)/lenth
	tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
	tempsum = pow(float(tempsum)/lenth,0.5)
	for i in range(lenth):
		data[i] = (data[i] - ave)/tempsum
	return data


def min_max(data):
	mmin = np.min(data)
	mmax = np.max(data)
	standard_data = (data - mmin) / (mmax  - mmin)
	return standard_data


def get_weights_map(num_groups, ratings_space, beta=1.0, standardization='z-score'):
	"""
		standardization: z-score / min-max
	"""
	# weights_file = './settings/LAB_weights_to58_L%d.txt' % num_groups
	# if os.path.exists(weights_file):
	# 	weights_map = np.loadtxt(weights_file)
	# else:
	distance_df = get_distance_map(num_groups, ratings_space)
	distance_map = distance_df.to_numpy()	## (num_groups*313 colors, 58/71/37 colors)
	weights_map = np.zeros_like(distance_map)
	for row_id, distances in enumerate(distance_map):		# every color in 7825 space to 58/71/37/20 space
		if standardization == 'z-score':
			dis_stand = Z_Score(distances)	# standardization by z-score
		elif standardization == 'min-max':
			dis_stand = min_max(distances)	# or standardization by min-max
		dis_stand = - dis_stand				# get minus
		dis_stand = dis_stand * beta		# parameter to tune
		weights = softmax(dis_stand)		# softmax
		weights_map[row_id] = weights
	# np.savetxt(weights_file, weights_map)
	return weights_map


################## tools for reading network outputs ##################


def read_network_output_txt(file_path):
	# output should be: pixel_num * 313 (max: 981568)
	# columns: id, L, a, b, probability
	# prob_313_all = np.loadtxt(file_path)
	# ab_channel	= np.loadtxt(file_path, usecols=(2, 3), max_rows=313)	### CONFIG.AB_PTS_IN_HULL
	num_lines	= count_lines(file_path)
	num_pixels	= int(num_lines / 313)
	results		= np.loadtxt(file_path, usecols=(1, 4))
	probs_labs	= results[:, 1].reshape(num_pixels,-1)
	lightness	= results[:, 0]
	lightness	= np.array([lightness[313*i] for i in range(num_pixels)])
	# return ab_channel, lightness, probs_labs
	return lightness, probs_labs


def read_network_output_npy1(file_path):
	# output should be: pixel_num * 313 (max: 981568)
	# columns: id, L, a, b, probability
	results		= np.load(file_path)
	num_lines	= results.shape[0]
	num_pixels	= int(num_lines / 313)
	probs_labs	= results[:, 1].reshape(num_pixels,-1)	# (num_px, 313)
	lightness	= results[:, 0]
	lightness	= np.array([lightness[313*i] for i in range(num_pixels)])
	return lightness, probs_labs


def read_network_output(folder, idx):
	L_file		= folder + '/L_%d.npy'	% idx
	probs_file	= folder + '/pro_%d.npy'% idx
	lightness	= np.load(L_file)			# (num_px,)
	probs_labs	= np.load(probs_file)		# (num_px, 313)
	# num_pixels	= probs_labs.shape[0]
	return lightness, probs_labs


################## tools for trails of tuning results, all abandoned at last ##################


def probs_temporature_scheme(probs, T=1.0):		# T=0.38
	"""
		probs: (px_num, 313)
	"""
	probs = np.log(probs)
	probs = probs / T
	probs = np.exp(probs)
	probs = probs / np.sum(probs, axis=1)[:, None] # or: .reshape(-1, 1)
	probs[np.isnan(probs)] = 0.0	# will be all nan the line is all 0
	return probs


def probs_ab_weighting(probs, sigma):
	"""
		probs: (num_pixels, 313)
		sigma: float
	"""
	assert isinstance(sigma, float)
	ab_weights = CONFIG.get_weights_for_pts_in_hull(sigma)
	num_pixels = probs.shape[0]
	for px_idx in range(num_pixels):
		p = probs[px_idx] * ab_weights
		probs[px_idx] = normalize(p)
	return probs


def probs_ab_weighting_dewhite(probs, white_range, sigma):
	"""
		probs: (num_pixels, 313)
		white_range: int
		sigma: float
	"""
	assert isinstance(sigma, float)
	# ab_weights = CONFIG.get_weights_for_pts_in_hull(sigma)
	pts = CONFIG.get_AB_PTS_IN_HULL()
	white_point_idxes = []
	for pt_idx, pt in enumerate(pts):
		if pt[0] + pt[1] <= white_range:
			white_point_idxes.append(pt_idx)
	num_pixels = probs.shape[0]
	for px_idx in range(num_pixels):
		for pt_idx in white_point_idxes:
			probs[px_idx][pt_idx] *= sigma
		probs[px_idx] = normalize(probs[px_idx])
	return probs


def probs_pixelwise_ab_weighting_wrong(probs, sigma):
	"""
		probs: (num_pixels, 313)
		sigma: float
	"""
	assert isinstance(sigma, float)
	ab_weights = CONFIG.get_weights_for_pts_in_hull(sigma)
	num_pixels = probs.shape[0]
	weights_pixels = np.zeros(num_pixels)
	for px_idx in range(num_pixels):
		weights_pixels[px_idx] = np.sum(probs[px_idx] * ab_weights)
	weights_pixels = normalize(weights_pixels)
	for px_idx in range(num_pixels):
		probs[px_idx] = probs[px_idx] * weights_pixels[px_idx]
	return probs


def probs_pixelwise_ab_weighting(probs, sigma):
	"""
		probs: (num_pixels, 313)
		sigma: float
	"""
	assert isinstance(sigma, float)
	pts = CONFIG.get_AB_PTS_IN_HULL()	# (313, 2)
	num_pixels = probs.shape[0]
	estimate_abs = np.zeros((num_pixels, 2))
	for px_idx in range(num_pixels):
		ab = (pts.T * probs[px_idx]).T
		ab = np.sum(ab, axis=0)
		estimate_abs[px_idx] = ab
	px_dis_to_00 = np.sum(estimate_abs**2, axis=1)**0.5
	d = px_dis_to_00 / px_dis_to_00.max()
	weights_pixels = np.exp( (d / sigma) **2 )
	probs_weighted = np.zeros_like(probs)
	for px_idx in range(num_pixels):
		probs_weighted[px_idx] = probs[px_idx] * weights_pixels[px_idx]
	return probs_weighted


def addup_same_color_probs(probs_labs, group_idx, num_groups, T2=1.0):
	"""
	Input:
		probs_labs: (num_px * 313,)
		group_idx: (num_px,)
		num_groups: 25 / 50 / 100
	Output:
		probs_addup: (313*num_groups,)
	"""
	probs_addup	= np.zeros((num_groups, 313))
	probs_labs	= probs_labs.reshape(-1, 313)	# (num_pixels, 313)
	for L_idx in range(num_groups):
		pixel_idx = np.argwhere(group_idx==L_idx).reshape(-1)
		probs = probs_labs[pixel_idx]
		probs_addup[L_idx] = np.sum(probs, axis=0)
	if T2 != 1.0:
		probs_addup = probs_temporature_scheme(probs_addup, T=T2)	### TODO
	return probs_addup.reshape(-1)


################## tools for exps, use all functions above ##################


def lightness_grouping(lightness, num_groups):
	""" turn lightness into quantized for each pixel """
	### L channel in CIELab space are in [0, 100]
	### idx for lightness smaller than first_right_edge will be 0
	### largest idx is num_groups-1
	first_right_edge = 100.0 / num_groups
	bins = np.linspace(first_right_edge, 100, num_groups)
	group_idx = np.digitize(lightness, bins, right=True)
	group_idx[lightness>100] = num_groups-1			## there are some L=101+
	# L_quantized = get_quantized_L(num_groups)
	# quantized_lightness = L_quantized[group_idx]
	# return quantized_lightness
	# np.save('group_idx.npy', group_idx)
	# exit()
	return group_idx


def mapping_pipeline(folder, idx, num_groups, ratings_space, mask=None, 
						T1=1.0, T2=1.0, beta=1.0, use_ab_weight=False, 
						sigma=None, do_dewhite=False, white_range=20,
						standardization='z-score'):
	### read network outputs
	### lightness: (num_pixels,)
	### probs_labs: (num_pixels, 313)
	# file_path = folder + '%d.txt'%idx
	# lightness, probs_labs = read_network_output(file_path)
	lightness, probs_labs = read_network_output(folder, idx)
	### mask scheme
	### mask: bool(num_pixels, )
	if mask is not None:
		lightness	= lightness[mask]
		probs_labs	= probs_labs[mask]
	### arange per-pixel 313 probability
	# if use_ab_weight: probs_labs = probs_ab_weighting(probs_labs, sigma)
	if do_dewhite: probs_labs = probs_ab_weighting_dewhite(probs_labs, white_range, sigma)
	if use_ab_weight: probs_labs = probs_pixelwise_ab_weighting(probs_labs, sigma)
	if T1 != 1.0: probs_labs = probs_temporature_scheme(probs_labs, T=T1)	### abandon
	### quantize lightness
	group_idx = lightness_grouping(lightness, num_groups)
	### mapping probability to (num_groups*313, )
	probs = addup_same_color_probs(probs_labs, group_idx, num_groups, T2=T2)
	### get weights from LAB distances (num_groups*313, 58/71/37)
	weights_map = get_weights_map(num_groups, ratings_space, beta=beta, standardization=standardization)
	### dot product to get the voting
	ratings = probs.dot(weights_map)
	# np.save('weights_map.npy', weights_map)
	# np.save('probs.npy', probs)
	# exit()
	return ratings


def get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1=1.0, T2=1.0, beta=1.0, figs_num=50,
									use_ab_weight=False, sigma=None, do_dewhite=False, white_range=None, standardization='z-score'):
	"""
		ratings_space: 58 / 71 / 37 / 20 / 30
	"""
	# final_ratings = np.zeros(ratings_space)
	per_img_ratings = np.zeros((figs_num, ratings_space))
	# net_results = list_dir(folder)
	# for file_name in tqdm(net_results):
		# file_path = folder + '/' + file_name
	for idx in tqdm(range(figs_num)):
		mask = None if masks_map is None else masks_map[idx]
		# img_ratings = mapping_pipeline(folder, idx, num_groups, ratings_space, mask, T1, T2, beta=beta)
		img_ratings = mapping_pipeline(folder, idx, num_groups, ratings_space, mask, T1, T2, beta=beta, 
										use_ab_weight=use_ab_weight, sigma=sigma, 
										do_dewhite=do_dewhite, white_range=white_range, 
										standardization=standardization)
		# final_ratings += img_ratings
		per_img_ratings[idx] = img_ratings
	final_ratings = np.sum(per_img_ratings, axis=0)
	return final_ratings, per_img_ratings



""" EXPs """


def deal_fruits12_exp(folder, save_file, num_groups, mask_type, T1=1.0, T2=1.0, beta=1.0,
						sigma=None, do_dewhite=False, white_range=20, standardization='z-score'):
	do_mask = True if mask_type == 'mask' else False
	fruits12 = CONFIG.FRUITS_12
	ratings_space = 58
	results = np.zeros((12, 58))
	if do_mask:
		with open(CONFIG.MASK_PATH_FOR_FRUITS_12, 'rb') as f:
			all_masks = pickle.load(f)
	for idx, f in enumerate(fruits12):
		print(f)
		masks_map = all_masks[f] if do_mask else None
		f_ratings58, per_img_ratings = get_ratings_for_one_concept(folder + '/' + f + '/', 
												num_groups, ratings_space, masks_map, T1, T2, beta,
												sigma=sigma, do_dewhite=do_dewhite, white_range=white_range, standardization=standardization)
		results[idx] = f_ratings58
	np.savetxt(save_file, results, delimiter=',')


def deal_fruits12_cartoon_exp(folder, save_file, num_groups, mask_type, T1=1.0, T2=1.0, beta=1.0, save_each_img=False):
	do_mask = True if mask_type == 'mask' else False
	fruits12 = CONFIG.FRUITS_12
	ratings_space = 58
	results = np.zeros((12, 58))
	if do_mask:
		with open(CONFIG.MASK_PATH_FOR_FRUITS_12_cartoon, 'rb') as f:
			all_masks = pickle.load(f)
	for idx, f in enumerate(fruits12):
		print(f)
		masks_map = all_masks[f] if do_mask else None
		f_ratings58, per_img_ratings = get_ratings_for_one_concept(folder + '/' + f + '/', num_groups, ratings_space, masks_map, T1, T2, beta)
		results[idx] = f_ratings58
		### save ratings for each img
		if not save_each_img: continue
		if 'pretrain' in save_file: continue
		for i in range(per_img_ratings.shape[0]):
			per_img_save_file = '/'.join(save_file.split('/')[:-1]) + '/fruits12_cartoon/' + f + '/'
			if not os.path.exists(per_img_save_file): os.mkdir(per_img_save_file)
			per_img_save_file += '/%d-%s.txt' % (i, mask_type)
			np.savetxt(per_img_save_file, per_img_ratings[i])
	np.savetxt(save_file, results, delimiter=',')


def deal_fruits12_photo_exp(folder, save_file, num_groups, mask_type, T1=1.0, T2=1.0, beta=1.0):
	do_mask = True if mask_type == 'mask' else False
	fruits12 = CONFIG.FRUITS_12
	ratings_space = 58
	results = np.zeros((12, 58))
	if do_mask:
		with open(CONFIG.MASK_PATH_FOR_FRUITS_12_photo, 'rb') as f:
			all_masks = pickle.load(f)
	for idx, f in enumerate(fruits12):
		print(f)
		masks_map = all_masks[f] if do_mask else None
		f_ratings58, per_img_ratings = get_ratings_for_one_concept(folder + '/' + f + '/', num_groups, ratings_space, masks_map, T1, T2, beta)
		results[idx] = f_ratings58
	np.savetxt(save_file, results, delimiter=',')


def deal_fruits5_exp(folder, save_file, num_groups, mask_type, T1=1.0, T2=1.0, beta=1.0):
	do_mask = True if mask_type == 'mask' else False
	fruits5 = CONFIG.FRUITS_5
	ratings_space = 71
	results = np.zeros((5, 71))
	if do_mask:
		with open(CONFIG.MASK_PATH_FOR_FRUITS_5, 'rb') as f:
			all_masks = pickle.load(f)
	for idx, f in enumerate(fruits5):
		print(f)
		masks_map = all_masks[f] if do_mask else None
		f_ratings71, per_img_ratings = get_ratings_for_one_concept(folder + '/' + f + '/', num_groups, ratings_space, masks_map, T1, T2, beta)
		results[idx] = f_ratings71
	np.savetxt(save_file, results, delimiter=',')


def deal_vegetable5_exp(folder, save_file, num_groups, mask_type, T1=1.0, T2=1.0, beta=1.0):
	do_mask = True if mask_type == 'mask' else False
	vegetable5 = CONFIG.VEGETABLES_5
	ratings_space = 71
	results = np.zeros((5, 71))
	if do_mask:
		with open(CONFIG.MASK_PATH_FOR_VEGETABLES_5, 'rb') as f:
			all_masks = pickle.load(f)
	for idx, v in enumerate(vegetable5):
		print(v)
		masks_map = all_masks[v] if do_mask else None
		v_ratings71, per_img_ratings = get_ratings_for_one_concept(folder + '/' + v + '/', num_groups, ratings_space, masks_map, T1, T2, beta)
		results[idx] = v_ratings71
	np.savetxt(save_file, results, delimiter=',')


def deal_recycle6_exp(folder, save_file, num_groups, mask_type, T1=1.0, T2=1.0, beta=1.0):
	do_mask = True if mask_type == 'mask' else False
	recycle6 = CONFIG.RECYCLE_6
	figs_num = 50
	# figs_nums = CONFIG.RECYCLE_num
	ratings_space = 37
	results = np.zeros((6, 37))
	if do_mask:
		with open(CONFIG.MASK_PATH_FOR_RECYCLE_6, 'rb') as f:
			all_masks = pickle.load(f)
	for idx, r in enumerate(recycle6):
		print(r)
		# figs_num = figs_nums[idx]
		masks_map = all_masks[r] if do_mask else None
		r_ratings37, per_img_ratings = get_ratings_for_one_concept(folder + '/' + r + '/', num_groups, ratings_space, masks_map, T1, T2, beta, figs_num=figs_num)
		results[idx] = r_ratings37
	np.savetxt(save_file, results, delimiter=',')


#### wrong exp, abandon
def deal_fruits12train_testing():
	result_dir = '/mnt/d/ziqi/vis/vis_color_concept/12fruit_eachtest/'
	sub_dirs = ['10ft_veg/', '12cartoon/', '12photo/', 'recycle/']
	mask_type = 'mask'
	num_groups = 25
	T1	= 0.38
	T2	= 1.0
	beta= 1.15
	### vegetables5
	folder = result_dir + sub_dirs[0]
	save_file = './results/exp6-vegetables5-ratings71-net-train_on_fruits12-mask-25.txt'
	deal_vegetable5_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)
	### fruits5
	folder = result_dir + sub_dirs[0]
	save_file = './results/exp6-fruits5-ratings71-net-train_on_fruits12-mask-25.txt'
	deal_fruits5_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)
	### fruits12-cartton
	folder = result_dir + sub_dirs[1]
	save_file = './results/exp6-fruits12_cartoon-ratings58-net-train_on_fruits12-mask-25.txt'
	deal_fruits12_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)
	### fruits12_photo
	folder = result_dir + sub_dirs[2]
	save_file = './results/exp6-fruits12_photo-ratings58-net-train_on_fruits12-mask-25.txt'
	deal_fruits12_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)
	### recyle6
	folder = result_dir + sub_dirs[3]
	save_file = './results/exp6-recycle6-ratings37-net-train_on_fruits12-mask-25.txt'
	deal_recycle6_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)


def debug():
	
	folder = '/Users/codeb/Downloads/eccv_result/othft_result/pro_313/eggplant/'
	idx = 0
	num_groups = 25
	ratings_space = 71
	T1 = 1.0
	T2 = 1.0
	beta = 1.2
	final_ratings = mapping_pipeline(folder, idx, num_groups, ratings_space, T1, T2, beta=beta)
	print(final_ratings.shape)
	print(final_ratings)

	# ratings_58 = mapping_pipeline('/Users/codeb/Downloads/eccv_result/pro313_clean/watermelon/0.txt', 25)
	# get_distance_map(100, 58)
	# get_weights_map(100)

	# import matplotlib.pyplot as plt
	# sorting_idx, sort_colors = CONFIG.get_sorted_RGB_58_to_visulize()
	# sort_ratings = ratings_58[sorting_idx]
	# fig = plt.figure(figsize=(10,4))
	# plt.bar(range(1,59), sort_ratings, color=sort_colors, width=1.0)
	# plt.savefig('mapping_watermelon0_T001.pdf')

	print()


def tune_params():

	num_groups = 25
	note = 'finetuned-mask-25'
	
	# for beta in [1.0, 0.5, 0.2, 0.7, 0.1, 0.9]:
	# 	for T in [0.1, 0.2, 0.5, 0.8, 0.9]:
	# for beta in np.linspace(1.1, 2, 10):
	# 	for T in [0.01, 0.38, 0.7, 1,1, 1.5, 2.0]:
		# for T in [0.01, 0.05, 0.1, 0.2, 0.3, 0.38]:
	for beta in np.linspace(1.1, 1.3, 5):
		for T in [1.1, 1.2]:
			for T_type in range(2):
				if T_type == 0:
					T1 = 1.0
					T2 = T
				else:
					T1 = T
					T2 = 1.0
				note2 = 'T%.2f-T%.2f-b%.2f' %(T1, T2, beta)

				save_name = 'exp1_1-ratings58-net-%s-%s.csv' % (note, note2)
				save_file = './results_tune_mapping/' + save_name
				if os.path.exists(save_file): continue
				
				result_folder = '/Users/codeb/Downloads/eccv_result/exp1_1-fullpipeline/'

				deal_fruits12_exp(result_folder, num_groups, save_file, T1, T2, beta)


def mapping_several_pixels():
	# targets: [9:12, 26:30]
	beta = 1.0
	lightness = np.load('/Users/codeb/Downloads/eccv_result/exp1_2-net-finetuned/blueberry/L_2.npy')
	probs = np.load('/Users/codeb/Downloads/eccv_result/exp1_2-net-finetuned/blueberry/pro_2.npy')
	lightness = lightness.reshape((56,56))
	probs = probs.reshape((56, 56, 313))
	for i in range(9, 12):
		for j in range(26, 30):
			L = lightness[i, j]		# float
			probs_px = probs[i, j]		# (313,)
			ab = CONFIG.get_AB_PTS_IN_HULL()
			all_lab = np.zeros((313, 3))
			all_lab[:, 0] = L
			all_lab[:,1:] = ab
			### distance (313, 58)
			LAB_58 = CONFIG.get_LAB_58()
			distance_map = np.zeros((313, 58))
			for idx, target_lab in enumerate(LAB_58):
				dis = cal_LAB_L2_distance(all_lab, target_lab)
				distance_map[:, idx] = dis
			### weights (313, 58)
			weights_map = np.zeros_like(distance_map)
			for row_id, distances in enumerate(distance_map):
				dis_z = Z_Score(distances)	# normalize by z-score
				dis_z = - dis_z				# get minus
				dis_z = dis_z * beta		# parameter to tune
				weights = softmax(dis_z)	# softmax
				weights_map[row_id] = weights
			###
			ratings58 = probs_px.dot(weights_map)
			ratings58 = normalize(ratings58)
			np.savetxt('./results/pixels/fruits12_cartoon-blueberry-2-px_%d_%d.txt'%(i,j), ratings58)


def mapping_one_pixel(T, beta):
	### param
	# T = 1.0
	# beta = 1.0
	### data
	probs_file = '../vis-tools/prob_40_25.txt'
	probs = np.loadtxt(probs_file)	## (313, )
	L = 40
	ab = CONFIG.get_AB_PTS_IN_HULL()
	all_lab = np.zeros((313, 3))
	all_lab[:, 0] = L
	all_lab[:,1:] = ab
	### probs
	if T != 1.0:
		probs = probs.reshape((1, 313))
		probs = probs_temporature_scheme(probs, T=T)
	### distance (313, 58)
	LAB_58	= CONFIG.get_LAB_58()
	distance_map = np.zeros((313, 58))
	for idx, target_lab in enumerate(LAB_58):
		dis = cal_LAB_L2_distance(all_lab, target_lab)
		distance_map[:, idx] = dis
	### weights (313, 58)
	weights_map = np.zeros_like(distance_map)
	for row_id, distances in enumerate(distance_map):
		dis_z = Z_Score(distances)	# normalize by z-score
		dis_z = - dis_z				# get minus
		dis_z = dis_z * beta		# parameter to tune
		weights = softmax(dis_z)	# softmax
		weights_map[row_id] = weights
	### 
	ratings_58 = probs.dot(weights_map)
	ratings_58 = normalize(ratings_58)
	np.savetxt('px_40_25/pixel_40_25-T%.2f-b%.2f.txt' % (T, beta), ratings_58)


def mapping_one_img():
	# img_file = './datasets/baseline_12fruit/top/watermelon/47.jpg'
	# save_file = './results/exp1_1-watermelon47-ratings58-finetuned-mask.txt'
	# save_file = './results/exp1_1-strawberry30-ratings58-finetuned-mask.txt'
	# save_file = './results/exp1_1-strawberry30-ratings58-finetuned-nomask.txt'
	# save_file = './results/exp1_2-orange_cartoon32-ratings58-finetuned-mask.txt'
	save_file = './results/exp1_2-orange_cartoon32-ratings58-finetuned-nomask.txt'

	# exp1_1_finetune_folder = '/Users/codeb/Downloads/eccv_result/exp1_1-net-finetuned/'
	exp1_2_finetune_folder = '/Users/codeb/Downloads/eccv_result/exp1_2-net-finetuned/'
	# concept = 'watermelon'
	concept = 'orange'
	folder = exp1_2_finetune_folder + concept + '/'
	idx = 32

	net_type = 'finetuned'
	mask_type = 'mask'
	num_groups = 25
	ratings_space = 58

	# with open(CONFIG.MASK_PATH_FOR_FRUITS_12, 'rb') as f:
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12_cartoon, 'rb') as f:
		all_masks = pickle.load(f)
	mask = all_masks[concept][idx]
	mask = None

	T1	= 1.0
	T2	= 1.0
	beta= 1.0

	img_ratings = mapping_pipeline(folder, idx, num_groups, ratings_space, mask, T1, T2, beta)
	np.savetxt(save_file, img_ratings)


""" EXPs for specific """

### exp 1.1 fruits12
def exp1_1():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = '/Users/codeb/Downloads/eccv_result/'
	finetune_folder = dir + 'exp1_1-net-finetuned/'
	pretrain_folder = dir + 'exp1_1-net-pretrain/'
	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	net_type = 'finetuned'
	mask_type = 'mask'
	note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	print(note)
	folder = finetune_folder if net_type=='finetuned' else pretrain_folder
	save_name = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
	# save_name = 'test2-exp1_1-fruits12-ratings58-net-%s.txt' % note			### test 1 2: remove white
	# save_name = 'test4-exp1_1-fruits12-ratings58-net-%s.txt' % note				### test 3: no minus z-score ## test4: min-max
	save_file = './results/' + save_name
	deal_fruits12_exp(folder, save_file, num_groups, mask_type, T1, T2, beta,
						sigma=None, do_dewhite=False, white_range=None, standardization='z-score')


def exp1_1_rounds(rounds):
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	# dir = '/Users/codeb/Downloads/eccv_result/'
	# finetune_folder = dir + 'exp1_1-net-finetuned/'
	# pretrain_folder = dir + 'exp1_1-net-pretrain/'
	# for net_type in ['finetuned', 'pretrain']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	net_type = 'finetuned'
	# mask_type = 'mask'
	for mask_type in ['mask', 'nomask']:
		note = '%s-%s-%d' % (net_type, mask_type, num_groups)
		if rounds: note += '-%d' % rounds
		print(note)
		folder = './temp_results/fruit_%d/' % rounds
		# folder = '/mnt/d/ziqi/vis/vis_color_concept/fruit_%d' % rounds
		# folder = finetune_folder if net_type=='finetuned' else pretrain_folder
		# save_name = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
		save_name = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
		save_file = './results/' + save_name
		deal_fruits12_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)

def exp1_1_no_pretrain():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = './temp_results/'
	params = ['bstop_ir2step5', 'cctop_ir3step6', 'cgtop_origin']
	for p in params:
		folder = dir + p + '/'
		net_type = 'nopretrain'
		mask_type = 'mask'
		note = '%s-%s-%d-%s' % (net_type, mask_type, num_groups, p)
		print(note)
		save_name = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
		save_file = './results/' + save_name
		deal_fruits12_exp(folder, save_file, num_groups, mask_type, T1, T2, beta,
							sigma=None, do_dewhite=False, white_range=None, standardization='z-score')


### exp 1.2 fruits12_cartoon
def exp1_2(save_each_img=True):
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = '/Users/codeb/Downloads/eccv_result/'
	finetune_folder = dir + 'exp1_2-net-finetuned/'
	pretrain_folder = dir + 'exp1_2-net-pretrain/'
	# for net_type in ['finetuned', 'pretrain']:
	for net_type in ['pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			print(note)
			folder = finetune_folder if net_type=='finetuned' else pretrain_folder
			save_name = 'exp1_2-fruits12_cartoon-ratings58-net-%s.txt' % note
			save_file = './results/' + save_name
			deal_fruits12_cartoon_exp(folder, save_file, num_groups, mask_type, T1, T2, beta, save_each_img)

### exp 1.3 fruits12_photo
def exp1_3():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = '/Users/codeb/Downloads/eccv_result/'
	finetune_folder = dir + 'exp1_3-net-finetuned/'
	pretrain_folder = dir + 'exp1_3-net-pretrain/'
	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask',]:
		# for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			print(note)
			folder = finetune_folder if net_type=='finetuned' else pretrain_folder
			save_name = 'exp1_3-fruits12_photo-ratings58-net-%s.txt' % note
			save_file = './results/' + save_name
			deal_fruits12_photo_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)

### exp 1.5 recycle6
def exp1_5():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = '/Users/codeb/Downloads/eccv_result/'
	finetune_folder = dir + 'exp1_5-net-finetuned/'
	pretrain_folder = dir + 'exp1_5-net-pretrain/'
	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
	# net_type = 'finetuned'
	# mask_type = 'mask'
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			print(note)
			folder = finetune_folder if net_type=='finetuned' else pretrain_folder
			# folder = '/mnt/d/ziqi/vis/vis_color_concept/recy_%d/' % rounds
			# folder = './temp_results/recy_%d/' % rounds
			save_name = 'exp1_5-recycle6-ratings37-net-%s.txt' % note
			save_file = './results/' + save_name
			deal_recycle6_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)

def exp1_5_rounds(rounds=None):
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	# dir = '/Users/codeb/Downloads/eccv_result/'
	# finetune_folder = dir + 'exp1_5-net-finetuned/'
	# pretrain_folder = dir + 'exp1_5-net-pretrain/'
	# for net_type in ['finetuned', 'pretrain']:
	net_type = 'finetuned'
	for mask_type in ['mask', 'nomask']:
		if net_type=='pretrain' and mask_type=='nomask': continue
	# mask_type = 'mask'
		note = '%s-%s-%d' % (net_type, mask_type, num_groups)
		if rounds: note += '-%d' % rounds
		print(note)
		# folder = finetune_folder if net_type=='finetuned' else pretrain_folder
		folder = '/mnt/d/ziqi/vis/vis_color_concept/recycle_%d/' % rounds
		# folder = './temp_results/recycle_%d/' % rounds
		save_name = 'exp1_5-recycle6-ratings37-net-%s.txt' % note
		save_file = './results/' + save_name
		deal_recycle6_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)

### exp 2.1 vegetables5
def exp2_1():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = '/Users/codeb/Downloads/eccv_result/'
	finetune_folder = dir + 'exp2_1-net-finetuned/'
	pretrain_folder = dir + 'exp2_1-net-pretrain/'
	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			print(note)
			folder = finetune_folder if net_type=='finetuned' else pretrain_folder
			save_name = 'exp2_1-vegetables5-ratings71-net-%s.txt' % note
			save_file = './results/' + save_name
			deal_vegetable5_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)

### exp 2.2 fruits5
def exp2_2():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	dir = '/Users/codeb/Downloads/eccv_result/'
	finetune_folder = dir + 'exp2_2-net-finetuned/'
	pretrain_folder = dir + 'exp2_2-net-pretrain/'
	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			print(note)
			folder = finetune_folder if net_type=='finetuned' else pretrain_folder
			save_name = 'exp2_2-fruits5-ratings71-net-%s.txt' % note
			save_file = './results/' + save_name
			deal_fruits5_exp(folder, save_file, num_groups, mask_type, T1, T2, beta)

### exp7 single-concept fruits12
def exp7_single_concept_fruits12():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	ratings_space = 58
	mask_type = 'mask'
	dir = '/Users/codeb/Downloads/eccv_result/fruits12-single_train/'
	save_file = './results/exp7-fruits12-ratings58-net-single_concept-%s-%d.txt' % (
						mask_type, num_groups)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12, 'rb') as f:
		all_masks = pickle.load(f)
	all_ratings = np.zeros((12 ,58))
	for idx, concept in enumerate(CONFIG.FRUITS_12):
		folder = dir + concept + '/'
		masks_map = all_masks[concept]
		ratings, per_img_ratings = get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1, T2, beta)
		all_ratings[idx] = ratings
	np.savetxt(save_file, all_ratings, delimiter=',')

### exp7 single-concept recycle6
def exp7_single_concept_recycle6():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	ratings_space = 37
	mask_type = 'mask'
	dir = '/Users/codeb/Downloads/eccv_result/recycle6-single_train/'
	save_file = './results/exp7-recycle6-ratings37-net-single_concept-%s-%d.txt' % (
						mask_type, num_groups)
	with open(CONFIG.MASK_PATH_FOR_RECYCLE_6, 'rb') as f:
		all_masks = pickle.load(f)
	all_ratings = np.zeros((6 ,37))
	for idx, concept in enumerate(CONFIG.RECYCLE_6):
		folder = dir + concept + '/'
		masks_map = all_masks[concept]
		ratings, per_img_ratings = get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1, T2, beta)
		all_ratings[idx] = ratings
	np.savetxt(save_file, all_ratings, delimiter=',')

### exp7 single-concept fruits5
def exp7_single_concept_fruits5():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	ratings_space = 71
	mask_type = 'mask'
	dir = '/Users/codeb/Downloads/eccv_result/fruits5-single_train/'
	save_file = './results/exp7-fruits5-ratings71-net-single_concept-%s-%d.txt' % (
						mask_type, num_groups)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_5, 'rb') as f:
		all_masks = pickle.load(f)
	all_ratings = np.zeros((5 ,71))
	for idx, concept in enumerate(CONFIG.FRUITS_5):
		folder = dir + concept + '/'
		masks_map = all_masks[concept]
		ratings, per_img_ratings = get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1, T2, beta)
		all_ratings[idx] = ratings
	np.savetxt(save_file, all_ratings, delimiter=',')

### exp7 single-concept fruits5
def exp7_single_concept_vegetables5():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	ratings_space = 71
	mask_type = 'mask'
	dir = '/Users/codeb/Downloads/eccv_result/vegetables5-single_train/'
	save_file = './results/exp7-vegetables5-ratings71-net-single_concept-%s-%d.txt' % (
						mask_type, num_groups)
	with open(CONFIG.MASK_PATH_FOR_VEGETABLES_5, 'rb') as f:
		all_masks = pickle.load(f)
	all_ratings = np.zeros((5 ,71))
	for idx, concept in enumerate(CONFIG.VEGETABLES_5):
		folder = dir + concept + '/'
		masks_map = all_masks[concept]
		ratings, per_img_ratings = get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1, T2, beta)
		all_ratings[idx] = ratings
	np.savetxt(save_file, all_ratings, delimiter=',')

### exp 8
def naive_baseline_all():
	T1	= 1.0
	T2	= 1.0
	beta= 1.0
	num_groups = 25
	# for mask_type in ['mask', 'nomask']:
	for mask_type in ['nomask']:
		dir = '/Users/codeb/Downloads/eccv_result/soft_encodings/'
		exp6_fucs = {
			# 'fruits12':			deal_fruits12_exp,
			# 'fruits12_cartoon':	deal_fruits12_cartoon_exp,
			# 'fruits12_photo':	deal_fruits12_photo_exp,
			'recycle6':			deal_recycle6_exp,	
			# 'vegetables5':		deal_vegetable5_exp,
			# 'fruits5':			deal_fruits5_exp,
		}
		for dsn in exp6_fucs.keys():
			folder = dir + dsn + '/'
			save_file = './results/exp8-%s-ratings-GT-soft-encoding-%s.txt' % (dsn, mask_type)
			exp6_fucs[dsn](folder, save_file, num_groups, mask_type, T1, T2, beta)

""""""
### 
def get_all_one_in_fruits12(concept='blueberry'):
	T1			= 1.0
	T2			= 1.0
	beta		= 1.0
	mask_type	= 'mask'
	num_groups	= 25
	figs_num	= 50
	ratings_space = 58
	folder = '/Users/codeb/Downloads/eccv_result/exp1_1-net-finetuned/' + concept + '/'
	save_folder = './results/fruits12/' + concept + '/'
	if not os.path.exists(save_folder): os.mkdir(save_folder)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12, 'rb') as f:
		all_masks = pickle.load(f)
	masks_map = all_masks[concept]
	final_ratings, per_img_ratings = get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1, T2, beta, figs_num)
	for i in range(per_img_ratings.shape[0]):
		per_img_save_file = save_folder + '/%d-%s.txt' % (i, mask_type)
		np.savetxt(per_img_save_file, per_img_ratings[i])

### 
def get_all_one_in_fruits12_cartoon(concept='blueberry'):
	T1			= 1.0
	T2			= 1.0
	beta		= 1.0
	mask_type	= 'mask'
	num_groups	= 25
	figs_num	= 50
	ratings_space = 58
	folder = '/Users/codeb/Downloads/eccv_result/exp1_2-net-finetuned/' + concept + '/'
	save_folder = './results/fruits12_cartoon/' + concept + '/'
	if not os.path.exists(save_folder): os.makedirs(save_folder)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12_cartoon, 'rb') as f:
		all_masks = pickle.load(f)
	masks_map = all_masks[concept]
	final_ratings, per_img_ratings = get_ratings_for_one_concept(folder, num_groups, ratings_space, masks_map, T1, T2, beta, figs_num)
	for i in range(per_img_ratings.shape[0]):
		per_img_save_file = save_folder + '/%d-%s.txt' % (i, mask_type)
		np.savetxt(per_img_save_file, per_img_ratings[i])


if __name__=='__main__':
	
	# debug()
	# tune_params()
	# mapping_one_img()
	# deal_fruits12train_testing()
	# temp_single_concept()

	# get_all_one_in_fruits12('honeydew')
	# get_all_one_in_fruits12_cartoon(concept='blueberry')
	# mapping_several_pixels()

	# exp1_1_no_pretrain()
	# exp1_1()
	# exp1_2(False)
	# exp1_3()
	# exp1_5()
	# exp2_1()
	# exp2_2()
	# naive_baseline_all()

	# exp1_1_rounds(8000)
	exp1_5_rounds(12000)

	# exp7_single_concept()
	# exp7_single_concept_recycle6()
	# exp7_single_concept_fruits5()
	# exp7_single_concept_vegetables5()

	# for T in np.linspace(0.1, 1, 10):
	# 	for beta in np.linspace(0.1, 1, 10):
	# for T in [1.0]:
	# 	for beta in np.linspace(1.1, 2, 10):
	# 		mapping_one_pixel(T, beta)
	
	# # net_type = 'finetuned'
	# net_type = 'pretrain'
	# mask_type = 'mask'
	# # mask_type = 'nomask'
	# num_groups = 25
	# # num_groups = 50
	# # num_groups = 100
	# note = '%s-%s-%d' % (net_type, mask_type, num_groups)

	# T1	= 0.38
	# T2	= 1.0
	# beta= 1.15
	
	print('done')



# %%
#%%
