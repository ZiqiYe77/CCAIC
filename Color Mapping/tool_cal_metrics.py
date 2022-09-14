#-*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
# from pyemd import emd
from scipy.stats import pearsonr, entropy

from settings.config import CONFIG

""" tools to read results """

def normalize(l):
	return l/np.sum(l)

def read_gt(gt_file, concepts):
	gt_ratings = np.loadtxt(gt_file, delimiter=',')
	for i in range(len(gt_ratings)):
		gt_ratings[i] = normalize(gt_ratings[i])
	gt_ratings_per_concept = dict(zip(concepts, gt_ratings))
	return gt_ratings_per_concept

def read_baseline(baseline_csv, baseline_col, concepts):
	df = pd.read_csv(baseline_csv)
	baseline_ratings_per_concept = {}
	for c in concepts:
		ratings = df[df['Concept']==c][baseline_col]
		ratings = normalize(np.array(ratings))
		baseline_ratings_per_concept[c] = ratings
	return baseline_ratings_per_concept

def read_naive(folder, concepts):
	naive_ratins_per_concept = {}
	for c in concepts:
		ratings = np.loadtxt(folder + c + '.txt')
		ratings = normalize(ratings)
		naive_ratins_per_concept[c] = ratings
	return naive_ratins_per_concept

### reduce results around a=0 and b=0
def reduce_00(result_ratings, sigma, return_0_color_idx=False):
	num_concepts, ratings_space = result_ratings.shape
	if ratings_space == 58: lab_colors = CONFIG.get_LAB_58()
	if ratings_space == 71: lab_colors = CONFIG.get_LAB_71()
	if ratings_space == 37: lab_colors = CONFIG.get_LAB_37()
	if ratings_space == 20: lab_colors = CONFIG.get_LAB_Tableau_20()
	if ratings_space == 30: lab_colors = CONFIG.get_LAB_EXP_30()
	if ratings_space == 27: lab_colors = CONFIG.get_LAB_EXP_27()
	if ratings_space == 28: lab_colors = CONFIG.get_LAB_EXP_28()
	# if ratings_space == 65: lab_colors = CONFIG.get_LAB_UWE_65()
	if ratings_space == 65: lab_colors = CONFIG.get_LAB_UWE_66()	##TODO temp
	# if ratings_space == 66: lab_colors = CONFIG.get_LAB_UWE_66()
	idx_00 = []
	for lab_idx, lab in enumerate(lab_colors):
		if abs(lab[1])<1 and abs(lab[2])<1: idx_00.append(lab_idx)
		# if lab[1]==0 and lab[2]==0: idx_00.append(lab_idx)
	reduced_result_ratings = result_ratings.copy()
	for i in range(num_concepts):
		for idx in idx_00:
			reduced_result_ratings[i][idx] = result_ratings[i][idx] * sigma
	if return_0_color_idx:
		return reduced_result_ratings, idx_00
	return reduced_result_ratings

def read_net_results(result_file, concepts, reduce_00_sigma=None):
	result_ratings = np.loadtxt(result_file, delimiter=',')
	if reduce_00_sigma is not None:
		result_ratings = reduce_00(result_ratings, reduce_00_sigma)
	for i in range(len(result_ratings)):
		result_ratings[i] = normalize(result_ratings[i])
	result_ratings_per_concept = dict(zip(concepts, result_ratings))
	return result_ratings_per_concept


""" metircs 
	All imput his1 and his2 should be normalized
"""

def cal_emd(his1, his2):
	from pyemd import emd
	length = len(his1)
	distance_matrix = CONFIG.get_lab_color_L2_distance_matrix(length)
	emd_score = emd(his1, his2, distance_matrix)
	return emd_score

def cal_entropy_distance(his1, his2):
	entro_dis = abs(entropy(his1) - entropy(his2))
	return entro_dis

def cal_sqErr(his1, his2):
	return np.sum(pow(his1 - his2, 2))

def cal_pearsonr(his1, his2):
	return pearsonr(his1, his2)

### from Mukherjee_2022, maybe equals L1 distance
def cal_total_variation(his1, his2):
	return np.sum(abs(his1-his2)) / 2

### ref: https://hiweller.github.io/colordistance/color-metrics.html
### somehow bias EMD with chi_square
def cal_weighted_pairs(his1, his2):
	pass

def cal_chi_square(his1, his2):
	return np.sum((his1 - his2)**2 / (his1 + his2))

def cal_chi_square_v2(his1, his2):
	pass

def cal_Jerry_divergence(his1, his2):
	ms = (his1 + his2) / 2
	h1 = his1 * np.log(his1/ms)
	h2 = his2 * np.log(his2/ms)
	h1[np.isnan(h1)] = 0
	h2[np.isnan(h2)] = 0
	Dj = np.sum(h1 + h2)
	return Dj

def cross_entropy(pred, target):
	delta = 1e-10
	return -np.sum(target * np.log(pred+delta))
	# return -np.sum(target * np.log(pred))

### calculate all metrics defined
def cal_all_metrics(gt_ratings, result_ratings, concepts):
	emd_entro_results = []
	entro_dis_results = []
	emd_results = []
	dj_results = []
	chi2_results = []
	tv_results = []
	##TODO
	cro_en_results = []
	# wp_results = []
	sqErr_results = []
	Corr_results = []
	pVal_results = []
	# cal metrics
	for c in concepts:
		his1 = gt_ratings[c]
		his2 = result_ratings[c]
		### EMD & entropy distance
		emd_score = cal_emd(his1, his2)
		entro_dis = cal_entropy_distance(his1, his2) 
		emd_entro = entro_dis * emd_score
		emd_entro_results.append(emd_entro)
		entro_dis_results.append(entro_dis)
		emd_results.append(emd_score)
		### cross-entropy
		cro_en_results.append(cross_entropy(his2, his1))
		### TV
		tv_results.append(cal_total_variation(his1, his2))
		### Jeffrey Divergence
		dj_results.append(cal_Jerry_divergence(his1, his2))
		### chi-square
		chi2_results.append(cal_chi_square(his1, his2))
		### L2
		sqErr_results.append(cal_sqErr(his1, his2))
		### pearson corr
		Corr, pVal = cal_pearsonr(his1, his2)
		Corr_results.append(Corr)
		pVal_results.append(pVal)
	# average
	# concepts.append('average')
	emd_entro_results.append(	np.mean(emd_entro_results))
	entro_dis_results.append(	np.mean(entro_dis_results))
	emd_results.append(			np.mean(emd_results))
	dj_results.append(			np.mean(dj_results))
	chi2_results.append(		np.mean(chi2_results))
	tv_results.append(			np.mean(tv_results))
	cro_en_results.append(		np.mean(cro_en_results))
	sqErr_results.append(		np.mean(sqErr_results))
	Corr_results.append(		np.mean(Corr_results))
	pVal_results.append(		np.mean(pVal_results))
	# save to df
	metrics_df = pd.DataFrame({
		'Concept':  	concepts + ['average'],
		'EMD_entropy':	emd_entro_results,
		'Entro_dis':	entro_dis_results,
		'EMD':      	emd_results,
		'Cross_entro':	cro_en_results,
		'DJeffrey':		dj_results,
		'CHI-2':		chi2_results,
		'TV':			tv_results,
		'sqErr':    	sqErr_results,
		'Corr':     	Corr_results,
		'pVal':     	pVal_results
	})
	return metrics_df


"""
	EXPs
"""

""" expd """
def cal_exp1_1_naive_metrics_old():
	dir = './results/'
	result_folder = '/mnt/d/ziqi/vis/vis_color_concept/navie_result/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	naive_ratings = read_naive(result_folder, concepts)

	metrics_df = cal_all_metrics(gt_ratings, naive_ratings, concepts)
	metrics_df.to_csv(dir + 'exp1_1-naive-nomask-metrics.csv', index=False)


""" basline """

# fruits12 - baseline
def cal_exp1_1_baseline_metrics():
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	baseline_file = 'fruits_baseline.csv'
	baseline_col = 'Top50_Sector+Category'
	baseline_ratings = read_baseline(dir+baseline_file, baseline_col, concepts)

	metrics_df = cal_all_metrics(gt_ratings, baseline_ratings, concepts)
	metrics_df.to_csv(dir + 'exp1_1-fruits12-metrics-baseline.csv', index=False)

# fruits12_cartoon - baseline
def cal_exp1_2_baseline_metrics():
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	baseline_file = 'fruits_baseline.csv'
	baseline_col = 'Cartoon50_Sector+Category'
	baseline_ratings = read_baseline(dir+baseline_file, baseline_col, concepts)

	metrics_df = cal_all_metrics(gt_ratings, baseline_ratings, concepts)
	metrics_df.to_csv(dir + 'exp1_2-fruits12_cartoon-metrics-baseline.csv', index=False)

# fruits12_photo - baseline
def cal_exp1_3_baseline_metrics():
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	baseline_file = 'fruits_baseline.csv'
	baseline_col = 'Photo50_Sector+Category'
	baseline_ratings = read_baseline(dir+baseline_file, baseline_col, concepts)

	metrics_df = cal_all_metrics(gt_ratings, baseline_ratings, concepts)
	metrics_df.to_csv(dir + 'exp1_3-fruits12_photo-metrics-baseline.csv', index=False)

# recycle6 - baseline
def cal_exp1_5_baseline_metrics():
	dir = './results/'
	concepts = CONFIG.RECYCLE_6
	gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()

	baseline_folder = '/Users/codeb/Downloads/eccv_result/get_baseline_rt/recycle6/'
	baseline_ratings = {}
	for c in concepts:
		ratings37 = np.load(baseline_folder + c + '.npy')
		baseline_ratings[c] = normalize(ratings37)

	metrics_df = cal_all_metrics(gt_ratings, baseline_ratings, concepts)
	metrics_df.to_csv(dir + 'exp1_5-recycle6-metrics-baseline.csv', index=False)

# vegetables5 - baseline
def cal_exp2_1_baseline_metrics():
	dir = './results/'
	concepts = CONFIG.VEGETABLES_5
	gt_ratings = CONFIG.get_VEGETABLES_5_RATING_GT_71()

	baseline_folder = '/Users/codeb/Downloads/eccv_result/get_baseline_rt/vegetable5/'
	baseline_ratings = {}
	for c in concepts:
		ratings71 = np.load(baseline_folder + c + '.npy')
		baseline_ratings[c] = normalize(ratings71)
	
	metrics_df = cal_all_metrics(gt_ratings, baseline_ratings, concepts)
	metrics_df.to_csv(dir + 'exp2_1-vegetables5-metrics-baseline.csv', index=False)

# fruits5 - baseline
def cal_exp2_2_baseline_metrics():
	dir = './results/'
	concepts = CONFIG.FRUITS_5
	gt_ratings = CONFIG.get_FRUITS_5_RATING_GT_71()

	baseline_folder = '/Users/codeb/Downloads/eccv_result/get_baseline_rt/fruits5/'
	baseline_ratings = {}
	for c in concepts:
		ratings71 = np.load(baseline_folder + c + '.npy')
		baseline_ratings[c] = normalize(ratings71)
	
	metrics_df = cal_all_metrics(gt_ratings, baseline_ratings, concepts)
	metrics_df.to_csv(dir + 'exp2_2-fruits5-metrics-baseline.csv', index=False)

### re-do all basline
def cal_baselines_metirc(dataset):
	dir = '/Users/codeb/Downloads/eccv_result/'
	results_dir = dir + 'baseline_%s_result/' % dataset
	if dataset == 'fruits12':
		concepts	= CONFIG.FRUITS_12
		gt_ratings	= CONFIG.get_FRUITS_12_RATING_GT_58()
	elif dataset == 'recycle6':
		concepts	= CONFIG.RECYCLE_6
		gt_ratings	= CONFIG.get_RECYCLE_6_RATING_GT_37()
	elif dataset == 'fruits5':
		concepts	= CONFIG.FRUITS_5
		gt_ratings	= CONFIG.get_FRUITS_5_RATING_GT_71()
	elif dataset == 'vegetables5':
		concepts	= CONFIG.VEGETABLES_5
		gt_ratings	= CONFIG.get_VEGETABLES_5_RATING_GT_71()
	
	beseline_ratings = {}
	for c in concepts:
		ratings = np.load(results_dir + '%s.npy' % c)
		beseline_ratings[c] = normalize(ratings)
	
	metrics_df = cal_all_metrics(gt_ratings, beseline_ratings, concepts)
	metrics_df.to_csv('./results/real_baseline-%s-metrics.csv' % dataset, index=False)


""" net """


# fruits12 - net
def cal_exp1_1_result_metrics(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	# net_type = 'finetuned'
	net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	result_file = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
	# result_file = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
	result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
	
	if reduce_00_sigma:	note += '-%.1f'	% reduce_00_sigma
	
	save_name = 'exp1_1-fruits12-metrics-net-%s.csv' % note
	metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)


def cal_exp1_1_result_metrics_nopretrain():
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	net_type = 'nopretrain'
	mask_type = 'mask'
	num_groups = 25

	params = ['bstop_ir2step5', 'cctop_ir3step6', 'cgtop_origin']
	for p in params:
		note = '%s-%s-%d-%s' % (net_type, mask_type, num_groups, p)

		result_file = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
		result_ratings = read_net_results(dir + result_file, concepts)
		
		save_name = 'exp1_1-fruits12-metrics-net-%s.csv' % note
		metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
		metrics_df.to_csv(dir + save_name, index=False)


def cal_exp1_1_result_metrics_rounds(rounds=None, reduce_00_sigma=None):
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	net_type = 'finetuned'
	# net_type = 'pretrain'
	# mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	for mask_type in ['mask', 'nomask']:
		note = '%s-%s-%d' % (net_type, mask_type, num_groups)
		if rounds: note += '-%d' % rounds
		
		# result_file = 'test2-exp1_1-fruits12-ratings58-net-%s.txt' % note
		result_file = 'exp1_1-fruits12-ratings58-net-%s.txt' % note
		result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
		
		if reduce_00_sigma:	note += '-%.1f'	% reduce_00_sigma

		save_name = 'exp1_1-fruits12-metrics-net-%s.csv' % note
		# save_name = 'test2-exp1_1-fruits12-metrics-net-%s.csv' % note
		metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
		metrics_df.to_csv(dir + save_name, index=False)


# fruits12_cartoon - net
def cal_exp1_2_result_metrics():
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	# net_type = 'finetuned'
	net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
	
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			
			result_file = 'exp1_2-fruits12_cartoon-ratings58-net-%s.txt' % note
			result_ratings = read_net_results(dir + result_file, concepts)
			
			save_name = 'exp1_2-fruits12_cartoon-metrics-net-%s.csv' % note
			metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
			metrics_df.to_csv(dir + save_name, index=False)

# fruits12_photo - net
def cal_exp1_3_result_metrics():
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	# net_type = 'finetuned'
	net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue
	
			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			
			result_file = 'exp1_3-fruits12_photo-ratings58-net-%s.txt' % note
			result_ratings = read_net_results(dir + result_file, concepts)
			
			save_name = 'exp1_3-fruits12_photo-metrics-net-%s.csv' % note
			metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
			metrics_df.to_csv(dir + save_name, index=False)

# recycle6 - net
def cal_exp1_5_result_metrics(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.RECYCLE_6
	gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()
	
	# net_type = 'finetuned'
	net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	result_file = 'exp1_5-recycle6-ratings37-net-%s.txt' % note
	result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
	
	if reduce_00_sigma:	note += '-%.2f'	% reduce_00_sigma

	save_name = 'exp1_5-recycle6-metrics-net-%s.csv' % note
	metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)


def cal_exp1_5_result_metrics_rounds(reduce_00_sigma, rounds=None):
	dir = './results/'
	concepts = CONFIG.RECYCLE_6
	gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()
	
	net_type = 'finetuned'
	# net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	for mask_type in ['mask', 'nomask']:

		note = '%s-%s-%d' % (net_type, mask_type, num_groups)
		if rounds: note += '-%d'	% rounds
		
		result_file = 'exp1_5-recycle6-ratings37-net-%s.txt' % note
		result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
		
		if reduce_00_sigma:	note += '-%.1f'	% reduce_00_sigma

		save_name = 'exp1_5-recycle6-metrics-net-%s.csv' % note
		metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
		metrics_df.to_csv(dir + save_name, index=False)


# vegetables5 - net
def cal_exp2_1_result_metrics(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.VEGETABLES_5
	gt_ratings = CONFIG.get_VEGETABLES_5_RATING_GT_71()

	net_type = 'finetuned'
	# net_type = 'pretrain'
	# mask_type = 'mask'
	mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue

			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			
			result_file = 'exp2_1-vegetables5-ratings71-net-%s.txt' % note
			result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
			
			if reduce_00_sigma:	note += '-%.1f'	% reduce_00_sigma

			save_name = 'exp2_1-vegetables5-metrics-net-%s.csv' % note
			metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
			metrics_df.to_csv(dir + save_name, index=False)

# fruits5 - net
def cal_exp2_2_result_metrics(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.FRUITS_5
	gt_ratings = CONFIG.get_FRUITS_5_RATING_GT_71()

	# net_type = 'finetuned'
	# net_type = 'pretrain'
	# mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	for net_type in ['finetuned', 'pretrain']:
		for mask_type in ['mask', 'nomask']:
			if net_type=='pretrain' and mask_type=='nomask': continue

			note = '%s-%s-%d' % (net_type, mask_type, num_groups)
			
			result_file = 'exp2_2-fruits5-ratings71-net-%s.txt' % note
			result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
			
			if reduce_00_sigma:	note += '-%.1f'	% reduce_00_sigma

			save_name = 'exp2_2-fruits5-metrics-net-%s.csv' % note
			metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
			metrics_df.to_csv(dir + save_name, index=False)

""" exp 6 """

## wrong
def deal_fruits12train_testing():
	dir = './results/'
	sub_dirs = ['10ft_veg/', '12cartoon/', '12photo/', 'recycle/']
	mask_type = 'mask'
	num_groups = 25
	### vegetables5
	result_file = './exp6-vegetables5-ratings71-net-train_on_fruits12-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, CONFIG.VEGETABLES_5)
	metrics_df = cal_all_metrics(CONFIG.get_VEGETABLES_5_RATING_GT_71(), result_ratings, CONFIG.VEGETABLES_5)
	metrics_df.to_csv(dir + 'exp6-vegetables5-metrics-net-train_on_fruits12-mask-25.csv')
	### fruits5
	result_file = './exp6-fruits5-ratings71-net-train_on_fruits12-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, CONFIG.FRUITS_5)
	metrics_df = cal_all_metrics(CONFIG.get_FRUITS_5_RATING_GT_71(), result_ratings, CONFIG.FRUITS_5)
	metrics_df.to_csv(dir + 'exp6-fruits5-metrics-net-train_on_fruits12-mask-25.csv')
	### fruits12_cartoon
	result_file = './exp6-fruits12_cartoon-ratings58-net-train_on_fruits12-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, CONFIG.FRUITS_12)
	metrics_df = cal_all_metrics(CONFIG.get_FRUITS_12_RATING_GT_58(), result_ratings, CONFIG.FRUITS_12)
	metrics_df.to_csv(dir + 'exp6-fruits12_cartoon-metrics-net-train_on_fruits12-mask-25.csv')
	### fruits12_photo
	result_file = './exp6-fruits12_photo-ratings58-net-train_on_fruits12-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, CONFIG.FRUITS_12)
	metrics_df = cal_all_metrics(CONFIG.get_FRUITS_12_RATING_GT_58(), result_ratings, CONFIG.FRUITS_12)
	metrics_df.to_csv(dir + 'exp6-fruits12_photo-metrics-net-train_on_fruits12-mask-25.csv')
	### recyle6
	result_file = './exp6-recycle6-ratings37-net-train_on_fruits12-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, CONFIG.RECYCLE_6)
	metrics_df = cal_all_metrics(CONFIG.get_RECYCLE_6_RATING_GT_37(), result_ratings, CONFIG.RECYCLE_6)
	metrics_df.to_csv(dir + 'exp6-recycle6-metrics-net-train_on_fruits12-mask-25.csv')

## baseline
def deal_baseline_fruits12train_testing():
	save_dir = './results/'
	results_dir = './temp_results/exp6-baseline-fruits12trained/'
	result_settings = {
		'fruits12':			('baseline_test12fruit.npy', CONFIG.FRUITS_12,	CONFIG.get_FRUITS_12_RATING_GT_58()),
		# 'fruits12_cartoon':	('cartoon.npy',			CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58()),
		# 'fruits12_photo':	('photo.npy',			CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58()),
		'recycle6':			('recycle.npy',			CONFIG.RECYCLE_6,		CONFIG.get_RECYCLE_6_RATING_GT_37()),
		'vegetables5':		('5_veg.npy',			CONFIG.VEGETABLES_5,	CONFIG.get_VEGETABLES_5_RATING_GT_71()),
		'fruits5':			('5other_fruit.npy',	CONFIG.FRUITS_5,		CONFIG.get_FRUITS_5_RATING_GT_71()),
	}
	for k in result_settings.keys():
		concepts = result_settings[k][1]
		gt_ratings = result_settings[k][2]
		result_ratings = np.load(results_dir + result_settings[k][0])
		for i in range(len(concepts)):
			result_ratings[i] = normalize(result_ratings[i])
		
		result_ratings = dict(zip(concepts, result_ratings))		### TODO
		metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
		metrics_df.to_csv(save_dir + 'exp6-%s-metrics-baseline-fruits12trained.csv' % k, index=False)


""" exp 7 """
def temp_single_concept():
	concept = ['mango']
	ratings1 = {'mango': normalize(np.loadtxt('./results/exp7-mango-tmp1.txt'))}
	ratings2 = {'mango': normalize(np.loadtxt('./results/exp7-mango-tmp2.txt'))}
	m1 = cal_all_metrics(CONFIG.get_FRUITS_12_RATING_GT_58(), ratings1, concept)
	m2 = cal_all_metrics(CONFIG.get_FRUITS_12_RATING_GT_58(), ratings2, concept)
	print(m1)
	print(m2)

def exp7_single_concept_fruits12(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	
	net_type = 'finetuned'
	# net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	# note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	result_file = 'exp7-fruits12-ratings58-net-single_concept-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
	
	note = ''
	if reduce_00_sigma:	note = '-%.1f'	% reduce_00_sigma
	save_name = 'exp7-fruits12-metrics-net-single_concept-mask-25%s.csv' % note
	metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)

def exp7_single_concept_recycle6(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.RECYCLE_6
	gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()
	
	net_type = 'finetuned'
	# net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	# note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	result_file = 'exp7-recycle6-ratings37-net-single_concept-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
	
	note = ''
	if reduce_00_sigma:	note = '-%.2f'	% reduce_00_sigma
	save_name = 'exp7-recycle6-metrics-net-single_concept-mask-25%s.csv' % note
	metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)

def exp7_single_concept_fruits5(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.FRUITS_5
	gt_ratings = CONFIG.get_FRUITS_5_RATING_GT_71()
	
	net_type = 'finetuned'
	# net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	# note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	note = ''
	result_file = 'exp7-fruits5-ratings71-net-single_concept-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
	
	if reduce_00_sigma:	note = '-%.1f'	% reduce_00_sigma
	save_name = 'exp7-fruits5-metrics-net-single_concept-mask-25%s.csv' % note
	metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)

def exp7_single_concept_vegetables5(reduce_00_sigma):
	dir = './results/'
	concepts = CONFIG.VEGETABLES_5
	gt_ratings = CONFIG.get_VEGETABLES_5_RATING_GT_71()
	
	net_type = 'finetuned'
	# net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25
	# num_groups = 50
	# num_groups = 100

	# for net_type in ['finetuned', 'pretrain']:
	# 	for mask_type in ['mask', 'nomask']:
	# 		if net_type=='pretrain' and mask_type=='nomask': continue
	
	# note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	note = ''
	result_file = 'exp7-vegetables5-ratings71-net-single_concept-mask-25.txt'
	result_ratings = read_net_results(dir + result_file, concepts, reduce_00_sigma=reduce_00_sigma)
	
	if reduce_00_sigma:	note = '-%.1f'	% reduce_00_sigma
	save_name = 'exp7-vegetables5-metrics-net-single_concept-mask-25%s.csv' % note
	metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)


""" exp 8 naive baseline """

### exp 8
def naive_baseline_all():
	
	exp8_settings = {
		'fruits12':			(CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58()),
		# 'fruits12_cartoon':	(CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58()),
		# 'fruits12_photo':	(CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58()),
		'recycle6':			(CONFIG.RECYCLE_6,		CONFIG.get_RECYCLE_6_RATING_GT_37()),
		'vegetables5':		(CONFIG.VEGETABLES_5,	CONFIG.get_VEGETABLES_5_RATING_GT_71()),
		'fruits5':			(CONFIG.FRUITS_5,		CONFIG.get_FRUITS_5_RATING_GT_71()),
	}
	for mask_type in ['mask', 'nomask']:
		for dsn in exp8_settings.keys():
			concepts	= exp8_settings[dsn][0]
			gt_ratings	= exp8_settings[dsn][1]
			ratings_file = './results/exp8-%s-ratings-GT-soft-encoding-%s.txt' % (dsn, mask_type)
			result_ratings = read_net_results(ratings_file, concepts)		### TODO reduce_00_sigma
			save_file = './results/exp8-%s-metrics-GT-soft-encoding-%s.csv' % (dsn, mask_type)
			metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
			metrics_df.to_csv(save_file, index=False)


""" tune mapping """

def tune_mapping():
	concepts = CONFIG.FRUITS_12
	gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	dir = './results_tune_mapping/'
	note = 'finetuned-mask-25'
	for beta in [1.0, 0.5, 0.2, 0.7, 0.1, 0.9, 
				 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
				 1.15, 1.25]:
		for T in [0.1, 0.2, 0.5, 0.8, 0.9, 0.38, 1.0, 1.5, 0.01, 0.05, 0.3, 1.5, 2.0]:
	# for beta in [1.4]:
	# 	for T in [1.0]:
			for T_type in range(2):
				if T_type == 0:
					T1 = 1.0
					T2 = T
				else:
					T1 = T
					T2 = 1.0
				note2 = 'T%.2f-T%.2f-b%.2f' %(T1, T2, beta)
				result_name = 'exp1_1-ratings58-net-%s-%s.csv' % (note, note2)
				result_file = dir + result_name
				if not os.path.exists(result_file): continue
				result_ratings = read_net_results(result_file, concepts)

				save_name = 'tuning-%s-metrics.csv' % note2
				metrics_df = cal_all_metrics(gt_ratings, result_ratings, concepts)
				metrics_df.to_csv(dir + save_name, index=False)



def debug():
	dir = './results/'
	concepts = CONFIG.RECYCLE_6
	gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()
	
	# net_type = 'finetuned'
	net_type = 'pretrain'
	mask_type = 'mask'
	# mask_type = 'nomask'
	num_groups = 25

	note = '%s-%s-%d' % (net_type, mask_type, num_groups)
	
	result_file = 'exp1_5-recycle6-ratings37-net-%s.txt' % note
	result_ratings = read_net_results(dir + result_file, concepts)
	
	save_name = 'temp-recycles-%s.csv' % note
	metrics_df = cal_emd_temp(gt_ratings, result_ratings, concepts)
	metrics_df.to_csv(dir + save_name, index=False)


if __name__ == '__main__':
	# cal_naive_metrics()

	# abandon
	# cal_exp1_1_baseline_metrics()
	# cal_exp1_2_baseline_metrics()
	# cal_exp1_3_baseline_metrics()
	# cal_exp1_5_baseline_metrics()
	# cal_exp2_1_baseline_metrics()
	# cal_exp2_2_baseline_metrics()
	
	### baseline
	# for dataset in ['fruits12', 'recycle6', 'fruits5', 'vegetables5']:
	# 	cal_baselines_metirc(dataset)
	# cal_baselines_metirc('fruits12')
	
	reduce_00_sigma = 0.6
	# cal_exp1_1_result_metrics(reduce_00_sigma=0.3)
	# cal_exp1_2_result_metrics()
	# cal_exp1_3_result_metrics()
	# cal_exp1_5_result_metrics(reduce_00_sigma=0.75)
	# cal_exp2_1_result_metrics(reduce_00_sigma=None)
	# cal_exp2_2_result_metrics(reduce_00_sigma=None)

	cal_exp1_1_result_metrics_rounds(rounds=8000, reduce_00_sigma=None)
	# cal_exp1_5_result_metrics_rounds(rounds=10000, reduce_00_sigma=None)
	
	# cal_exp1_1_result_metrics_nopretrain()
	
	# exp7_single_concept_fruits12(reduce_00_sigma)
	# exp7_single_concept_recycle6(reduce_00_sigma=0.75)
	# exp7_single_concept_fruits5(reduce_00_sigma)
	# exp7_single_concept_vegetables5(reduce_00_sigma)

	# deal_baseline_fruits12train_testing()

	# naive_baseline_all()


	# tune_mapping()
	# debug()

	# temp_single_concept()
#%%
# %%

# %%
