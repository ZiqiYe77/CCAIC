import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from settings.config import CONFIG
from tool_cal_metrics import read_net_results, read_baseline, normalize


"""
	Tools
"""
def fig_3_in_1(concept, net_metrics, baseline_metrics,
				gt_ratings, net_ratings, baseline_ratings, 
				save_file, ratings_space=58, box_pos=[0.85, 0.3], txt=True):

	### sorting
	if ratings_space==58:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_58_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[3] = np.array([0.3]*3)
	elif ratings_space==71:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_71_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[2] = np.array([0.3]*3)
	elif ratings_space==37:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_37_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[20] = np.array([0.3]*3)
	gt_ratings			= gt_ratings[sorting_idx]
	net_ratings			= net_ratings[sorting_idx]
	baseline_ratings	= baseline_ratings[sorting_idx]

	max_y_1 = np.max(gt_ratings)
	max_y_2 = np.max(net_ratings)
	max_y_3 = np.max(baseline_ratings)
	max_y = np.max([max_y_1, max_y_2, max_y_3])

	if txt:
		network_txt =	\
						'Corr:    %.3f\n'		% net_metrics['Corr']	+ \
						'TV:       %.4f\n'		% net_metrics['TV']		+ \
						'EMD:    %.3f\n'		% net_metrics['EMD']		+ \
						'ED:       %.3f'		% net_metrics['Entro_dis']		#+ \
						# 'EMD*entro:  %.4f\n'	% net_metrics['EMD_entropy']	+ \
						# 'Cross_entro: %.4f\n' % net_metrics['Cross_entro']		+ \
						# 'DJeffrey:      %.4f\n'	% net_metrics['DJeffrey']		+ \
						# 'CHI-2:          %.4f\n'	% net_metrics['CHI-2']		+ \
						# 'sqErr:          %.4f'	% net_metrics['sqErr']	#+ \
						# 'pVal:           %.4f'	% net_metrics['pVal']
		
		baseline_txt = 	\
						'Corr:    %.3f\n'		% baseline_metrics['Corr']	+ \
						'TV:       %.4f\n'		% baseline_metrics['TV']	+ \
						'EMD:    %.3f\n'		% baseline_metrics['EMD']	+ \
						'ED:       %.3f'		% baseline_metrics['Entro_dis']	#+ \
						# 'EMD*entro:  %.4f\n'	% baseline_metrics['EMD_entropy']	+ \
						# 'Cross_entro: %.4f\n' % baseline_metrics['Cross_entro']		+ \
						# 'DJeffrey:      %.4f\n'	% baseline_metrics['DJeffrey']	+ \
						# 'CHI-2:          %.4f\n'	% baseline_metrics['CHI-2']	+ \
						# 'sqErr:          %.4f'	% baseline_metrics['sqErr']#	+ \
						# 'pVal:           %.4f'	% baseline_metrics['pVal']

	plt.figure(figsize=(8,10))
	length = len(gt_ratings)
	box = {'facecolor':'0.9', 'edgecolor':'k', 'boxstyle':'round'}
	bx, by = box_pos
	txtsize = 10

	plt.subplot(3, 1, 1)
	ax = plt.gca() 
	plt.bar(range(1,length+1), net_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	# plt.text(45, box_y, network_txt,  bbox=box)
	if txt: plt.text(bx, by, network_txt, bbox=box, transform=ax.transAxes, fontsize=txtsize)
	plt.title('Ours (self-supervised colorization)', fontsize=15, y=-0.3)
	# plt.savefig('color_distri_plots/' + dn + '_network.pdf')
	# plt.clf()

	# fig= plt.figure(figsize=(10,3))
	plt.subplot(3, 1, 2)
	ax = plt.gca() 
	plt.bar(range(1,length+1), baseline_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	# plt.text(45, box_y, baseline_txt,  bbox=box)
	if txt: plt.text(bx, by, baseline_txt, bbox=box, transform=ax.transAxes, fontsize=txtsize)
	plt.title('Baseline (GT regression)', fontsize=15, y=-0.3)
	# plt.title('Soft-encoding from GT', fontsize=15, y=-0.3)
	# plt.title('No mask', fontsize=15, y=-0.3)
	# plt.savefig('color_distri_plots/' + dn + '_baseline.pdf')
	# plt.clf()

	# fig= plt.figure(figsize=(10,3))
	plt.subplot(3, 1, 3)
	plt.bar(range(1,length+1), gt_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	plt.title('GT', fontsize=15, y=-0.3)
	
	plt.suptitle('Data: ' + concept, fontsize=20, y=0.95)
	plt.subplots_adjust(hspace=0.5)
	plt.savefig(save_file)
	plt.close()
	plt.clf()
	

def fig_4_in_1(	concept, 
				net_metrics, nomask_metrics, baseline_metrics,
				gt_ratings, net_ratings, nomask_ratings, baseline_ratings, 
				save_file, ratings_space=58, box_pos=[0.75, 0.6]):

	### sorting
	if ratings_space==58:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_58_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[3] = np.array([0.3]*3)
	elif ratings_space==71:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_71_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[2] = np.array([0.3]*3)
	elif ratings_space==37:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_37_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[20] = np.array([0.3]*3)
	gt_ratings			= gt_ratings[sorting_idx]
	net_ratings			= net_ratings[sorting_idx]
	nomask_ratings		= nomask_ratings[sorting_idx]
	baseline_ratings	= baseline_ratings[sorting_idx]

	max_y_1 = np.max(gt_ratings)
	max_y_2 = np.max(net_ratings)
	max_y_3 = np.max(nomask_ratings)
	max_y_4 = np.max(baseline_ratings)
	max_y = np.max([max_y_1, max_y_2, max_y_3, max_y_4])

	# network_txt = 	'EMD:         %.4f\n'	% net_metrics['EMD']	+ \
	# 				'sqErr:        %.4f\n'	% net_metrics['sqErr']	+ \
	# 				'CorrCoeff: %.4f\n'		% net_metrics['Corr']	+ \
	# 				'pVal:         %.4f'	% net_metrics['pVal']
	
	# nomask_txt = 	'EMD:         %.4f\n'	% nomask_metrics['EMD']	+ \
	# 				'sqErr:        %.4f\n'	% nomask_metrics['sqErr']	+ \
	# 				'CorrCoeff: %.4f\n'		% nomask_metrics['Corr']	+ \
	# 				'pVal:         %.4f'	% nomask_metrics['pVal']
	
	# baseline_txt = 	'EMD:         %.4f\n'	% baseline_metrics['EMD']	+ \
	# 				'sqErr:        %.4f\n'	% baseline_metrics['sqErr']	+ \
	# 				'CorrCoeff: %.4f\n'		% baseline_metrics['Corr']	+ \
	# 				'pVal:         %.4f'	% baseline_metrics['pVal']

	# fig = plt.figure(figsize=(8,10))
	fig = plt.figure(figsize=(8,13))
	length = len(gt_ratings)
	box = {'facecolor':'0.9', 'edgecolor':'k', 'boxstyle':'round'}
	bx, by = box_pos

	# plt.subplot(3, 1, 1)
	plt.subplot(4, 1, 1)
	ax = plt.gca() 
	plt.bar(range(1,length+1), net_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	# plt.text(45, box_y, network_txt,  bbox=box)
	# plt.text(bx, by, network_txt, bbox=box, transform=ax.transAxes)
	plt.title('Ours (self-supervised colorization)', fontsize=15, y=-0.3)
	# plt.savefig('color_distri_plots/' + dn + '_network.pdf')
	# plt.clf()

	# plt.subplot(3, 1, 1)
	plt.subplot(4, 1, 2)
	ax = plt.gca() 
	plt.bar(range(1,length+1), nomask_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	# plt.text(45, box_y, network_txt,  bbox=box)
	# plt.text(bx, by, pretrain_txt, bbox=box, transform=ax.transAxes)
	plt.title('No Mask', fontsize=15, y=-0.3)
	# plt.savefig('color_distri_plots/' + dn + '_network.pdf')
	# plt.clf()

	# fig= plt.figure(figsize=(10,3))
	# plt.subplot(3, 1, 2)
	plt.subplot(4, 1, 3)
	ax = plt.gca() 
	plt.bar(range(1,length+1), baseline_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	# plt.text(45, box_y, baseline_txt,  bbox=box)
	# plt.text(bx, by, baseline_txt, bbox=box, transform=ax.transAxes)
	plt.title('Baseline (GT regression)', fontsize=15, y=-0.3)
	# plt.savefig('color_distri_plots/' + dn + '_baseline.pdf')
	# plt.clf()

	# fig= plt.figure(figsize=(10,3))
	# plt.subplot(3, 1, 3)
	plt.subplot(4, 1, 4)
	plt.bar(range(1,length+1), gt_ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.1)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	plt.title('GT', fontsize=15, y=-0.3)
	
	plt.suptitle('Data: ' + concept, fontsize=20, y=0.95)
	plt.subplots_adjust(hspace=0.5)
	plt.savefig(save_file)
	plt.clf()


def draw_two_hist(fig_title, ratings1, ratings2, name1, name2, save_file, ratings_space):
	### sorting
	if ratings_space==58:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_58_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[3] = np.array([0.3]*3)
	elif ratings_space==71:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_71_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[2] = np.array([0.3]*3)
	elif ratings_space==37:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_37_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[20] = np.array([0.3]*3)
	
	ratings1 = normalize(ratings1)
	ratings1 = ratings1[sorting_idx]
	ratings2 = normalize(ratings2)
	ratings2 = ratings1[sorting_idx]
	length = len(ratings1)
	max_y1 = np.max(ratings1)
	max_y2 = np.max(ratings2)
	max_y = np.max((max_y1, max_y2))
	
	plt.figure(figsize=(8,5))
	
	plt.subplot(2, 1, 1)
	plt.bar(range(1, length+1), ratings1, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.5)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	plt.xlabel('Colors')
	plt.title(name1, fontsize=20)

	plt.subplot(2, 1, 2)
	plt.bar(range(1, length+1), ratings2, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.5)
	plt.ylim(0, max_y+0.01)
	plt.ylabel('Ratings')
	plt.xlabel('Colors')
	plt.title(name2, fontsize=20)

	plt.subplots_adjust(hspace=0.5)
	plt.savefig(save_file)
	plt.close()
	plt.clf()


def draw_one_hist(fig_title, ratings, save_file, ratings_space, max_y=None, no_txt=False):
	### sorting
	if ratings_space==58:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_58_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[3] = np.array([0.3]*3)
	elif ratings_space==71:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_71_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[2] = np.array([0.3]*3)
	elif ratings_space==37:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_37_to_visulize()
		edge_color = sort_colors.copy()
		edge_color[20] = np.array([0.3]*3)
	elif ratings_space == 20: ### Tableau 20
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_Tableau20_to_visulize()
		edge_color = sort_colors.copy()
	elif ratings_space == 30: ### Tableau 20 + Expert10
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_EXP30_to_visulize()
		edge_color = sort_colors.copy()
	elif ratings_space == 27: ### Tableau 20 + Expert7
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_EXP27_to_visulize()
		edge_color = sort_colors.copy()
	elif ratings_space == 28: ### Tableau 20 + Expert7 + black
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_EXP28_to_visulize()
		edge_color = sort_colors.copy()
	elif ratings_space == 65: ### UW 58 + Expert7
		# sorting_idx, sort_colors = CONFIG.get_sorted_RGB_UWE65_to_visulize()
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_UWE66_to_visulize()	#TODO temp
		edge_color = sort_colors.copy()
	# elif ratings_space == 66: ### UW 58 + Expert7 + black
	# 	sorting_idx, sort_colors = CONFIG.get_sorted_RGB_UWE66_to_visulize()
	# 	edge_color = sort_colors.copy()
	
	ratings = normalize(ratings)
	ratings = ratings[sorting_idx]
	length = len(ratings)
	if max_y is None:
		max_y = np.max(ratings)
	
	# plt.figure(figsize=(8,3))
	plt.figure(figsize=(5,3))
	plt.bar(range(1,length+1), ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.5)
	plt.ylim(0, max_y+0.01)
	# plt.ylabel('Ratings')
	# plt.xlabel('Colors')
	if no_txt:
		plt.xticks([])
		plt.yticks([])
	if not no_txt:
		plt.title(fig_title, fontsize=15)
	plt.savefig(save_file)
	plt.close()
	plt.clf()


def read_metrics(file_path):
	df = pd.read_csv(file_path, index_col='Concept')
	metrics = {}
	for c in df.index:
		metrics[c] = df.loc[c]#.to_numpy()
	return metrics


"""
	EXPs
"""


### fruits12
def exp1_1(rounds=None, reduce_00_sigma=None):
	concepts = CONFIG.FRUITS_12
	dir = './results/'

	### GT
	all_gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	ratings_space = 58

	### baseline
	# baseline_file			= 'fruits_baseline.csv'
	# baseline_col			= 'Top50_Sector+Category'
	# all_baseline_ratings	= read_baseline(dir+baseline_file, baseline_col, concepts)
	

	baseline_metrics_file	= 'exp1_1-fruits12-metrics-baseline.csv'
	all_baseline_metrics	= read_metrics(dir+baseline_metrics_file)

	### net results
	note = ''
	if rounds: note += '-%s' % rounds
	# net_file			= 'exp1_1-ratings58-net-finetuned-mask-25.txt'
	# net_metrics_file	= 'exp1_1-net-finetuned-mask-25-metrics.csv'
	net_file			= 'exp1_1-fruits12-ratings58-net-finetuned-mask-25%s.txt' % note
	all_net_ratings		= read_net_results(dir+net_file, concepts, reduce_00_sigma)

	if reduce_00_sigma: note += '-%.1f' % reduce_00_sigma
	net_metrics_file	= 'exp1_1-fruits12-metrics-net-finetuned-mask-25%.csv' % note
	all_net_metrics		= read_metrics(dir+net_metrics_file)

	for c in tqdm(concepts):
		fig_3_in_1(	c, 
					all_net_metrics[c], 
					all_baseline_metrics[c],
					all_gt_ratings[c],
					all_net_ratings[c],
					all_baseline_ratings[c],
					'figs/exp1_1-%s-finetuned-mask-25%s.pdf' % (c, note),
					# 'figs/exp1_1-%s-finetuned-mask-25.pdf' % c,
					ratings_space=ratings_space,
					box_pos=[0.85, 0.5],
					)
	
### fruits12_cartoon
def exp1_2():
	concepts = CONFIG.FRUITS_12
	dir = './results/'

	### GT
	all_gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	ratings_space = 58

	### baseline
	baseline_file			= 'fruits_baseline.csv'
	baseline_col			= 'Cartoon50_Sector+Category'
	all_baseline_ratings	= read_baseline(dir+baseline_file, baseline_col, concepts)
	baseline_metrics_file	= 'exp1_2-fruits12_cartoon-metrics-baseline.csv'
	all_baseline_metrics	= read_metrics(dir+baseline_metrics_file)

	### net results
	net_file			= 'exp1_2-fruits12_cartoon-ratings58-net-finetuned-mask-25.txt'
	net_metrics_file	= 'exp1_2-fruits12_cartoon-metrics-net-finetuned-mask-25.csv'
	all_net_ratings		= read_net_results(dir+net_file, concepts)
	all_net_metrics		= read_metrics(dir+net_metrics_file)

	# ### nomask results
	# nomask_file			= 'exp1_2-fruits12_cartoon-ratings58-net-finetuned-nomask-25.txt'
	# nomask_ratings		= read_net_results(dir+nomask_file, concepts)

	### soft encoding results
	# soft_file		= 'exp8-fruits12_cartoon-ratings-GT-soft-encoding-mask.txt'
	# soft_ratings	= read_net_results(dir+soft_file, concepts)

	for c in tqdm(concepts):
		fig_3_in_1(	c, 
					all_net_metrics[c], 
					all_baseline_metrics[c],
					all_gt_ratings[c],
					all_net_ratings[c],

					# nomask_ratings[c],
					# soft_ratings[c],
					all_baseline_ratings[c],
					
					# 'figs/soft/exp1_2-%s_cartoon-finetuned-mask-25.pdf' % c,
					'figs/debug-exp1_2-%s_cartoon-finetuned-mask-25.pdf' % c,
					ratings_space=ratings_space,
					box_pos=[0.85, 0.5],
					txt=True
					)

### fruits12_photo
def exp1_3():
	concepts = CONFIG.FRUITS_12
	dir = './results/'

	### GT
	all_gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
	ratings_space = 58

	### baseline
	baseline_file			= 'fruits_baseline.csv'
	baseline_col			= 'Photo50_Sector+Category'
	all_baseline_ratings	= read_baseline(dir+baseline_file, baseline_col, concepts)
	baseline_metrics_file	= 'exp1_3-fruits12_photo-metrics-baseline.csv'
	all_baseline_metrics	= read_metrics(dir+baseline_metrics_file)

	### net results
	# net_file			= 'exp1_1-ratings58-net-finetuned-mask-25.txt'
	# net_metrics_file	= 'exp1_1-net-finetuned-mask-25-metrics.csv'
	net_file			= 'exp1_3-fruits12_photo-ratings58-net-finetuned-mask-25.txt'
	net_metrics_file	= 'exp1_3-fruits12_photo-metrics-net-finetuned-mask-25.csv'
	all_net_ratings		= read_net_results(dir+net_file, concepts)
	all_net_metrics		= read_metrics(dir+net_metrics_file)

	for c in tqdm(concepts):
		fig_3_in_1(	c, 
					all_net_metrics[c], 
					all_baseline_metrics[c],
					all_gt_ratings[c],
					all_net_ratings[c],
					all_baseline_ratings[c],
					'figs/exp1_3-%s_photo-finetuned-mask-25.pdf' % c,
					ratings_space=ratings_space,
					box_pos=[0.85, 0.5],
					)

### recycle6
def exp1_5():
	concepts = CONFIG.RECYCLE_6
	dir = './results/'

	### GT
	all_gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()
	ratings_space = 37

	### baseline
	baseline_folder = '/Users/codeb/Downloads/eccv_result/get_baseline_rt/recycle6/'
	all_baseline_ratings = {}
	for c in concepts:
		ratings37 = np.load(baseline_folder + c + '.npy')
		all_baseline_ratings[c] = normalize(ratings37)
	
	baseline_metrics_file	= 'exp1_5-recycle6-metrics-baseline.csv'
	all_baseline_metrics	= read_metrics(dir+baseline_metrics_file)


	# ### pretrain results
	# net_file			= 'exp1_5-recycle6-ratings37-net-pretrain-mask-25.txt'
	# all_pre_ratings		= read_net_results(dir+net_file, concepts)
	# net_metrics_file	= 'exp1_5-recycle6-metrics-net-pretrain-mask-25.csv'
	# all_pre_metrics		= read_metrics(dir+net_metrics_file)

	### net results
	net_file			= 'exp1_5-recycle6-ratings37-net-finetuned-mask-25.txt'
	all_net_ratings		= read_net_results(dir+net_file, concepts)
	net_metrics_file	= 'exp1_5-recycle6-metrics-net-finetuned-mask-25.csv'
	all_net_metrics		= read_metrics(dir+net_metrics_file)

	for c in tqdm(concepts):
		fig_3_in_1(	c, 
					all_net_metrics[c],
					# all_pre_metrics[c], 
					all_baseline_metrics[c],
					all_gt_ratings[c],
					all_net_ratings[c],
					# all_pre_ratings[c],
					all_baseline_ratings[c],
					'figs/exp1_5-%s-finetuned-mask-25.pdf' % c,
					# 'figs/exp1_1-%s-finetuned-mask-25.pdf' % c,
					ratings_space=ratings_space,
					box_pos=[0.9, 0.5],
					# box_pos=[0.05,0.65],
					)

# vegetables5
def exp2_1():
	concepts = CONFIG.VEGETABLES_5
	dir = './results/'

	### GT
	all_gt_ratings = CONFIG.get_VEGETABLES_5_RATING_GT_71()
	ratings_space = 71

	### baseline
	baseline_folder = '/Users/codeb/Downloads/eccv_result/get_baseline_rt/vegetable5/'
	all_baseline_ratings = {}
	for c in concepts:
		ratings71 = np.load(baseline_folder + c + '.npy')
		all_baseline_ratings[c] = normalize(ratings71)
	
	baseline_metrics_file	= 'exp2_1-vegetables5-metrics-baseline.csv'
	all_baseline_metrics	= read_metrics(dir+baseline_metrics_file)


	# ### pretrain results
	# net_file			= 'exp1_5-recycle6-ratings37-net-pretrain-mask-25.txt'
	# all_pre_ratings		= read_net_results(dir+net_file, concepts)
	# net_metrics_file	= 'exp1_5-recycle6-metrics-net-pretrain-mask-25.csv'
	# all_pre_metrics		= read_metrics(dir+net_metrics_file)

	### net results
	net_file			= 'exp2_1-vegetables5-ratings71-net-finetuned-mask-25.txt'
	all_net_ratings		= read_net_results(dir+net_file, concepts)
	net_metrics_file	= 'exp2_1-vegetables5-metrics-net-finetuned-mask-25.csv'
	all_net_metrics		= read_metrics(dir+net_metrics_file)

	for c in tqdm(concepts):
		fig_3_in_1(	c, 
					all_net_metrics[c],
					all_baseline_metrics[c],
					all_gt_ratings[c],
					all_net_ratings[c],
					all_baseline_ratings[c],
					'figs/exp2_1-%s-finetuned-mask-25.pdf' % c,
					ratings_space=ratings_space,
					box_pos=[0.85, 0.55],
					)

# fruits5
def exp2_2():
	concepts = CONFIG.FRUITS_5
	dir = './results/'

	### GT
	all_gt_ratings = CONFIG.get_FRUITS_5_RATING_GT_71()
	ratings_space = 71

	### baseline
	baseline_folder = '/Users/codeb/Downloads/eccv_result/get_baseline_rt/fruits5/'
	all_baseline_ratings = {}
	for c in concepts:
		ratings71 = np.load(baseline_folder + c + '.npy')
		all_baseline_ratings[c] = normalize(ratings71)
	
	baseline_metrics_file	= 'exp2_2-fruits5-metrics-baseline.csv'
	all_baseline_metrics	= read_metrics(dir+baseline_metrics_file)

	# ### pretrain results
	# net_file			= 'exp1_5-recycle6-ratings37-net-pretrain-mask-25.txt'
	# all_pre_ratings		= read_net_results(dir+net_file, concepts)
	# net_metrics_file	= 'exp1_5-recycle6-metrics-net-pretrain-mask-25.csv'
	# all_pre_metrics		= read_metrics(dir+net_metrics_file)

	### net results
	net_file			= 'exp2_2-fruits5-ratings71-net-finetuned-mask-25.txt'
	all_net_ratings		= read_net_results(dir+net_file, concepts)
	net_metrics_file	= 'exp2_2-fruits5-metrics-net-finetuned-mask-25.csv'
	all_net_metrics		= read_metrics(dir+net_metrics_file)

	for c in tqdm(concepts):
		fig_3_in_1(	c, 
					all_net_metrics[c],
					all_baseline_metrics[c],
					all_gt_ratings[c],
					all_net_ratings[c],
					all_baseline_ratings[c],
					'figs/exp2_2-%s-finetuned-mask-25.pdf' % c,
					ratings_space=ratings_space,
					box_pos=[0.85, 0.45],
					)

### newest function
def draw_any_dataset(dataset, seperate=False):
	"""
		dataset: exp1_1-fruits12 / 
				exp1_5-recycle6 / exp2_1-vegetables5 / exp2_2-fruits5
	"""
	dsn = dataset.split('-')[-1]
	if dsn == 'fruits12':
		rounds = '-8000'
		reduce_sigma = 0.3
		reduce_sigma_note = '-0.3'
		concepts = CONFIG.FRUITS_12
		gt_ratings = CONFIG.get_FRUITS_12_RATING_GT_58()
		ratings_space = 58
		box_pos = [0.8, 0.6]
	if dsn == 'recycle6':
		rounds = '-10000'
		reduce_sigma = 0.75
		reduce_sigma_note = '-0.75'
		concepts = CONFIG.RECYCLE_6
		gt_ratings = CONFIG.get_RECYCLE_6_RATING_GT_37()
		ratings_space = 37
		box_pos = [0.87, 0.6]
	if dsn == 'fruits5':
		rounds = ''
		reduce_sigma = 0.3
		reduce_sigma_note = '-0.3'
		concepts = CONFIG.FRUITS_5
		gt_ratings = CONFIG.get_FRUITS_5_RATING_GT_71()
		ratings_space = 71
		box_pos = [0.8, 0.6]
	if dsn == 'vegetables5':
		rounds = ''
		reduce_sigma = 0.3
		reduce_sigma_note = '-0.3'
		concepts = CONFIG.VEGETABLES_5
		gt_ratings = CONFIG.get_VEGETABLES_5_RATING_GT_71()
		ratings_space = 71
		box_pos = [0.8, 0.6]

	##### metrics
	baseline_metrics	= read_metrics('./results/real_baseline-%s-metrics.csv'						% dsn)
	no_net_metrics		= read_metrics('./results/exp8-%s-metrics-GT-soft-encoding-mask.csv'		% dsn)
	no_fine_metics		= read_metrics('./results/%s-metrics-net-pretrain-mask-25%s.csv'			% (dataset,	reduce_sigma_note))
	no_mask_metrics		= read_metrics('./results/%s-metrics-net-finetuned-nomask-25%s%s.csv'		% (dataset,	rounds,	reduce_sigma_note))
	full_metrics		= read_metrics('./results/%s-metrics-net-finetuned-mask-25%s%s.csv'			% (dataset,	rounds,	reduce_sigma_note))
	single_metrics		= read_metrics('./results/exp7-%s-metrics-net-single_concept-mask-25%s.csv'	% (dsn, reduce_sigma_note))
	dumb_metrics		= read_metrics('./results/exp6-%s-metrics-baseline-fruits12trained.csv'		% dsn)

	##### results
	### basline
	# baseline_dir = '/Users/codeb/Downloads/eccv_result/baseline_%s_result/' % dsn
	baseline_dir = './net_results/baseline_%s_result/' % dsn
	baseline_ratings = {}
	for c in concepts: baseline_ratings[c] = normalize(np.load(baseline_dir + c + '.npy'))
	### full pipeline
	full_file = './results/%s-ratings%d-net-finetuned-mask-25%s.txt' % (dataset, ratings_space, rounds)
	full_ratings = read_net_results(full_file, concepts, reduce_00_sigma=reduce_sigma)

	##### drawing
	for c in tqdm(concepts):
		if not seperate:
			fig_3_in_1(	c, 
						full_metrics[c],
						baseline_metrics[c],
						gt_ratings[c],
						full_ratings[c],
						baseline_ratings[c],
						'figs/%s-%s-ratings.pdf' % (dsn, c),
						ratings_space=ratings_space,
						box_pos=box_pos,
						)
		else:
			### draw one by one
			max_y_1 = np.max(gt_ratings[c])
			max_y_2 = np.max(full_ratings[c])
			max_y_3 = np.max(baseline_ratings[c])
			max_y = np.max([max_y_1, max_y_2, max_y_3])
			### net
			fig_title = '%s-%s Ours (self-supervised colorization)'  % (dsn, c)
			# save_file = './figs/seperate_images/%s-%s-ours.pdf' % (dsn, c)
			save_file = './figs/US_images/%s-%s-ours.pdf' % (dsn, c)
			draw_one_hist(fig_title, full_ratings[c], save_file, ratings_space, max_y, no_txt=True)
			### baseline
			fig_title = '%s-%s Baseline (GT regression)'  % (dsn, c)
			# save_file = './figs/seperate_images/%s-%s-baseline.pdf' % (dsn, c)
			save_file = './figs/US_images/%s-%s-baseline.pdf' % (dsn, c)
			draw_one_hist(fig_title, baseline_ratings[c], save_file, ratings_space, max_y, no_txt=True)
			### gt
			fig_title = '%s-%s GT' % (dsn, c)
			# save_file = './figs/seperate_images/%s-%s-GT.pdf' % (dsn, c)
			save_file = './figs/US_images/%s-%s-GT.pdf' % (dsn, c)
			draw_one_hist(fig_title, gt_ratings[c], save_file, ratings_space, max_y, no_txt=True)

"""
	temp exps
"""

def draw_fruits12_trained_baseline():
	save_dir = './results/'
	results_dir = './temp_results/exp6-baseline-fruits12trained/'
	result_settings = {
		'fruits12':			('baseline_test12fruit.npy', CONFIG.FRUITS_12,	CONFIG.get_FRUITS_12_RATING_GT_58(),	58),
		# 'fruits12_cartoon':	('cartoon.npy',			CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58(),	58),
		# 'fruits12_photo':	('photo.npy',			CONFIG.FRUITS_12,		CONFIG.get_FRUITS_12_RATING_GT_58(),	58),
		'recycle6':			('recycle.npy',			CONFIG.RECYCLE_6,		CONFIG.get_RECYCLE_6_RATING_GT_37(),	37),
		'vegetables5':		('5_veg.npy',			CONFIG.VEGETABLES_5,	CONFIG.get_VEGETABLES_5_RATING_GT_71(),	71),
		'fruits5':			('5other_fruit.npy',	CONFIG.FRUITS_5,		CONFIG.get_FRUITS_5_RATING_GT_71(),		71),
	}
	for k in result_settings.keys():
		all_result_ratings = np.load(results_dir + result_settings[k][0])
		result_ratings = normalize(result_ratings)
		concepts = result_settings[k][1]
		gt_ratings = result_settings[k][2]
		ratings_space = result_settings[k][3]
		for c in concepts:
			fig_title = 'Fruit12trained_baseline-%s-%s' % (k, c)
			ratings1 = result_ratings
			ratings2 = gt_ratings
			name1 = 'result'
			name2 = 'GT'
			save_file = 'figs/Fruit12trained_baseline-%s-%s.pdf' % (k, c)
			draw_two_hist(fig_title, ratings1, ratings2, name1, name2, save_file, ratings_space)


def draw_watermelon47():
	concept = 'watermelon47'
	ratings = np.loadtxt('./results/exp1_1-watermelon47-ratings58-finetuned-mask.txt')
	save_file = './figs/watermelon47-finetuned-mask.pdf'
	draw_one_hist(concept, ratings, save_file)

def draw_strawberry30():
	concept = 'strawberry30'
	# ratings = np.loadtxt('./results/exp1_1-strawberry30-ratings58-finetuned-mask.txt')
	# save_file = './figs/strawberry30-finetuned-mask.pdf'
	ratings = np.loadtxt('./results/exp1_1-strawberry30-ratings58-finetuned-nomask.txt')
	save_file = './figs/strawberry30-finetuned-nomask.pdf'
	draw_one_hist(concept, ratings, save_file, 58)

def draw_cartoon_orange_32():
	concept = 'orange_cartoon32'
	ratings = np.loadtxt('./results/exp1_2-orange_cartoon32-ratings58-finetuned-mask.txt')
	save_file = './figs/orange_cartoon32-finetuned-mask.pdf'
	# ratings = np.loadtxt('./results/exp1_2-orange_cartoon32-ratings58-finetuned-nomask.txt')
	# save_file = './figs/orange_cartoon32-finetuned-nomask.pdf'
	draw_one_hist(concept, ratings, save_file, 58)

def draw_all_img_in_one_concept(dataset='fruits12', concept='blueberry'):
	ratings_dir = './results/%s/%s/'	% (dataset, concept)
	save_dir = './figs/%s/%s/'			% (dataset, concept)
	os.makedirs(save_dir, exist_ok=True)
	ratings_sapce = 58
	for i in range(50):
		fig_title = '%s_%s-%d' % (dataset,concept, i)
		ratings = np.loadtxt(ratings_dir + '%d-mask.txt' % i)
		save_file = save_dir + '%d-mask.pdf' % i
		draw_one_hist(fig_title, ratings, save_file, ratings_sapce)

def tune_mapping():
	concepts = CONFIG.FRUITS_12
	note = 'finetuned-mask-25'
	dir = './results_tune_mapping/'
	note = 'finetuned-mask-25'
	# for beta in [1.0, 0.5, 0.2, 0.7, 0.1, 0.9]:
		# for T in [0.1, 0.2, 0.5, 0.8, 0.9]:
	for beta in [1.0, 0.5, 0.2, 0.7, 0.1, 0.9, 
				 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
				 1.15, 1.25]:
		for T in [0.1, 0.2, 0.5, 0.8, 0.9, 0.38, 1.0, 1.5, 0.01, 0.05, 0.3, 1.5, 2.0]:
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
				for c in concepts:
					ratings = result_ratings[c]
					fig_file = dir + 'figs/%s-%s.pdf' % (c, note2)
					draw_one_hist(c, ratings, fig_file)
	
def draw_several_px():
	for i in range(9, 12):
		for j in range(26, 30):
			title = 'cartoon-blueberry-2 px(%d,%d)' % (i,j)
			ratings_file = './results/pixels/fruits12_cartoon-blueberry-2-px_%d_%d.txt'%(i,j)
			ratings = np.loadtxt(ratings_file)
			save_file = './figs/pixels/fruits12_cartoon-blueberry-2-px_%d_%d.pdf'%(i,j)
			draw_one_hist(title, ratings, save_file, 58)

def draw_color_distance():
	ratings_space = 58

	if ratings_space==58:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_58_to_visulize()
		lab_colors = CONFIG.get_LAB_58()[sorting_idx]
		edge_color = sort_colors.copy()
		edge_color[3] = np.array([0.3]*3)
	elif ratings_space==71:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_71_to_visulize()
		lab_colors = CONFIG.get_LAB_71()[sorting_idx]
		edge_color = sort_colors.copy()
		edge_color[2] = np.array([0.3]*3)
	elif ratings_space==37:
		sorting_idx, sort_colors = CONFIG.get_sorted_RGB_37_to_visulize()
		lab_colors = CONFIG.get_LAB_37()[sorting_idx]
		edge_color = sort_colors.copy()
		edge_color[20] = np.array([0.3]*3)
	
	dis = np.zeros(ratings_space)
	for i in range(ratings_space):
		dis[i] = (np.sum(lab_colors[i][1:]**2))**0.5
	
	ratings = dis
	# ratings = np.ones(ratings_space)
	length = len(ratings)
	max_y = np.max(ratings)
	
	plt.figure(figsize=(25,5))
	plt.bar(range(1,length+1), ratings, color=sort_colors, 
				width=1.0, edgecolor=edge_color, linewidth=0.5)
	plt.ylim(0, max_y+0.1)
	plt.ylabel('Ratings')
	plt.xlabel('Colors')
	plt.title('color ab distance to (0,0)', fontsize=20)

	for i in range(ratings_space):
		plt.text(i+0.5, dis[i]+0.1, '%.1f'%dis[i])

	plt.savefig('temp/color_distance.pdf')
	plt.close()
	plt.clf()
	print(lab_colors)

if __name__ == '__main__':
	# exp1_1()
	# exp1_2()
	# # draw_several_px()
	# exp1_3()
	# exp1_5()
	# exp2_1()
	# exp2_2()
	# draw_color_distance()
	# draw_watermelon47()
	# draw_strawberry30()
	# draw_cartoon_orange_32()
	# draw_all_img_in_one_concept('fruits12_cartoon', 'blueberry')
	
	# draw_fruits12_trained_baseline()
	# tune_mapping()

	# for T in np.linspace(0.1, 1, 10):
		# for beta in np.linspace(0.1, 1, 10):
	# for T in [1.0]:
	# 	for beta in np.linspace(1.1, 2, 10):
	# 		n = 'pixel_40_25-T%.2f-b%.2f' % (T, beta)
	# 		r = np.loadtxt('./px_40_25/%s.txt' % n)
	# 		draw_one_hist(n, r, './px_40_25/%s.pdf' % n)

	datasets = ['exp1_1-fruits12', 'exp1_5-recycle6', 'exp2_1-vegetables5', 'exp2_2-fruits5']
	for d in datasets:
		draw_any_dataset(d, seperate=True)
	# draw_any_dataset(datasets[1])