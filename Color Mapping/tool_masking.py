#-*- coding: UTF-8 -*-
import os
import numpy as np
import rembg
import rembg.detect
# import rembg.detect as rd
import matplotlib.pyplot as plt
import PIL.Image
import pickle
from tqdm import tqdm
from settings.config import CONFIG



"""
	Masking Tools
"""

### 这个方法在最后被发现不太好，应该还是要用一个有alpha的mask
def generate_masks(img_file, target_size, mask_threshold=5, get_masked_img=True):
	"""
	Input:
		img_file: 			str					| image file path
		target_size:		(int, int)			| (w=56,h=56) in our setting
		mask_threshold:		int					| 5 in our setting. this is not a good solution
		get_masked_img: 	bool				| if True, return masked image as well (in PIL format)
	Output:
		mask_onehot:		(w,h) bool array	| True for foreground pixel, False for background pixel
		img_pil_masked:		PIL image			| masked image if get_masked_img is True
	"""
	### mask
	img_pil = PIL.Image.open(img_file)
	img_rgb_np = np.array(img_pil.convert('RGB'))
	mask = rembg.detect.predict(rembg.detect.ort_session('u2net'), img_rgb_np).convert('L')
	mask_resize = mask.resize((target_size), PIL.Image.LANCZOS)
	mask_np = np.asarray(mask_resize)		### range: 0~255
	mask_onehot = mask_np > mask_threshold	### True / False
	### masked img
	if get_masked_img:	
		img_pil_resize = img_pil.resize((target_size), PIL.Image.LANCZOS)
		img_np = np.array(img_pil_resize.convert('RGB'))
		img_np[np.where(mask_onehot==False)] = np.array([0, 0, 0])
		img_pil_masked = PIL.Image.fromarray(img_np, mode='RGB')
		return mask_onehot, img_pil_masked
	return mask_onehot


### 不使用threshold，后续的步骤要在mapping里改
### 没测试过
def generate_alpha_masks(img_file, target_size, get_masked_img=True):
	"""
	Input:
		img_file: 			str							| image file path
		target_size:		(int, int)					| (w=56,h=56) in our setting
		get_masked_img: 	bool						| if True, return masked image as well (in PIL format)
	Output:
		alpha_mask:			(w,h) array in range 0~1	| 1.0 for foreground pixel, 0.0 for background pixel
		img_pil_masked:		PIL image					| masked image if get_masked_img is True
	"""
	### mask
	img_pil = PIL.Image.open(img_file)
	img_rgb_np = np.array(img_pil.convert('RGB'))
	mask = rembg.detect.predict(rembg.detect.ort_session('u2net'), img_rgb_np).convert('L')
	mask_resize = mask.resize((target_size), PIL.Image.LANCZOS)
	mask_np = np.asarray(mask_resize)	### range: 0   ~ 255
	alpha_mask = mask_np / 255			### range: 0.0 ~ 1.0	
	### masked img
	if get_masked_img:
		img_pil_resize = img_pil.resize((target_size), PIL.Image.LANCZOS).convert('RGB')
		img_pil_masked = rembg.bg.naive_cutout(img_pil_resize, mask_resize)
		### or rembg.bg.alpha_matting_cutout(), see more details at bg.py from rembg
		return alpha_mask, img_pil_masked
	return alpha_mask


"""
	Experiments
"""

exp1_1_trainlist_folder		= '/Users/codeb/Downloads/eccv_result/exp1_1-net-pretrain/'
exp1_1_datasets_img_folder	= './datasets/baseline_12fruit/top/'
fruits12_mask_folder		= './datasets/fruits12_masked/'

exp1_2_trainlist_folder		= '/Users/codeb/Downloads/eccv_result/exp1_2-net-finetuned/'
exp1_2_datasets_img_folder	= './datasets/baseline_12fruit/cartoon/'
fruits12_cartoon_mask_folder= './datasets/fruits12_cartoon_masked/'

exp1_3_trainlist_folder		= '/Users/codeb/Downloads/eccv_result/exp1_3-net-finetuned/'
exp1_3_datasets_img_folder	= './datasets/baseline_12fruit/photo/'
fruits12_photo_mask_folder	= './datasets/fruits12_photo_masked/'

# exp1_5_trainlist_folder		= '/Users/codeb/Downloads/eccv_result/exp1_5-net-finetuned/'	### all(over50)
exp1_5_trainlist_folder		= '/Users/codeb/Downloads/eccv_result/recycle6-single_train/'
exp1_5_datasets_img_folder	= './datasets/baseline_6recycle/'
recycle6_mask_folder		= './datasets/recycle6_masked/'

exp2_datasets_img_folder	= './datasets/ours_10ftadveg/'
vegetables5_mask_folder		= './datasets/vegetables5_masked/'
fruits5_mask_folder			= './datasets/fruits5_masked/'

exp11_100_datasets_img_folder	= './datasets/100_5ftadveg/'
exp11_200_datasets_img_folder	= './datasets/200_5ftadveg/'
exp11_100_fruits5_mask_folder	= './datasets/fruits5_100_masked/'
exp11_200_fruits5_mask_folder	= './datasets/fruits5_200_masked/'


def debug():
	# veg_mask
	img_file_list = np.loadtxt(exp2_datasets_img_folder + 'train_veg.txt', dtype=str)[:, 0]

	img_file = img_file_list[4]

	# concept = img_file.split('/')[0]
	img_file = exp2_datasets_img_folder + img_file

	img_pil = PIL.Image.open(img_file)
	plt.imshow(img_pil)

	mask = rembg.detect.predict(rembg.detect.ort_session('u2net'), np.array(img_pil.convert('RGB')))
	mask_L = mask.convert('L')
	mask_r = mask_L.resize((56, 56), PIL.Image.LANCZOS)
	plt.imshow(mask_r)

	mask_np = np.asarray(mask_r)
	for i in range(20):
		mask_onehot = mask_np > i
		print(np.sum(mask_onehot))
	plt.imshow(mask_onehot)



def exp1_1_fruits12_masks(target_size, mask_threshold=5):
	# img_file_list = np.loadtxt(exp2_datasets_img_folder + 'train_veg.txt', dtype=str)[:, 0]
	if not os.path.exists(fruits12_mask_folder):
		os.mkdir(fruits12_mask_folder)
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for c in CONFIG.FRUITS_12:
		img_file_list = np.loadtxt(exp1_1_trainlist_folder + c + 'imglist.txt', dtype=str)
		num_img = len(img_file_list)
		for in_concept_idx, img_file_path in tqdm(enumerate(img_file_list)):
			if not c in all_masks.keys():
				all_masks[c] = np.zeros(( num_img , num_pixels )).astype(bool)
				# os.mkdir(fruits12_mask_folder + c + '-masked')
			img_file = img_file_path.split('/')[-1]
			img_file_path = exp1_1_datasets_img_folder + c + '/' + img_file
			# mask, img_masked = generate_masks(img_file_path, target_size, mask_threshold)
			mask = generate_masks(img_file_path, target_size, mask_threshold)
			all_masks[c][in_concept_idx] = mask.reshape(-1,)
			# img_masked.save(fruits12_mask_folder + c + '-masked/%d.png' % in_concept_idx)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


def exp1_2_fruits12_cartoon_masks(target_size, mask_threshold=5):
	# img_file_list = np.loadtxt(exp2_datasets_img_folder + 'train_veg.txt', dtype=str)[:, 0]
	if not os.path.exists(fruits12_cartoon_mask_folder):
		os.mkdir(fruits12_cartoon_mask_folder)
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for c in CONFIG.FRUITS_12:
		img_file_list = np.loadtxt(exp1_2_trainlist_folder + c + 'imglist.txt', dtype=str, delimiter='|')
		num_img = len(img_file_list)
		for in_concept_idx, img_file_path in tqdm(enumerate(img_file_list)):
			if not c in all_masks.keys():
				all_masks[c] = np.zeros(( num_img , num_pixels )).astype(bool)
				os.mkdir(fruits12_cartoon_mask_folder + c + '-masked')
			img_file = img_file_path.split('/')[-1]
			img_file_path = exp1_2_datasets_img_folder + c + '/' + img_file
			mask, img_masked = generate_masks(img_file_path, target_size, mask_threshold)
			# mask = generate_masks(img_file_path, target_size, mask_threshold)
			all_masks[c][in_concept_idx] = mask.reshape(-1,)
			img_masked.save(fruits12_cartoon_mask_folder + c + '-masked/%d.png' % in_concept_idx)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12_cartoon, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


def exp1_3_fruits12_photo_masks(target_size, mask_threshold=5):
	# img_file_list = np.loadtxt(exp2_datasets_img_folder + 'train_veg.txt', dtype=str)[:, 0]
	if not os.path.exists(fruits12_photo_mask_folder):
		os.mkdir(fruits12_photo_mask_folder)
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for c in CONFIG.FRUITS_12:
		img_file_list = np.loadtxt(exp1_3_trainlist_folder + c + 'imglist.txt', dtype=str, delimiter='|')
		num_img = len(img_file_list)
		for in_concept_idx, img_file_path in tqdm(enumerate(img_file_list)):
			if not c in all_masks.keys():
				all_masks[c] = np.zeros(( num_img , num_pixels )).astype(bool)
				os.mkdir(fruits12_photo_mask_folder + c + '-masked')
			img_file = img_file_path.split('/')[-1]
			img_file_path = exp1_3_datasets_img_folder + c + '/' + img_file
			mask, img_masked = generate_masks(img_file_path, target_size, mask_threshold)
			# mask = generate_masks(img_file_path, target_size, mask_threshold)
			all_masks[c][in_concept_idx] = mask.reshape(-1,)
			img_masked.save(fruits12_photo_mask_folder + c + '-masked/%d.png' % in_concept_idx)
	with open(CONFIG.MASK_PATH_FOR_FRUITS_12_photo, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


def exp1_5_recycle6_masks(target_size, mask_threshold=5):
	if not os.path.exists(recycle6_mask_folder):
		os.mkdir(recycle6_mask_folder)
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for c in CONFIG.RECYCLE_6:
		img_file_list = np.loadtxt(exp1_5_trainlist_folder + c + 'imglist.txt', dtype=str)
		num_img = len(img_file_list)
		for in_concept_idx, img_file_path in tqdm(enumerate(img_file_list)):
			if not c in all_masks.keys():
				all_masks[c] = np.zeros(( num_img , num_pixels )).astype(bool)
				os.mkdir(recycle6_mask_folder + c + '-masked')
			img_file = img_file_path.split('/')[-1]
			img_file_path = exp1_5_datasets_img_folder + c + '/' + img_file
			mask, img_masked = generate_masks(img_file_path, target_size, mask_threshold)
			# mask = generate_masks(img_file_path, target_size, mask_threshold)
			all_masks[c][in_concept_idx] = mask.reshape(-1,)
			img_masked.save(recycle6_mask_folder + c + '-masked/%d.png' % in_concept_idx)
	with open(CONFIG.MASK_PATH_FOR_RECYCLE_6, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)
	

def exp2_1_vegetables5_masks(target_size, mask_threshold=5):
	img_file_list = np.loadtxt(exp2_datasets_img_folder + 'train_veg.txt', dtype=str)[:, 0]
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for img_file in tqdm(img_file_list):
		concept = img_file.split('/')[0]
		if not concept in all_masks.keys():
		# if not os.path.exists(vegetables5_mask_folder + concept):
			all_masks[concept] = np.zeros(( 50 , num_pixels )).astype(bool)
			in_concept_idx = 0
			# os.mkdir(vegetables5_mask_folder + concept + '-masked')
		# mask, img_masked = generate_masks(exp2_datasets_img_folder + img_file, target_size, mask_threshold)
		mask = generate_masks(exp2_datasets_img_folder + img_file, target_size, mask_threshold)
		all_masks[concept][in_concept_idx] = mask.reshape(-1,)
		# img_masked.save(vegetables5_mask_folder + concept + '-masked/%d.png' % in_concept_idx)
		in_concept_idx += 1
	with open(CONFIG.MASK_PATH_FOR_VEGETABLES_5, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


def exp2_2_fruits5_masks(target_size, mask_threshold=5):
	img_file_list = np.loadtxt(exp2_datasets_img_folder + 'train_5fruit.txt', dtype=str)[:, 0]
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for img_file in tqdm(img_file_list):
		concept = img_file.split('/')[0]
		if not concept in all_masks.keys():
			all_masks[concept] = np.zeros(( 50 , num_pixels )).astype(bool)
			in_concept_idx = 0
			# os.mkdir(fruits5_mask_folder + concept + '-masked')
		# mask, img_masked = generate_masks(exp2_datasets_img_folder + img_file, target_size, mask_threshold)
		mask = generate_masks(exp2_datasets_img_folder + img_file, target_size, mask_threshold)
		all_masks[concept][in_concept_idx] = mask.reshape(-1,)
		# img_masked.save(fruits5_mask_folder + concept + '-masked/%d.png' % in_concept_idx)
		in_concept_idx += 1
	with open(CONFIG.MASK_PATH_FOR_FRUITS_5, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


def exp11_masks(dataset, num_figs, target_size, mask_threshold=5):
	img_folder				= './datasets/%d_5ftadveg/' % num_figs
	masked_img_save_folder	= './datasets/%s-%d-masked/' % (dataset, num_figs)
	masks_save_path			= './datasets/%s-%d-masks.pkl' % (dataset, num_figs)
	if dataset == 'fruits5':		img_list_file = img_folder + 'fruit_train.txt'
	if dataset == 'vegetables5':	img_list_file = img_folder + 'veg_train.txt'
	os.mkdir(masked_img_save_folder)
	
	img_file_list = np.loadtxt(img_list_file, dtype=str)[:, 0]
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for img_file in tqdm(img_file_list):
		concept = img_file.split('/')[0]
		if not concept in all_masks.keys():
			all_masks[concept] = np.zeros(( num_figs, num_pixels )).astype(bool)
			in_concept_idx = 0
			os.mkdir(masked_img_save_folder + concept + '-masked')
		mask, img_masked = generate_masks(img_folder + img_file, target_size, mask_threshold)
		all_masks[concept][in_concept_idx] = mask.reshape(-1,)
		img_masked.save(masked_img_save_folder + concept + '-masked/%d.png' % in_concept_idx)
		in_concept_idx += 1
	with open(masks_save_path, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


def app_masks(dataset, target_size, mask_threshold=5):
	masked_img_save_folder	= './datasets/%s-masked/' % (dataset)
	masks_save_path			= './datasets/%s-masks.pkl' % (dataset)
	if dataset == 'fruits5_app1':
		concepts	= ['apple_red', 'banana', 'cherry', 'grape_green', 'peach']
		img_folder	= '/Users/codeb/Downloads/net_results/ours_10ftadveg/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'fruits5_app2':
		concepts	= ['apple_green', 'banana', 'cherry', 'grape_purple', 'peach']
		img_folder	= '/Users/codeb/Downloads/net_results/ours_10ftadveg/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'fruits5_100':
		concepts	= ['apple', 'banana', 'cherry', 'grape', 'peach']
		img_folder	= '/Users/codeb/Downloads/net_results/ours_fruits5_100/'
		img_order	= CONFIG.RESULT_ORDER3
		num_figs	= 100
	if dataset == 'brand':
		concepts	= ['Apple', 'ATT', 'HomeDepot', 'Kodak', 'Starbucks', 'Target', 'Yahoo']
		img_folder	= '/Users/codeb/Downloads/net_results/brand/dataset/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'abstract1':
		concepts	= ['Comfort', 'Efficiency', 'Reliability', 'Safety', 'Speed']
		# img_folder	= '/Users/codeb/Downloads/net_results/abstract/dataset/'
		img_folder	= './net_results/abstract/abstract1/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'abstract2':
		concepts	= ['Driving', 'Eating', 'Leisure', 'Sleeping', 'Working']
		# img_folder	= '/Users/codeb/Downloads/net_results/abstract/dataset/'
		img_folder	= './net_results/abstract/abstract2/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'drinks':
		concepts	= ['AW', 'Coca', 'Pepper', 'Pepsi', 'Sprite', 'Sunkist', 'Welch']
		# img_folder	= '/Users/codeb/Downloads/net_results/abstract/dataset/'
		img_folder	= './net_results/drinks/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'food':
		concepts	= ['Cheese', 'Cream', 'Lettuce', 'Onions', 'Potato', 'Steak', 'Tomato']
		# img_folder	= '/Users/codeb/Downloads/net_results/abstract/dataset/'
		img_folder	= './net_results/food/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50
	if dataset == 'emotion':
		concepts	= ['Calm', 'Disturbing', 'Exciting', 'Negative', 'Playful', 'Positive', 'Serious', 'Trustworthy']
		# img_folder	= '/Users/codeb/Downloads/net_results/abstract/dataset/'
		img_folder	= './net_results/emotion/'
		img_order	= CONFIG.RESULT_ORDER2
		num_figs	= 50

	os.mkdir(masked_img_save_folder)
	
	all_masks = {}
	num_pixels = target_size[0] * target_size[1]
	for concept in concepts:
		if not concept in all_masks.keys():
			all_masks[concept] = np.zeros(( num_figs, num_pixels )).astype(bool)
			in_concept_idx = 0
			os.mkdir(masked_img_save_folder + concept + '-masked')
		for img_id in tqdm(img_order):
			img_file_path = img_folder + concept + '/%d.jpg' % img_id
			mask, img_masked = generate_masks(img_file_path, target_size, mask_threshold)
			all_masks[concept][in_concept_idx] = mask.reshape(-1,)
			img_masked.save(masked_img_save_folder + concept + '-masked/%d.png' % in_concept_idx)
			in_concept_idx += 1
	with open(masks_save_path, 'wb') as pkl_file:
		pickle.dump(all_masks, pkl_file)


if __name__ == '__main__':
	target_size = (56, 56)
	mask_threshold = 5
	# exp1_1_fruits12_masks(target_size, mask_threshold)
	# exp1_2_fruits12_cartoon_masks(target_size, mask_threshold)
	# exp1_3_fruits12_photo_masks(target_size, mask_threshold)
	# exp1_5_recycle6_masks(target_size, mask_threshold)
	# exp2_1_vegetables5_masks(target_size, mask_threshold)
	# exp2_2_fruits5_masks(target_size, mask_threshold)

	# for dataset in ['fruits5', 'vegetables5']:
	# 	for num_figs in [100, 200]:
	# 		exp11_masks(dataset, num_figs, target_size, mask_threshold)

	### 2022-06-20
	# app_masks('fruits5_app1', target_size, mask_threshold)
	# app_masks('fruits5_app2', target_size, mask_threshold)
	# app_masks('fruits5_100', target_size, mask_threshold)
	### 2022-06-22
	# app_masks('brand', target_size, mask_threshold)
	# app_masks('abstract1', target_size, mask_threshold)
	# app_masks('abstract2', target_size, mask_threshold)
	# app_masks('drinks', target_size, mask_threshold)
	app_masks('food', target_size, mask_threshold)
	# app_masks('emotion', target_size, mask_threshold)
