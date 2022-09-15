#coding:utf-8
import glob
import numpy as np
import os
import skimage.color as color
import imageio
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
import caffe
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--prototxt',dest='prototxt',help='prototxt filepath', type=str, default='/mnt/d/ziqi/vis/cg/colorization-caffe/models/colorization_deploy_v2.prototxt')
    parser.add_argument('--caffemodel',dest='caffemodel',help='caffemodel filepath', type=str, default='/mnt/d/ziqi/vis/bs/colorization-caffe/train/model_thirdfood/colornet_iter_10000.caffemodel')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_args()

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu)

	# Select desired model
	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
	(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
	# input image size(224,224) of network
	(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
	# output image size(56,56) of network 
	pts_in_hull = np.load('./resources/pts_in_hull.npy') # load cluster centers
	net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
	fruit=['Cream','Cheese','Steak','Lettuce','Onions','Potato','Tomato']
	#you can use other concept you like
	for j in range(len(fruit)):      
		imgpath='./dataset'+ fruit[j]		
		t = open('./result/'+fruit[j]+'imglist.txt', 'a')#output image list 
		all_lung_image = os.listdir(imgpath)
		all_lung_image=sorted(all_lung_image)
		for i in range(len(all_lung_image)):
			all_lung_image[i]=imgpath+'/'+all_lung_image[i]
			t.write(all_lung_image[i]+'\n')
			img_rgb = caffe.io.load_image(all_lung_image[i])
			img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
			img_l = img_lab[:,:,0] # pull out L channel,0L1A2B
			(H_orig,W_orig) = img_rgb.shape[:2] # original image size
			# create grayscale version of image (just for displaying)
			img_lab_bw = img_lab.copy()
			img_lab_bw[:,:,1:] = 0
			img_rgb_bw = color.lab2rgb(img_lab_bw)

			# resize image to network input size
			img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size 256×256
			img_lab_rs = color.rgb2lab(img_rs)
			img_l_rs = img_lab_rs[:,:,0]
			net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering，input Image

			net.forward() # run network
			pp = np.load('./pts_in_hull.npy')
            #input L A B probability to fruit[j] image i pixel 56×56
			L_dec_us = sni.zoom(img_l_rs,(1.*H_out/H_in,1.*W_out/W_in)) # 56×56 L channel  upsample to match size of original image L (56, 56, 1)  
			yy = net.blobs['class8_313_rh'].data[0,:,:,:].transpose((1,2,0))    #change probability ab 313×56×56 to 56×56×313
			print(str(H_out),str(W_out))


			L_flaten=L_dec_us.flatten().transpose()
			yy=np.array(yy)
			ab_pro=yy.reshape(H_out*H_out,313)
			L_flaten=np.array(L_flaten)
			np.save("./result/"+fruit[j]+'/L_'+str(i)+'.npy', L_flaten)
			np.save("./result/"+fruit[j]+'/pro_'+str(i)+'.npy', ab_pro)
			print(fruit[j]+' '+str(i)+'th image done')    


			# get colorized image from the network

			# ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result       
			# ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L   
			# img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
			# img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb						
			# isave = './result/coloried_img/' + fruit[j] +  s+'.png'
			# plt.imsave(isave, img_rgb_out)

