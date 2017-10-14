#-*-coding:utf8 -*-

import cv2
import numpy as np
import pdb

n_classes = 607

def load_data(path,txt_path):

	x = []
	y = []
	with open(txt_path,'r') as f:
		lines = f.readlines()
		for line in lines:
			filename = line.split(' ')[0]
			filepath = path + filename
			label = int(line.split(' ')[1])
			im = cv2.imread(filepath,0)
			im = np.expand_dims(im,2)
			print(im.shape)
			#pdb.set_trace()
			x.append(im)
			tmp = np.zeros(n_classes)
			tmp[label] = 1
			y.append(tmp)
	np.savez('./data/train.npz',x=x,y=y)


path = './data/train/'
txt_path ='./data/train_label.txt'
load_data(path,txt_path)
