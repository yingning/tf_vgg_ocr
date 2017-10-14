#-*-coding:utf8 -*-

__author__ = "buyizhiyou"
__datetime__ = "2017-10-12"

import tensorflow as tf 
import numpy as vgg16_npy_path
import os
import time 
import pdb

'''
vgg 反复堆叠3*3的卷积和2×2的池化 
'''
class Vgg16:
	def __init__(self):
		pass

	def build(self,input):

		#第一段，两个卷积层的卷积核都是3*3,卷积核的数目都是64,步长都是1*1,input.shape=(32,224,224,1)
		self.conv1_1 = self.conv_layer(input,kh=3,kw=3,kernel_num=64,dh=1,dw=1,name='conv1_1')#shape:(?,224,224,64)
		self.conv1_2 = self.conv_layer(self.conv1_1,kh=3,kw=3,kernel_num=64,dh=1,dw=1,name='conv1_2')#shape:(?,224,224,64)
		self.pool1 = self.pool_layer(self.conv1_2,name='pool1')	#shape:(?,112,112,64)
		#第二段
		self.conv2_1 = self.conv_layer(self.pool1,kh=3,kw=3,kernel_num=128,dh=1,dw=1,name='conv2_1')#shape:(?,112,112,128)
		self.conv2_2 = self.conv_layer(self.conv2_1,kh=3,kw=3,kernel_num=128,dh=1,dw=1,name='conv2_2')#shape:(?,112,112,128)
		self.pool2 = self.pool_layer(self.conv2_2,name='pool2')#shape:(?,56,56,128)
		#第三段
		self.conv3_1 = self.conv_layer(self.pool2,kh=3,kw=3,kernel_num=256,dh=1,dw=1,name='conv3_1')#shape:(?,56,56,256)
		self.conv3_2 = self.conv_layer(self.conv3_1,kh=3,kw=3,kernel_num=256,dh=1,dw=1,name='conv3_2')#shape:(?,56,56,256)
		self.conv3_3 = self.conv_layer(self.conv3_2,kh=3,kw=3,kernel_num=256,dh=1,dw=1,name='conv3_3')#shape:(?,56,56,256)
		self.pool3 = self.pool_layer(self.conv3_3,name='pool3')#(?,28,28,256)
		#第四段
		self.conv4_1 = self.conv_layer(self.pool3,kh=3,kw=3,kernel_num=512,dh=1,dw=1,name='conv4_1')#shape:(?,28,28,512)
		self.conv4_2 = self.conv_layer(self.conv4_1,kh=3,kw=3,kernel_num=512,dh=1,dw=1,name='conv4_2')#shape:(?,28,28,512)
		self.conv4_3 = self.conv_layer(self.conv4_2,kh=3,kw=3,kernel_num=512,dh=1,dw=1,name='conv4_3')#shape:(?,28,28,512)
		self.pool4 = self.pool_layer(self.conv4_3,name='pool4')#shape:(?,14,14,512)
		#第五段
		self.conv5_1 = self.conv_layer(self.pool4,kh=3,kw=3,kernel_num=512,dh=1,dw=1,name='conv5_1')#shape:(?,14,14,512)
		self.conv5_2 = self.conv_layer(self.conv5_1,kh=3,kw=3,kernel_num=512,dh=1,dw=1,name='conv5_2')#shape:(?,14,14,512)
		self.conv5_3 = self.conv_layer(self.conv5_2,kh=3,kw=3,kernel_num=512,dh=1,dw=1,name='conv5_3')#shape:(?,14,14,512)
		self.pool5 = self.pool_layer(self.conv5_3,name='pool5')#shape:(?,7,7,512)
		#第六段
		shape = self.pool5.get_shape()
		flattened_shape = shape[1].value*shape[2].value*shape[3].value#7*7*512=25088
		self.reshape_data = tf.reshape(self.pool5,[-1,flattened_shape],name='reshape')
		self.fc6 = self.fc_layer(self.reshape_data,kernel_num=4096,name='fc6')#shape:(?,4096)
		self.fc6_drop = tf.nn.dropout(self.fc6,keep_prob=0.5,name='fc6_drop')
		#第七段
		self.fc7 = self.fc_layer(self.fc6_drop,kernel_num=4096,name='fc7')#shape:(?,4096)
		self.fc7_drop = tf.nn.dropout(self.fc7,keep_prob=0.5,name='fc7_drop')
		#第八段
		self.fc8 = self.fc_layer(self.fc7_drop,kernel_num=607,name='fc8')#shape:(?,607)
		self.softmax = tf.nn.softmax(self.fc8)#shape:(607,)
		self.pred= tf.argmax(self.softmax, 1) 
		self.pred = tf.to_float(self.pred, name='ToFloat')

		return self.softmax

	def conv_layer(self,bottom,kh,kw,kernel_num,dh,dw,name):

		n_in = bottom.get_shape()[-1].value#获取输入的通道数
		with tf.name_scope(name) as scope:
			kernel = tf.Variable(tf.truncated_normal([kh,kw,n_in,kernel_num],dtype=tf.float32,stddev=1e-1),name=name+'weight')
			conv = tf.nn.conv2d(bottom,kernel,(1,dh,dw,1),padding='SAME')
			biases = tf.Variable(tf.constant(0.0,shape=[kernel_num],dtype=tf.float32),trainable=True,name=name+'bias')
			z = tf.nn.bias_add(conv,biases)
			activation = tf.nn.relu(z)

			return activation

	def pool_layer(self,bottom,name):

		return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=None)

	def fc_layer(self,bottom,kernel_num,name):

		with tf.name_scope(name) as scope:
			shape = bottom.get_shape().as_list()
			dim = 1
			for i in shape[1:]:
				dim *= i
			x = tf.reshape(bottom,[-1,dim])
			kernel = tf.Variable(tf.truncated_normal([dim,kernel_num],dtype=tf.float32,stddev=1e-1),name=name+'weights')
			biases = tf.Variable(tf.constant(1.0,shape=[kernel_num],dtype=tf.float32),trainable=True,name=name+'bias')			
			z = tf.nn.bias_add(tf.matmul(x,kernel),biases)
			activation = tf.nn.relu(z)

			return activation


