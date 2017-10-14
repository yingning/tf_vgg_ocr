#-*-coding:utf8 -*-


from vgg16 import Vgg16
import random
import numpy as np
import tensorflow as tf
import pdb

#超参数设置
learning_rate = 0.001
max_steps = 20
batch_size = 32
val_step = 10
n_classes = 607
save_step = 15


def test(test_data_path):
	with tf.Session() as sess:
		#加载数据
		test_data = np.load(test_data_path)
		test_x = test_data['x']
		test_y = test_data['y']

		x = tf.placeholder(tf.float32,[None,224,224,1])
		y = tf.placeholder(tf.float32,[None,n_classes])

		#构建模型
		model = Vgg16()
		pred = model.build(x)
		#测试网络
		correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
		#初始化所有共享变量
		sess.run(tf.global_variables_initializer()) 
		#保存模型
		saver = tf.train.Saver()

		print("===============Testing===================")
		saver.restore(sess,'model/...')
		batch_x = test_x
		batch_y = test_y
		accu = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
		loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
		print('Testing Accuracy= %.5f'%(accu))


test_data_path='./data/test.npz'
train(test_data_path)