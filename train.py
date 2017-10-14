#-*-coding:utf8 -*-

from vgg16 import Vgg16
import random
import numpy as np
import tensorflow as tf
import pdb

#超参数设置
learning_rate = 0.001
max_steps = 20000
batch_size = 32
val_step = 100
n_classes = 607
save_step = 1000


def train(train_data_path,val_data_path):
	with tf.Session() as sess:
		#加载数据
		train_data = np.load(train_data_path)
		train_x = train_data['x']
		train_y = train_data['y']
		val_data = np.load(val_data_path)
		val_x = val_data['x']
		val_y = val_data['y']

		x = tf.placeholder(tf.float32,[None,224,224,1])
		y = tf.placeholder(tf.float32,[None,n_classes])

		#构建模型
		model = Vgg16()
		#pdb.set_trace()
		pred = model.build(x)
		#定义损失函数和学习器
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		#测试网络
		correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
		#初始化所有共享变量
		sess.run(tf.global_variables_initializer()) 
		#保存模型
		saver = tf.train.Saver(max_to_keep=5)#保存最近的5个模型

		step=1 
		print("===============Begin Training===================")
		#参数可视化
		tf.summary.scalar('loss', cost)
		tf.summary.scalar('accuracy', accuracy)
		merged_summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter('vgg_log')

		while step < max_steps:
			batch_x = train_x[batch_size*(step-1):batch_size*step]
			batch_y = train_y[batch_size*(step-1):batch_size*step]
			sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
			print('Training step:%d'%(step))
			step += 1

			if step%val_step ==0:
				print("===============Valiation=================")
				batch_x = val_x[batch_size*(step-1):batch_size*step]
				batch_y = val_y[batch_size*(step-1):batch_size*step]

				accu = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
				loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
				summary_str = sess.run(merged_summary_op,feed_dict={x:batch_x,y:batch_y})
				print('Val Iter %s ,Minibatch Loss= %.6f ,Training Accuracy= %.5f'%(str(step),loss,accu))
				summary_writer.add_summary(summary_str,step)
			if step%save_step ==0:
				saver.save(sess,'model/vgg16.ckpt',global_step=step)
		print("Optimization Finished!")

train_data_path='./data/train.npz'
val_data_path = './data/val.npz'
train(train_data_path,val_data_path)