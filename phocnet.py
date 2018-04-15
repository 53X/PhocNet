#library dependencies

import tensorflow as tf 


# convolutional layer for the PHOCNET architecture


def convolutional_layer(input_tensor,filter_length,filter_width,num_filters,strides=[1,1,1,1],pooling=True):

	input_shape=input_tensor.shape
	input_channels=int(input_shape[-1])

	#Filter Weights and biases

	weights_for_layer=tf.Variable(tf.truncated_normal(shape=[filter_length,filter_width,input_channels,num_filters],mean=0.4))
	biases_for_layer=tf.Variable(tf.constant(0.1,shape=[num_filters]))

	#applying the convolution operation

	conv_tensor=tf.nn.conv2d(input=input_tensor,filter=weights_for_layer,padding='SAME',strides=strides)
	conv_tensor=tf.add(conv_tensor,biases_for_layer)
	conv_tensor=tf.nn.relu(conv_tensor)

	#Pooling operation

	if(pooling==False):

		return conv_tensor
	else:
	
		conv_tensor=tf.nn.max_pool(conv_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

		return conv_tensor	
