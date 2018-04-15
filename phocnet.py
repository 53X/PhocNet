#library dependencies

import tensorflow as tf 


# convolutional layer for the PHOCNET architecture


def convolutional_layer(input_tensor,filter_length,filter_width,num_filters,strides=[1,1,1,1],pooling=True):

	input_channels=input_tensor.get_shape().as_list()[-1]

	#Filter Weights and biases

	weights_for_layer=tf.Variable(tf.truncated_normal(shape=[filter_length,filter_width,input_channels,num_filters],mean=0.4),tf.float32)
	biases_for_layer=tf.Variable(tf.constant(0.1,shape=[num_filters]),tf.float32)

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


#  Fully Connected Layer for the PHOCNET architecture


def fully_connected_layer(input_tensor,output_dimensions,activation='relu'):

	input_dimensions=input_tensor.get_shape().as_list()[-1]
	
	#weights and biases

	weights_for_layer=tf.Variable(tf.truncated_normal(0.5,shape=[input_dimensions,output_dimensions]),tf.float32)
	biases_for_layer=tf.Variable(tf.constant(0.1,shape=[output_dimensions]),tf.float32) 

	#FC Layer operation

	final_tensor=tf.matmul(input_tensor,weights_for_layer)
	final_tensor=tf.add(final_tensor,biases_for_layer)

	return final_tensor


# Flatten layer for the PHOCNET Architecture

def flatten_layer(input_tensor):

	shape=input_tensor.get_shape().as_list()
	num_features=shape[1]*shape[2]*shape[3]

	#flatten operation

	flattened_tensor=tf.reshape(input_tensor,shape=[-1,num_features])

	return(flattened_tensor)




	