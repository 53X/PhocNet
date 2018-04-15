#library dependencies

import tensorflow as tf
import numpy as np
from phocnet import flatten_layer 

#Spatial Pyramidal Pooling Layer


class Spatial_Pyramidal_Pooling_Layer():

	def spp_layer(input_tensor,pooling_bins):

		input_shape=input_tensor.get_shape().as_list()
		pool_list=[]

		for levels in pooling_bins:

			window_height=int(np.ceil(input_shape[1]/levels))
			window_width=int(np.ceil(input_shape[2]/levels))
			stride_height=int(np.floor(input_shape[1]/levels))
			stride_width=int(np.floor(input_shape[2]/levels))

			pool_tensor=tf.nn.max_pool(input_tensor,ksize=[1,window_height,window_width,1],strides=[1,stride_height,stride_width,1],padding='VALID')
			flattened_tensor=flatten_layer(pool_tensor)
			pool_list.append(flattened_tensor)

		spp_out=tf.concat(pool_list,axis=-1)

		return(spp_out)	




	