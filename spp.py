#library dependencies

from keras .layers import MaxPool2D , Flatten , merge
import numpy as np 


class Spatial_Pyramidal_Pooling_Layer():

	def spatial_pooling_2D(input_tensor,pooling_bins):
		pool_list=[]
		shape=input_tensor.get_shape().as_list()
		for level in pooling_bins:
			window_height=int(np.ceil(int(shape[1])/level))
			window_width=int(np.ceil(int(shape[2])/level))
			stride_height=int(np.floor(int(shape[1])/level))
			stride_width=int(np.floor(int(shape[2])/level))
			pooling=MaxPool2D(pool_size=(window_height,window_width),strides=(stride_height,stride_width))(input_tensor)
			pool_list.append(Flatten()(pooling))
		output_tensor=merge(pool_list,mode='concat',concat_axis=-1)
		return(output_tensor)

