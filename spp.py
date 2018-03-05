# library dependencies

from keras.layers import MaxPool2D 
import keras.backend as K
import numpy as np 


class Spatial_Pyramidal_Pooling:

	def Pooling2D(input_tensor, pool_levels):
		pool_list=[]
		shape=input_tensor.shape
		for level in pool_levels:
			window_height=np.ceil(int(shape[1])/level)
			window_width=np.ceil(int(shape[2])/level)
			stride_height=np.floor(int(shape[1])/level)
			stride_width=np.floor(int(shape[2])/level)
			pooling=MaxPool2D(pool_size=(window_height,window_width),strides=(stride_height,stride_width))(input_tensor)
			flattened_tensor=K.flatten(pooling)
			pool_list.append(flattened_tensor)
		output_tensor=K.concatenate(pool_list,axis=-1)
		return(output_tensor)

