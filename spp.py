# library dependencies

from keras.layers import MaxPool2D 
from keras.layers import  Flatten
from keras.layers import concatenate
import numpy as np 
from numpy import ceil
from numpy import floor


class Spatial_Pyramidal_Pooling:

	def Pooling2D(input_tensor, pool_levels):
		shape=input_tensor.shape
		individual_pool_tensors=[]
		for n in pool_levels:
			window_height=ceil(int(shape[1])/n)
			window_width=ceil(int(shape[2])/n)
			stride_height=floor(int(shape[1])/n)
			stride_width=floor(int(shape[2])/n)
			level_pool=MaxPool2D(kernel_size=(window_height,window_width),strides=(stride_height,stride_width),name='pool_level_'+str(n))(input_tensor)
			flattened_level_pool=Flatten(name='flattening_level_'+str(n))(level_pool)
			individual_pool_tensors.append(flattened_level_pool)
		output_tensor=concatenate(individual_pool_tensors,axis=-1)
		return(output_tensor)





