# library dependencies

from keras.layers import Input, Conv2D , MaxPool2D , Dense , Dropout ,Lambda
from spp import Spatial_Pyramidal_Pooling_Layer
from keras.models import Model 

# the architecture

input_layer=Input(shape=(224,224,3),name='Input_Image')
conv_1=Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_1',activation='relu')(input_layer)
conv_2=Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_2',activation='relu')(conv_1)
pool_layer_1=MaxPool2D(pool_size=(2,2),strides=(2,2),name='pooling_1',data_format='channels_last')(conv_2)
conv_3=Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_3',activation='relu')(pool_layer_1)
conv_4=Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_4',activation='relu')(conv_3)
pool_layer_2=MaxPool2D(pool_size=(2,2),strides=(2,2),name='pooling_2',data_format='channels_last')(conv_4)
conv_5=Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_5',activation='relu')(pool_layer_2)
conv_6=Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_6',activation='relu')(conv_5)
conv_7=Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_7',activation='relu')(conv_6)
conv_8=Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_8',activation='relu')(conv_7)
conv_9=Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_9',activation='relu')(conv_8)
conv_10=Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_10',activation='relu')(conv_9)
conv_11=Conv2D(filters=512,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_11',activation='relu')(conv_10)
conv_12=Conv2D(filters=512,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_12',activation='relu')(conv_11)
conv_13=Conv2D(filters=512,kernel_size=(3,3),padding='same',strides=(1,1),data_format='channels_last',name='convolution_layer_13',activation='relu')(conv_12)
spatial_pool=Lambda(Spatial_Pyramidal_Pooling_Layer.spatial_pooling_2D,arguments={'pooling_bins':[6,4,3,1]})(conv_13)
#print("spatial pooling ", spatial_pool.shape)
fully_connected_1=Dense(4096,activation='relu')(spatial_pool)
Dropout_layer_1=Dropout(0.5,name='Dropout_Layer_1')(fully_connected_1)
fully_connected_2=Dense(4096,activation='relu')(conv_13)
Dropout_layer_2=Dropout(0.5,name='Dropout_layer_2')(fully_connected_2)
output_layer=Dense(604,activation='sigmoid')(Dropout_layer_2)

model=Model(inputs=input_layer,outputs=conv_13)
model.summary()


