#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:19:53 2019

@author: stephan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""
from tensorflow.keras.layers import Conv2D, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121

def build_densenet121_old(train_model = True, **kwargs):
	weights = kwargs.get('weights', 'imagenet')
	#num_classes = kwargs.get('num_classes')
	
	inputs = Input(shape=(None, None, len(kwargs.get('channels')) ))
	densenet = DenseNet121(include_top=False, input_tensor=inputs , weights=weights)
	x = densenet.output
	x = Conv2D(filters=128,kernel_size=(4,4),activation='relu')(x) # 8
	x = Conv2D(filters=64,kernel_size=(1,1),activation='relu')(x)
	preds = Conv2D(filters=2, kernel_size=(1,1),activation='softmax')(x) 
	if train_model:
		preds = Flatten()(preds)
		
	model = Model(inputs=densenet.input, outputs=preds)
	

	#model.summary()
	model.summary()
	return model

def build_densenet121(train_model = True, **kwargs):
    weights = kwargs.get('weights', 'imagenet')
    #num_classes = kwargs.get('num_classes')
    
    inputs = Input(shape=(None, None, len(kwargs.get('channels')) ))
    dense_filter = Conv2D(filters=3, kernel_size=(3,3), padding='same')(inputs)

    densenet = DenseNet121(include_top=False, weights=weights)(dense_filter)
    #x = densenet.output
    x = Conv2D(filters=128,kernel_size=(4,4),activation='relu')(densenet) # 8
    x = Conv2D(filters=64,kernel_size=(1,1),activation='relu')(x)
    preds = Conv2D(filters=kwargs.get('num_classes'), kernel_size=(1,1),activation='softmax')(x) 
    if train_model:
        preds = Flatten()(preds)
    model = Model(inputs=inputs, outputs=preds)

    #model.summary()
    model.summary()
    return model