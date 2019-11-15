#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#suppress information messages
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
#tf.compat.v1.logging.set_verbosity(2)

from tensorflow.keras.layers import Conv2D, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50


def build_resnet50(train_model=True, **kwargs): 
	inputs = Input(shape=(None, None, len(kwargs.get('channels')) ))
	resnet = ResNet50(include_top=False, weights=None, input_tensor=inputs)
	
	x = resnet.output
	x = Conv2D(filters = 512, kernel_size=(4,4), activation='relu')(x)
	x = Conv2D(filters = 128, kernel_size=(1,1), activation='relu')(x)
	preds = Conv2D(filters = kwargs.get('num_classes'), kernel_size=(1,1), activation='softmax')(x)
	if train_model:
		preds = Flatten()(preds)

	#preds = Dense(2, activation='softmax', name='fc1000')(x)
	
	model = Model(inputs=resnet.input, outputs=preds)
	#model.summary()
	
	return model

def build_resnet50_imagenet(train_model = True, **kwargs):
    weights = kwargs.get('weights', 'imagenet')
    #num_classes = kwargs.get('num_classes')
    
    inputs = Input(shape=(None, None, len(kwargs.get('channels')) ))
    dense_filter = Conv2D(filters=3, kernel_size=(3,3), padding='same')(inputs)

    densenet = ResNet50(include_top=False, weights=weights)(dense_filter)
    #x = densenet.output
    x = Conv2D(filters=128,kernel_size=(4,4),activation='relu')(densenet) # 8
    x = Conv2D(filters=64,kernel_size=(1,1),activation='relu')(x)
    preds = Conv2D(filters=kwargs.get('num_classes'), kernel_size=(1,1),activation='softmax')(x) 
    if train_model:
        preds = Flatten()(preds)
    model = Model(inputs=inputs, outputs=preds)

    #model.summary()
    #model.summary()
    return model
