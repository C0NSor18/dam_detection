#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Input
from tensorflow.keras.models import Model



# TODO: target size with variable channels
# TODO: variable number of classes

# TEMPORARY MODEL FOR TESTING PURPOSES ONLY


def build_dilated_fcn_61(train_model = True, **kwargs): 
    ## DEFINE THE ABOVE DESCRIBED MODEL HERE
    x_in = Input(batch_shape=(None, None, None, len(kwargs.get('channels')) ))
    x = Conv2D(filters=32,kernel_size=(7,7),activation='relu')(x_in) # 55, 55
    x = Conv2D(filters=32,kernel_size=(7,7), dilation_rate = 2, activation='relu')(x) # 43, 43
    x = Conv2D(filters=32,kernel_size=(5,5), activation='relu')(x) # 37, 37
    x = Conv2D(filters=32,kernel_size=(5,5), dilation_rate = 2, activation='relu')(x) # 31, 31
    x = Conv2D(filters=64,kernel_size=(5,5), activation='relu')(x) # 27, 27
    x = Conv2D(filters=64,kernel_size=(5,5), dilation_rate = 2, activation='relu')(x) # 19, 19
    x = Conv2D(filters=64,kernel_size=(3,3), activation='relu')(x) # 17, 17
    x = Conv2D(filters=64,kernel_size=(3,3), dilation_rate = 2, activation='relu')(x) # 13, 13
    x = Conv2D(filters=64,kernel_size=(3,3), activation='relu')(x) # 11, 11
    x = Conv2D(filters=64,kernel_size=(3,3), dilation_rate = 2, activation='relu')(x) # 7, 7
    x = Conv2D(filters=128,kernel_size=(3,3), activation='relu')(x) # 5, 5
    x = Conv2D(filters=128,kernel_size=(3,3), activation='relu')(x) # 3, 3
    x = Conv2D(filters=64,kernel_size=(3,3), activation='relu')(x) # 1, 1
    x = Conv2D(filters=64,kernel_size=(1,1), activation='relu')(x)
    x_out = Conv2D(filters=kwargs.get('num_classes'),kernel_size=(1,1),activation='softmax')(x)
    if train_model:
        x_out = Flatten()(x_out)
    model = Model(inputs = x_in, outputs=x_out)

    #model.summary()
    return model
