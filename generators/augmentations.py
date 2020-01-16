#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:14:36 2019

@author: stephan


Augmentations to help generalize training routine

"""
from scripts.constants import SEED
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
ia.seed(SEED)


def apply_augmentation(img):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    
    # channel invariant augmentations
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Flipud(0.5),
        sometimes(iaa.Rot90((1, 3)))
    ], random_order=True)
    
    # RGB dependent augmentations
    seq2 = sometimes(
            iaa.SomeOf((1,4),[
                iaa.AddToHue((-8,8)),
                iaa.AddToSaturation((-10,10)),
                iaa.Multiply((0.90, 1.25)),
                iaa.LinearContrast((0.90, 1.3))
                ], random_order=True)
    )
    
    img = seq(image=img)
    img2 = np.array(img[:,:,0:3] * 255, dtype=np.uint8)
    img2 = seq2(image=img2)
    img2 = np.array(img2/255, dtype=np.float32)
    img[:,:,0:3] = img2
    #print(img)
    return img
