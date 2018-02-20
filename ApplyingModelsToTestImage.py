from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import pickle
import nibabel as nib
from TestData import TestData
import os
import numpy as np
import nibabel as nib
from tf_unet import unet, util, image_util

path = '/media/data1/artin/data/Cell/'
TrainPath = path + 'Data/train/'
TestPath  = path + 'Data/test/'

net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2)


padSize = 90
[AllImage , AllImage_logical] = TestData(net , TestPath , TrainPath , padSize)
