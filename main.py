from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData import TestData


def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    return intersection.sum()*2/(msk1.sum()+msk2.sum())

path = '/media/data1/artin/data/Cell/'
TrainPath = f'{path}Data/train/'
TestPath = f'{path}Data/test/'
Trained_Model_Path = f'{TrainPath}model/'

TrainData = image_util.ImageDataProvider(f"{TrainPath}*.tif", n_class=2)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True ) #  , cost="dice_coefficient"

trainer = unet.Trainer(net)
path = trainer.train(TrainData, Trained_Model_Path, training_iters=50, epochs=50, display_step=25) #, training_iters=100, epochs=100, display_step=10

# tensorboard --logdir=~/artin/data/Thalamus/ForUnet_Test2_IncreasingNumSlices/TestSubject0/train/model

padSize = 90
[AllImage , AllImage_logical] = TestData(net , TestPath , TrainPath , padSize)
