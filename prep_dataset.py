# from __future__ import division

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
from sklearn.preprocessing import OneHotEncoder
# from skimage.io import imread
from scipy.misc import imresize
from skimage.external.tifffile import imread , imsave
import matplotlib.pylab as plt
from PIL import Image

path = '/media/data1/artin/data/Cell/'
try:
    os.stat(path + 'Data/train/')
except:
    os.makedirs(path + 'Data/train/')

try:
    os.stat(path + 'Data/test/')
except:
    os.makedirs(path + 'Data/test/')


def load_image(path):
    im = tifffile.imread(path)
    im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
    return im

def saveImages(TestImageNum,padSizeFull):

    padSize = padSizeFull/2
    for imgNum in range(1,9):
        print imgNum

        if imgNum == TestImageNum:
            TestTrainFlag = 'test'
        else:
            TestTrainFlag = 'train'

        img  = load_image( path + 'Composite/img'+str(imgNum)+'.tif' )
        mask = load_image( path + 'manuals/'+str(imgNum)+'.tif' )
        # imgNum = 1
        # img  = Image.open( path + 'Composite/img'+str(imgNum)+'.tif' )
        # mask = Image.open( path + 'manuals/'+str(imgNum)+'.tif' )

        img = img[:1924,:1924]
        mask = mask[:1924,:1924]

        for i in range(int(img.shape[0]/148)):
            for j in range(int(img.shape[0]/148)):
                im = img[ 148*i:148*(i+1) , 148*j:148*(j+1) ]
                msk = mask[ 148*i:148*(i+1) , 148*j:148*(j+1) ]
                msk[msk<200] = 0

                im = np.pad(im,((padSize,padSize),(padSize,padSize)),'constant' ) #
                msk = np.pad(msk,((padSize,padSize),(padSize,padSize)),'constant' ) # , constant_values=(5)

                # imsave(path + 'Data/' + TestTrainFlag + '/img'+str(imgNum)+'_'+str(i)+'_'+str(j)+'.tif',im )
                # imsave(path + 'Data/' + TestTrainFlag + '/img'+str(imgNum)+'_'+str(i)+'_'+str(j)+'_mask.tif',msk )

                # tifffile.imsave(path + 'Data/' + TestTrainFlag + '/img'+str(imgNum)+'_'+str(i)+'_'+str(j)+'.tif',im )
                # tifffile.imsave(path + 'Data/' + TestTrainFlag + '/img'+str(imgNum)+'_'+str(i)+'_'+str(j)+'_mask.tif',msk )

                im = Image.fromarray(np.uint8(im))
                msk = Image.fromarray(np.uint8(msk))

                im.save(path + 'Data/' + TestTrainFlag + '/img'+str(imgNum)+'_'+str(i)+'_'+str(j)+'.tif')
                msk.save(path + 'Data/' + TestTrainFlag + '/img'+str(imgNum)+'_'+str(i)+'_'+str(j)+'_mask.tif')


padSizeFull = 90
TestImageNum = 7

saveImages(TestImageNum,padSizeFull)

imm = Image.open(path + 'Data/train/img1_1_6_mask.tif')
plt.imshow(imm,cmap='gray')
plt.show()
