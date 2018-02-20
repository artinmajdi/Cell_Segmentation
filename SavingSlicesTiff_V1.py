import nifti
import numpy as np
import pylab as p
import matplotlib.pyplot as plt
import Image
import os
import nibabel as nib
import tifffile
import pickle
Directory = '/media/data1/artin/data/Thalamus/OriginalData/'



subFolders = os.listdir(Directory)

subFolders2 = []
i = 0
for o in range(len(subFolders)-1):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;
print subFolders2
subFolders = subFolders2
with open(Directory+"subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

# with open(Directory+"subFolderList" ,"rb") as fp:
#     ssff2 = pickle.load(fp)



# mask = nib.load(Directory+subFolders2[0]+'/WMnMPRAGEdeformed.nii.gz')
# tifffile.imsave(Directory+subFolders2[0]+'/WMnMPRAGEdeformed.tiff',mask.get_data()[:,:,50])

i = 0
SlideNumbers = range(29,39)
# SlideNumbers = range(26,46)

# SlideNumbers = range(21,47)

for p in range(len(subFolders)):

    for sFi in range(len(subFolders)):
        mask = nib.load(Directory+subFolders2[sFi]+'/ThalamusSegDeformed_Croped.nii.gz')
        im = nib.load(Directory+subFolders2[sFi]+'/WMnMPRAGEdeformed_Croped.nii.gz')

        imD = im.get_data()
        maskD = mask.get_data()

        imD = imD[1:,41:40+93,SlideNumbers]
        maskD = maskD[1:,41:40+93,SlideNumbers]

        imD_padded = np.pad(imD,((20,20),(20,20),(0,0)),'constant')
        maskD_padded = np.pad(maskD,((20,20),(20,20),(0,0)),'constant')

        # print imD.shape
        # imD = np.reshape(imD,[572,572,10])

        #     os.makedirs(SaveDirectorySegment)
        if sFi == p:
            SaveDirectoryImage = Directory+'../ForUnet_Test3/TestSubject'+str(p)+'/test/'
        else:
            SaveDirectoryImage = Directory+'../ForUnet_Test3/TestSubject'+str(p)+'/train/'
        # SaveDirectorySegment = Directory+'ForUnet/Segments/'+subFolders2[sFi]+'/'

        try:
            os.stat(SaveDirectoryImage)
        except:
            os.makedirs(SaveDirectoryImage)

        # try:
        #     os.stat(SaveDirectoryImage+'ToLook/')
        # except:
        #     os.makedirs(SaveDirectoryImage+'ToLook/')


        for sliceInd in range(imD_padded.shape[2]):

            tifffile.imsave(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SlideNumbers[sliceInd])+'.tif',imD_padded[:,:,sliceInd])
            tifffile.imsave(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SlideNumbers[sliceInd])+'_mask.tif',maskD_padded[:,:,sliceInd])
            # if sFi == 7:
            #     tifffile.imsave(SaveDirectoryImage+'ToLook/'+subFolders2[sFi]+'_slice'+str(sliceInd)+'.tif',imD[:,:,sliceInd])
            #     tifffile.imsave(SaveDirectoryImage+'ToLook/'+subFolders2[sFi]+'_slice'+str(sliceInd)+'_mask.tif',maskD[:,:,sliceInd])
