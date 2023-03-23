import os
import numpy as np
import nibabel as nib
from tf_unet import unet, util, image_util
from PIL import Image

eps = np.finfo(float).eps

def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    return intersection.sum()*2/(msk1.sum()+msk2.sum())


def TestData(net , Test_Path , Train_Path , padSize):

    TestImageNum = 7

    Trained_Model_Path = f'{Train_Path}model/model.cpkt'
    TestResults_Path = f'{Test_Path}results/'

    try:
        os.stat(TestResults_Path)
    except:
        os.makedirs(TestResults_Path)

    AllImage_logical = np.zeros((1924,1924))
    AllImage = np.zeros((1924,1924))

    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(
        f'{Test_Path}*.tif', shuffle_data=False
    )

    L = len(TestData.data_files)
    DiceCoefficient  = np.zeros(L)
    LogLoss  = np.zeros(L)
    # BB_Cord = np.zeros(L,3)
    BB_Cord = np.zeros((L,2))


    aa = TestData.data_files
    for BB_ind in range(L):
    # BB_ind = 1
        bb = aa[BB_ind]
        d = bb.find('/img')
        cc = bb[d:len(bb)-4]
        dd = cc.split('_')
        # imageName = int(dd[0])
        xdim = int(dd[1])
        ydim = int(dd[2])
        # BB_Cord[ BB_ind , : ] = [xdim,ydim,imageName]
        BB_Cord[ BB_ind , : ] = [xdim,ydim]

    Data , Label = TestData(L)


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0
    for BB_ind in range(L):

        data[0,:,:,:]  = Data[BB_ind,:,:,:].copy()
        label[0,:,:,:] = Label[BB_ind,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction = net.predict( Trained_Model_Path, data)
        PredictedSeg = prediction[0,...,1] > 0.2

        # ix, iy, ImgNum = BB_Cord[ BB_ind , : ]
        ix, iy = BB_Cord[ BB_ind , : ]
        ix = int(148*ix)
        iy = int(148*iy)
        # AllImage[148*ix:148*(ix+1) , 148*iy:148*(iy+1) ,ImgNum] = prediction[0,...,1]
        # AllImage_logical[148*ix:148*(ix+1) , 148*iy:148*(iy+1) ,ImgNum] = PredictedSeg

        AllImage[ix:148+ix , iy:148+iy] = prediction[0,...,1]
        AllImage_logical[ix:148+ix , iy:148+iy] = PredictedSeg

        # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

        sz = label.shape

        A = (padSize/2)
        imgCombined = util.combine_img_prediction(data, label, prediction)
        DiceCoefficient[BB_ind] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
        util.save_image(
            imgCombined,
            f"{TestResults_Path}prediction_slice{str(BB_Cord[BB_ind])}.jpg",
        )


        Loss = unet.error_rate(prediction,label[:,A:sz[1]-A,A:sz[2]-A,:])
        LogLoss[BB_ind] = np.log10(Loss+eps)

    np.savetxt(f'{TestResults_Path}DiceCoefficient.txt', DiceCoefficient)
    np.savetxt(f'{TestResults_Path}LogLoss.txt', LogLoss)


    im = Image.fromarray(np.uint8(AllImage))
    msk = Image.fromarray(np.uint8(AllImage_logical))

    im.save(f'{TestResults_Path}PredictionSeg_{TestImageNum}.tif')
    msk.save(f'{TestResults_Path}PredictionSeg_{TestImageNum}_Logical.tif')


    return AllImage , AllImage_logical


def ThalamusExtraction(net , Test_Path , Train_Path , subFolders, CropDim , padSize):


    Trained_Model_Path = f'{Train_Path}model/model.cpkt'


    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(
        f'{Test_Path}*.tif', shuffle_data=False
    )

    L = len(TestData.data_files)
    DiceCoefficient  = np.zeros(L)
    LogLoss  = np.zeros(L)
    BB_Cord = np.zeros(L)

    for BB_ind in range(L):
        Stng = TestData.data_files[BB_ind]
        d = Stng.find('slice')
        BB_Cord[BB_ind] = int(Stng[d+5:].split('.')[0])

    BB_CordArg = np.argsort(BB_Cord)
    Data , Label = TestData(len(BB_Cord))


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0
    PredictionFull = np.zeros((szD[0],148,148,2))
    for BB_ind in BB_CordArg:

        data[0,:,:,:]  = Data[BB_ind,:,:,:].copy()
        label[0,:,:,:] = Label[BB_ind,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction = net.predict( Trained_Model_Path, data)
        PredictionFull[BB_ind,:,:,:] = prediction

    return PredictionFull
