import os
import pickle


Directory = '/media/data1/artin/data/Thalamus/ForUnet_Test6/'


subFolders = os.listdir(Directory)
subFolders2 = []
i = 0
for o in range(len(subFolders)-1):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;

with open(Directory + subFolders2[i] + '/test/results/DiceCoefficient.txt' ,"rb") as fp:
    subFolders = pickle.load(fp)
