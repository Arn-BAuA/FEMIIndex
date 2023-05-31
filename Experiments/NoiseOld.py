import numpy as np

from numpy.random import random
from copy import copy

import matplotlib.pyplot as plt

import torch
from FEMIDataIndex import computeFEMIIndex

from BenchmarkFW.BlockAndDatablock import DataBlock

from BenchmarkFW.SetWrappers.UCRArchive import loadData as DataSet
from BenchmarkFW.SetWrappers.UCRArchive import getDatasetsInArchive

UCRSets = getDatasetsInArchive()
selectedSet = 7

##############################################
#       Generating Noisy Data:               #
##############################################

noiseLevels = np.concatenate([np.linspace(0,0.01,11),np.linspace(0.1,0.5,7),np.linspace(0.5,1,6),np.linspace(1,10,19)])

trainingSet,validationSet,testSet = DataSet(
                                                DataSet = UCRSets[selectedSet],
                                                dimensions=1,
                                                TrainingSetSize = 100,
                                                ValidationSetSize = 100,
                                                TestSetSize = 10,
                                                anomalyPercentageTest = 10)


trainingSetSTD = torch.std(torch.stack(trainingSet.Data()))

EValues = [0]*len(noiseLevels)
MIValues = [0]*len(noiseLevels)

for i in range(0,len(noiseLevels)):
    
    noisedTrainingData = copy(trainingSet.Data())
    noisedValidationData = copy(validationSet.Data())
    
    dpShape = noisedTrainingData[0].shape #they all have the same shape

    for j in range(0,len(noisedTrainingData)):
        noisedTrainingData[j] += (0.5-torch.rand(dpShape))*2*(noiseLevels[i]/trainingSetSTD)
        
    for j in range(0,len(noisedValidationData)):
        noisedValidationData[j] += (0.5-torch.rand(dpShape))*2*(noiseLevels[i]/trainingSetSTD)
         
    noisedTrainingSet = DataBlock("",noisedTrainingData,trainingSet.Dimension())
    noisedValidationSet = DataBlock("",noisedValidationData,validationSet.Dimension())

    EValues[i],MIValues[i] = computeFEMIIndex(noisedTrainingSet,noisedValidationSet)


plt.plot(noiseLevels,EValues)
plt.show()
plt.close()
plt.plot(noiseLevels,MIValues)
plt.show()
plt.close()
