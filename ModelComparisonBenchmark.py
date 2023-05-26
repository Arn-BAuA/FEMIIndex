#!/bin/python

import datetime

from BenchmarkFW.Models.FeedForward import Model as FeedForwardAE
from BenchmarkFW.Models.RecurrendAE import Model as LSTMAE
from BenchmarkFW.Models.CNN_AE import Model as CNNAE
from BenchmarkFW.Models.AttentionBasedAE import Model as AttentionModel

from BenchmarkFW.SetWrappers.UCRArchive import loadData as UCRDataSet
from BenchmarkFW.SetWrappers.SMD import loadData as SMDDataSet
from BenchmarkFW.SetWrappers.ECGDataSet import loadData as ECGDataSet
from BenchmarkFW.DataGenerators.Sines import generateData as Sines
from BenchmarkFW.Trainers.SingleInstanceTrainer import Trainer as OnlineTrainer
from BenchmarkFW.Trainers.BatchedTrainer import Trainer as BatchedTrainer

from BenchmarkFW.Benchmark import benchmark,initializeDevice
from BenchmarkFW.Evaluation.QuickOverview import plotOverview


device = initializeDevice()

numDataSources = 3

selectedUCRSets = [
            "ACSF1",
            "AllGestureWiimoteX",
            "BME",
            "Chinatown",
            "Crop",
            "DodgerLoopDay",
            "EOGHorizontalSignal",
            "EthanolLevel",
            "FreezerRegularTrain",
            "Fungi",
            "GestureMidAirD3",
            "GesturePebbleZ2",
            "GunPoint",
            "HouseTwenty",
            "MixedShapesRegularTrain",
            "PigAirwayPressure",
            "PLAID",
            "PowerCons",
            "Rock",
            "SemgHandGenderCh2",
            "SemgHandMovementCh2",
            "SemgHandSubjectCh2",
            "SmoothSubspace",
            "UMD",
            "Wafer",
    ]


numDataSrcs = 32

def getData(Dimensions,DataSrcNumber):
    if DataSrcNumber == 0:
        return Sines(Dimensions)
    if DataSrcNumber == 1:
        return Sines(Dimensions,AnomalousAmplitudes=[1.2,1.2])
    if DataSrcNumber == 2:
        return Sines(Dimensions,AnomalousFrequency=[1,1.2])
    if DataSrcNumber == 3:
        return Sines(Dimensions,NoiseLevel = 0.1)
    

    if DataSrcNumber == 4:
        return ECGDataSet(Dimensions)

    if DataSrcNumber == 5:
        return SMDDataSet(Dimensions,nNormalDimensions=0)
    if DataSrcNumber == 6:
        return SMDDataSet(Dimensions,nNormalDimensions=int(Dimensions*0.5))
    
    return UCRDataSet(Dimensions,DataSet = selectedUCRSets[DataSrcNumber-1])

numModels = 15

def getModel(Dimensions,ModelNumber,sampleDataSet):
    if ModelNumber == 0: 
        return FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length(),sliceLength=0.2,LatentSpaceSize=0.1,NumLayersPerPart=4)
    if ModelNumber == 1: 
        return FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length(),sliceLength=0.2,LatentSpaceSize=0.25,NumLayersPerPart=4)
    if ModelNumber == 2: 
        return FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length(),sliceLength=0.2,LatentSpaceSize=0.5,NumLayersPerPart=2)
    if ModelNumber == 3: 
        return FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length(),sliceLength=0.2,LatentSpaceSize=1,NumLayersPerPart=1)
        
    if ModelNumber == 4: 
        return LSTMAE(Dimensions,device,CellKind = "LSTM")
    if ModelNumber == 5: 
        return LSTMAE(Dimensions,device,CellKind = "GRU")
    if ModelNumber == 6: 
        return LSTMAE(Dimensions,device,CellKind = "LSTM",PerformLatentFlip=False)
    if ModelNumber == 7: 
        return LSTMAE(Dimensions,device,CellKind = "GRU",PerformLatentFlip=False)
        
    if ModelNumber == 8: 
        return CNNAE(Dimensions,device,InputSize = trainingSet.Length())
    if ModelNumber == 9: 
        return CNNAE(Dimensions,device,hasFFTEncoder = True,InputSize = trainingSet.Length())
    if ModelNumber == 10: 
        return CNNAE(Dimensions,device,hasFFTEncoder = True,hasOnlyFFTEncoder=True,InputSize = trainingSet.Length())
    if ModelNumber == 11: 
        return CNNAE(Dimensions,device,hasFFTEncoder = True,HanningWindowBeforeFFT=False,InputSize = trainingSet.Length())
    if ModelNumber == 12: 
        return CNNAE(Dimensions,device,hasFFTEncoder = True,HanningWindowBeforeFFT=False,hasOnlyFFTEncoder=True,InputSize = trainingSet.Length())
        
    if ModelNumber == 13: 
        return AttentionModel(Dimensions,device,InputSize = trainingSet.Length())
    if ModelNumber == 14: 
        return AttentionModel(Dimensions,device,InputSize = trainingSet.Length(),FeedDirect=False)
        

masterPath = "HugeBenchmarkResults/"

#NBatches will not be transmitted at the moment...
def BenchmarkRun(Dimensions,DataSrcNumber,ModelNumber,nBatches=10):
    
    identifier = "Dimensions "+str(Dimensions)+" DataSrcNumber "+str(DataSrcNumber)+" ModelNumber "+str(ModelNumber)+"nBatches "+str(nBatches) 
    
    statusFile = open("Status.log","a")
    statusFile.write("["+str(datetime.datetime.now())+"] Started " + identifier +'\n')
    statusFile.close()

    try:
        trainingSet,validationSet,testSet = getData(Dimensions,DataSrcNumber)
        model = getModel(Dimensions,ModelNumber,trainingSet)
        
        pathToSave = masterPath + identifier 
        trainer = BatchedTrainer(model,device)

        resultFolder = benchmark(trainingSet,
                  validationSet,
                  testSet,
                  model,
                  trainer,
                  n_epochs=100,
                  pathToSave=pathToSave,
                  device = device)

    except e:
        errorFile = open(masterPath+Errors.log,'a')
        errorFile.write("["+str(datetime.datetime.now())+"] " + identifier +'\n')
        errorFile.write(e)
        errorFile.close()

    statusFile = open("Status.log","a")
    statusFile.write("["+str(datetime.datetime.now())+"] Ended " + identifier +'\n')
    statusFile.close()

import numpy as np

dimensions = [1,2,4,8,16,32]
models = np.arange(0,numModels)
dataSrcs = np.arange(0,numDataSrcs)

absolvedRuns = 0.0
startTime = datatime.datetime.now()
totalRuns = float(len(dimensions)*len(models)*len(dataSrcs))

for d in dimensions:
    for m in models:
        for s in dataSrcs:
            BenchmarkRun(d,s,m,nBatches=10)
            absolvedRuns +=1
            estimatedTime = startTime + ((datetime.datetime.now()-startTime)*(totalRuns/absolvedRuns))
            print("Estimated Date of termination : ",estimatedTime)


