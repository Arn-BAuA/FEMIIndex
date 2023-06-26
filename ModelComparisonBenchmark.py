#!/bin/python

import datetime
import traceback

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

from BenchmarkFW.Factorys.DataSource import getStandardDataSource as getData 


device = initializeDevice()
numModels = 15

def getModel(Dimensions,ModelNumber,trainingSet):
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
def BenchmarkRun(Dimensions,DataSrcNumber,ModelNumber,nBatches=10,interactive = False):
    
    identifier = "Dimensions "+str(Dimensions)+" DataSrcNumber "+str(DataSrcNumber)+" ModelNumber "+str(ModelNumber)+" nBatches "+str(nBatches) 
    
    statusFile = open(masterPath+"Status.log","a")
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
                  n_epochs=40,
                  pathToSave=pathToSave,
                  device = device)

    except Exception as e:
        print("Oh. An Error Occured             WHAT IS HAPPPENIIIIING")
        errorFile = open(masterPath+"Errors.log",'a')
        errorFile.write("\n \n \n")
        errorFile.write("["+str(datetime.datetime.now())+"] " + identifier +'\n')
        tb=traceback.format_exc()
        errorFile.write(tb)
        errorFile.write(str(e))
        errorFile.close()

        if interactive:
            print(tb)
            print(e)

    statusFile = open(masterPath+"Status.log","a")
    statusFile.write("["+str(datetime.datetime.now())+"] Ended " + identifier +'\n')
    statusFile.close()

import numpy as np

def fullBenchmark(dimensions =  [1,2,4,8,16,32]):

    print("Testing for dimensions: ",dimensions)
    models = np.arange(0,numModels)
    dataSrcs = np.arange(0,170)

    absolvedRuns = 0.0
    startTime = datetime.datetime.now()
    totalRuns = float(len(dimensions)*len(models)*len(dataSrcs))

    for d in dimensions:
        for s in dataSrcs:
            for m in models:
                BenchmarkRun(d,s,m,nBatches=10)
                absolvedRuns +=1
                estimatedTime = startTime + ((datetime.datetime.now()-startTime)*(totalRuns/absolvedRuns))
                print("Estimated Date of termination : ",estimatedTime)

import sys
if len(sys.argv) == 1:
    fullBenchmark()
else:
    dimensions = [0]*(len(sys.argv)-1)
    for i in range(1,len(sys.argv)):
        dimensions[i-1] = int(sys.argv[i])

    fullBenchmark(dimensions)

#import sys

#numDimensions = int(sys.argv[1])
#dataSourceNumber = int(sys.argv[2])
#modelNumber = int(sys.argv[3])

#BenchmarkRun(numDimensions,dataSourceNumber,modelNumber,nBatches=10,interactive = True)
