from ExperimentClass import Experiment

import numpy as np
import pandas as pd
import json

from BenchmarkFW.Factorys.DataSource import getStandardDataSource as DataSource
from BenchmarkFW.DataModifyers.smooth import rollingAverage
from FEMIDataIndex import computeFEMIIndex


#The goal of this experiment is to reason, how stable the FEMI-INdex is. To asses this, we simulate a case, where we randomly
# sample form the same data source. How much does the FEMI-Index change?
# in Addition, we average the values in the dataset to remove higher freqeuncies from the set. We look at how this changes styability.

class StabilityExperiment(Experiment):

    def __init__(self,args):
        Experiment.__init__(self,"NoiseExperiment/",args)

    def _conduct(self,args):
        
        percentualWindowLengths = np.array([0.01,0.02,0.04,0.08,0.1,0.25,0.5,0.75])
        nSamples = 10
        nDatasets = 32 #From the dataset Factory
        

        for dsIndex in range(0,nDatasets):
            
            print("Calculating for ds ",dsIndex)

            EComponent = np.zeros([len(percentualWindowLengths)+1,nSamples])
            MIComponent = np.zeros([len(percentualWindowLengths)+1,nSamples])
            EPolar = np.zeros([len(percentualWindowLengths)+1,nSamples])
            MIPolar = np.zeros([len(percentualWindowLengths)+1,nSamples])
            
            #################################
            #   Calculation of individual E and MI
            #
            

            for ds in range(0,nSamples):
                
                print("Sample ",ds)
                
                trainingSet,validationSet,testSet = DataSource(1,dsIndex,TrainingSetSize = 100,ValidationSetSize=100,TestSetSize=0)

                avgWindowLengths = (percentualWindowLengths*trainingSet.Length()).astype(int)
                for avgI in range(0,len(avgWindowLengths)):
                    
                    windowed_trainingSet = rollingAverage(avgWindowLengths[avgI],trainingSet)
                    windowed_validationSet = rollingAverage(avgWindowLengths[avgI],validationSet) 
                   

                    EComponent[avgI+1][ds],MIComponent[avgI+1][ds] = computeFEMIIndex(windowed_trainingSet,windowed_validationSet,polarFEMIIndex = False)
                    EPolar[avgI+1][ds],MIPolar[avgI+1][ds] = computeFEMIIndex(windowed_trainingSet,windowed_validationSet,polarFEMIIndex = True)

                
                EComponent[0][ds],MIComponent[0][ds] = computeFEMIIndex(trainingSet,validationSet,polarFEMIIndex = False)
                EPolar[0][ds],MIPolar[0][ds] = computeFEMIIndex(trainingSet,validationSet,polarFEMIIndex = True)
            
            ######################################
            #   Mean and STD as Value and Error
            #

            EComponentMean = np.mean(EComponent,axis = 1)
            EComponentStd = np.std(EComponent,axis = 1)
            MIComponentMean = np.mean(MIComponent,axis = 1)
            MIComponentStd = np.std(MIComponent,axis = 1)
            
            EPolarMean = np.mean(EPolar,axis = 1)
            EPolarStd = np.std(EPolar,axis = 1)
            MIPolarMean = np.mean(MIPolar,axis = 1)
            MIPolarStd = np.std(MIPolar,axis = 1)
            
            windowLenghths = np.concatenate([[0],percentualWindowLengths])

            df = pd.DataFrame({     
                    "Window Length":windowLenghths,
                    "E_Component":EComponentMean,
                    "Delta_E_Component":EComponentStd,
                    "MI_Component":MIComponentMean,
                    "Delta_MI_Component":MIComponentStd, 
                    "E_Polar":EPolarMean,
                    "Delta_E_Polar":EPolarStd,
                    "MI_Polar":MIPolarMean,
                    "Delta_MI_Polar":MIPolarStd,
                })
            df.to_csv(self.resultPath+"DatasetNr."+str(dsIndex)+".csv",sep="\t") 
            
            DataSetHP = trainingSet.hyperParameters()
            DataSetHP = {"DataSetName":trainingSet.Name(),"HyperParameters":DataSetHP}

            with open(self.resultPath+"DatasetNr."+str(dsIndex)+".json",'w') as f:
                json.dump(DataSetHP,f,default = str,indent=4)

    def _plot(self,args):
        #Code to plot experimental Results goes here.
        pass

import sys

StabilityExperiment(sys.argv)
