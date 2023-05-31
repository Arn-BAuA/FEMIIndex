from .ExperimentClass import Experiment

import numpy as np
import pandas as pd
from BenchmarkFW.Factorys.DataSource import getStandardDataSource as DataSource
from BenchmarkFW.DataModifyers.smooth import rollingAverage
from .FEMIDataIndex import computeFEMIIndex


#The goal of this experiment is to reason, how stable the FEMI-INdex is. To asses this, we simulate a case, where we randomly
# sample form the same data source. How much does the FEMI-Index change?
# in Addition, we average the values in the dataset to remove higher freqeuncies from the set. We look at how this changes styability.

class StabilityExperiment(Experiment):

    def __init__(self,args):
        Experiment.__init__("NoiseExperiment/",args)

    def _conduct(self,args):
        
        avgWindowLengths = np.linspace(2,15,14).astype(int)
        nSamples = 10
        nDatasets = 32 #From the dataset Factory
        

        for dsIndex in range(0,len(nDatasets)):
            
            print("Calculating for ds ",dsIndex)

            EComponent = [[0]*nSamples]*(len(avgWindowLengths)+1)
            MIComponent = [[0]*nSamples]*(len(avgWindowLengths)+1)
            EPolar = [[0]*nSamples]*(len(avgWindowLengths)+1)
            MIPolar = [[0]*nSamples]*(len(avgWindowLengths)+1)
            
            #################################
            #   Calculation of individual E and MI
            #

            for ds in range(0,nSamples):

                trainingSet,validationSet,testSet = DataSource(1,dsIndex,TrainingSetSize = 100,ValidationSetSize=100,TestSetSize=0)
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


            EComponenet = np.array(EComponenet)
            MIComponenet = np.array(MIComponenet)
            EPolar = np.array(EPolar)
            MIPolar = np.array(MIPolar)

            EComponentMean = np.mean(EComponent,axis = 1)
            EComponentStd = np.std(EComponent,axis = 1)
            MIComponentMean = np.mean(MIComponent,axis = 1)
            MIComponentStd = np.std(MIComponent,axis = 1)
            
            EPolarMean = np.mean(EPolar,axis = 1)
            EPolarStd = np.std(EPolar,axis = 1)
            MIPolarMean = np.mean(MIPolar,axis = 1)
            MIPolarStd = np.std(MIPolar,axis = 1)
            
            windowLenghths = np.concatenate([[1],avgWindowLenghts])

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
            df.to_csv(self.resultPath+"DatasetNr."+str(dsIndex),sep="\t") 
            

    def _plot(self,args):
        #Code to plot experimental Results goes here.
        pass
