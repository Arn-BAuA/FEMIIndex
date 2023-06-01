from ExperimentClass import Experiment

import numpy as np
import pandas as pd
import json

from BenchmarkFW.Factorys.DataSource import getStandardDataSource as DataSource
from BenchmarkFW.DataModifyers.addNoise import addUniformNoise
from FEMIDataIndex import computeFEMIIndex

import matplotlib.pyplot as plt

#The goal of this experiment is to reason, how stable the FEMI-INdex is. To asses this, we simulate a case, where we randomly
# sample form the same data source. How much does the FEMI-Index change?
# in Addition, we average the values in the dataset to remove higher freqeuncies from the set. We look at how this changes styability.

class NoiseExperiment(Experiment):

    def __init__(self,args):
        Experiment.__init__(self,"NoiseExperiment/",args)

    def _conduct(self,args):
        
        percentualNoiseLevels = np.array([0,0.01,0.02,0.04,0.08,0.1,0.25,0.5,0.75,2])
        nSamples = 10
        nDatasets = 22 #From the dataset Factory
        
        

        for dsIndex in range(0,nDatasets):

            print("Calculating for ds ",dsIndex)
            
            #There has to be a better way...
            trainingSet,validationSet,testSet = DataSource(1,dsIndex,TrainingSetSize = 1,ValidationSetSize=0,TestSetSize=0)#Just for the HPs
            DataSetHP = {"DataSetName":trainingSet.Name(),"HyperParameters":trainingSet.hyperParameters()}

            def alteredSetDispenser(percentualNoiseLevel):
                
                def setDispenser():
                    print("Dispensed Set for Noise. Level. ",percentualNoiseLevel)
                    trainingSet,validationSet,testSet = DataSource(1,dsIndex,TrainingSetSize = 100,ValidationSetSize=100,TestSetSize=0)
                    
                    noisy_trainingSet = addUniformNoise(percentualNoiseLevel,trainingSet)
                    noisy_validationSet = addUniformNoise(percentualNoiseLevel,validationSet) 

                    return noisy_trainingSet,noisy_validationSet
                
                return setDispenser

            self.recordSeries(
                                nSamples,
                                alteredSetDispenser,
                                parameters = percentualNoiseLevels,
                                parameterName = "Noise Level",
                                pathToSave = "DatasetNr."+str(dsIndex),
                                hyperparameters = DataSetHP,
                                includeBaseline = True,
                                baseLineDSIndex = dsIndex
                                )



    #As Arguments, the name of the files to be plotted are expected.
    #There are always two files per measurement. one csv and one json. Both are loaded, if one of them is provided.
    def _plot(self,args,autosave = True,showFig = False):
        self._fullPlotAgainstVariedQuantity("Noise Level",
                                            "Noise Lvl. PtP. in D.S. Std.Dev.",
                                            "Noise Analysis",
                                            "Noise Analysis",
                                            args=args,
                                            colorMap=plt.cm.cividis,
                                            autosave=True,
                                            showFig=False) 

import sys

NoiseExperiment(sys.argv)
