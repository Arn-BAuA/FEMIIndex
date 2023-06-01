

import os
import glob
import sys

import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib

colorMap = plt.cm.plasma

modelIndex = 1
epoch = 40

plotDir = "TestPlots/FEMIIndices/"
benchmarkPath = "HugeBenchmarkResults/"
baselinePaths = "TestResults/BaseLine/"
dimensionality = 1
nBatches = 10

#Check which results exist in the Benchmarkresults...

resultPaths = glob.glob(benchmarkPath+"Dimensions "+str(dimensionality)+" DataSrcNumber * ModelNumber "+str(modelIndex)+" nBatches "+str(nBatches))

dataSource = []
bPaths = [] #b .. Benchmark


#potentially, there are empty folders, which can happen, when there is an error during benchmarking. CHecking for this.
for path in resultPaths:
    if os.path.exists(path+"/Errors.csv"):
        dataSource.append(int(path.split(' ')[3])) 
        bPaths.append(path)

#Chech which results exist for the Baselines
existingBaselines = glob.glob(baselinePaths+"*.line")

dBaselineIndices = {}

for path in existingBaselines:
    index = path.split('/')[-1]
    index = index[:-5]
    index = int(index[8:])

    if index in dataSource:
        dBaselineIndices[index] = path

commons = []
commonBPaths = []
commonBLPaths = [] 


for i in range(0,len(dataSource)):
    if dataSource[i] in dBaselineIndices:
        commons.append(dataSource[i])
        commonBPaths.append(bPaths[i])
        commonBLPaths.append(dBaselineIndices[dataSource[i]])

#Load the AUC scores and FEMI-Indices for the existsing stuff

dataSets = {}

modelNameLoaded = False
ModelName = ""

for i,index in enumerate(commons):

    metaDataFile = open(commonBPaths[i]+"/HyperParametersAndMetadata.json",'r')
    metaData = json.load(metaDataFile)
    metaDataFile.close()

    benchmarkResults = pd.read_csv(commonBPaths[i]+"/Errors.csv",sep='\t')
    
    baseLine = open(commonBLPaths[i],'r')
    line = baseLine.readlines()[0]
    baseLine.close()
    line = line.split(',')
    
    sourceName = metaData["General Information"]["Used Dataset"]+" ("+str(index)+")"

    if not modelNameLoaded:
        ModelName =metaData["General Information"]["Used Model"]+" ("+str(modelIndex)+")"
        modelNameLoaded = True

    dataSets[sourceName] ={
                    "E_Component":float(line[0]),
                    "Delta_E_Component":float(line[1]),
                    "MI_Component":float(line[2]),
                    "Delta_MI_Component":float(line[3]),
                    "E_Polar":float(line[4]),
                    "Delta_E_Polar":float(line[5]),
                    "MI_Polar":float(line[6]),
                    "Delta_MI_Polar":float(line[7]),
                    "AUC":benchmarkResults.loc[benchmarkResults["#Epoch"] == epoch]["AUC Score on Validation Set"],
                    "AUC_Error":benchmarkResults.loc[benchmarkResults["#Epoch"] == epoch]["AUC Score on Validation Set Delta"],
                    "Benchmark_Data":benchmarkResults,
                    "Metadata":metaData,
                    }



types = ["Polar","Component"]
Quantitys = ["E","MI"]
legendMapping = {"E":"Entropy","MI":"Mutual Information"}

markers = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d",4,5,6,7,8,9,10,11]

for t in types:

    #Plot Values 2D
    fig,ax = plt.subplots()

    for i,src in enumerate(dataSets):

        ax.errorbar(
                [dataSets[src]["E_"+t]],
                [dataSets[src]["MI_"+t]],
                xerr=[dataSets[src]["Delta_E_"+t]],
                yerr=[dataSets[src]["Delta_MI_"+t]],
                linestyle="None",
                marker = markers[i%len(markers)],
                markersize = 12,
                color = 'k',
                label = src,
                )

        ax.errorbar(
                [dataSets[src]["E_"+t]],
                [dataSets[src]["MI_"+t]],
                marker = markers[i%len(markers)],
                markersize = 8,
                color = colorMap(dataSets[src]["AUC"]),
                )
    ax.set_title(t+" FEMI Index for "+ModelName)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Mutual Information")
    ax.legend()

    fig.colorbar(matplotlib.cm.ScalarMappable(cmap = colorMap),ax = ax,label="AUC Score")

    fig.set_figwidth(15)
    fig.set_figheight(10)
    
    plt.savefig(plotDir+t+" FEMI Index for "+ModelName+".pdf")
    plt.close()

    #Plot The 1D Stuff

    for q in Quantities:
        
        fig,ax = plt.subplots()

        for i,src in enumerate(dataSets):

            ax.errorbar(
                    [dataSets[src][q+"_"+t]],
                    [dataSets[src]["AUC"]],
                    xerr=[dataSets[src]["Delta_"+q+"_"+t]],
                    yerr=[dataSets[src]["AUC_Error"]],
                    linestyle="None",
                    marker = markers[i%len(markers)],
                    markersize = 12,
                    label = src,
                    )
            ax.set_title(t" "+legendMapping[q]+" VS AUC for "+ModelName)
            ax.set_xlabel(legendMapping[q])
            ax.set_ylabel("AUC")
            ax.legend()

            fig.set_figwidth(15)
            fig.set_figheight(10)
            
            plt.savefig(plotDir+t+" "+legendMapping[q]+" VS AUC for "+ModelName+".pdf")
            plt.close()




