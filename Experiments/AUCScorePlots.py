

import os
import glob
import sys

import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib

colorMap = plt.cm.plasma

plotDir = "TestPlots/FEMIIndices/"
benchmarkPath = "HugeBenchmarkResults/"
baselinePaths = "TestResults/BaseLine/"
dfExportPaths = "TestResults/AUCandFEMI/"
dimensionality = 1
nBatches = 10

def plotFEMIandAUC(modelIndex,epoch):
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
    
    print(len(commons),"Datasets found for model ",modelIndex,"at epoch",epoch)

    dataSets = {}

    modelNameLoaded = False
    ModelName = ""

    #To keep the naming scheme consistent:
    dSourceCol = "Data_Source"
    ECCol = "E_Component"
    dECCol = "Delta_E_Component"
    MICCol = "MI_Component"
    dMICCol = "Delta_MI_Component"
    EPCol = "E_Polar"
    dEPCol = "Delta_E_Polar"
    MIPCol = "MI_Polar"
    dMIPCol = "Delta_MI_Polar"
    AUCCol = "AUC"
    dAUCCol = "AUC_Error"
    
    DFtoSave = pd.DataFrame(columns = [dSourceCol,ECCol,dECCol,MICCol,dMICCol,EPCol,dEPCol,MIPCol,dMIPCol,AUCCol])
    

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
        
        dataDict = {
                        dSourceCol:sourceName,
                        ECCol:float(line[0]),
                        dECCol:float(line[1]),
                        MICCol:float(line[2]),
                        dMICCol:float(line[3]),
                        EPCol:float(line[4]),
                        dEPCol:float(line[5]),
                        MIPCol:float(line[6]),
                        dMIPCol:float(line[7]),
                        AUCCol:float(benchmarkResults.loc[benchmarkResults["#Epoch"] == epoch]["AUC Score on Validation Set"]),
                        dAUCCol:float(benchmarkResults.loc[benchmarkResults["#Epoch"] == epoch]["AUC Score on Validation Set Delta"])
                        }
        
        print(dataDict)

        DFtoSave = DFtoSave.append(dataDict,ignore_index = True)

        dataDict["Benchmark_Data"] = benchmarkResults
        dataDict["Metadata"] = metaData

        dataSets[sourceName]= dataDict
    
    DFtoSave.to_csv(dfExportPaths+"Model No "+str(modelIndex)+" epoch "+str(epoch)+".csv",sep='\t')

    types = ["Polar","Component"]
    Quantities = ["E","MI"]
    legendMapping = {"E":"Entropy","MI":"Mutual Information"}

    markers = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d",4,5,6,7,8,9,10,11]

    if len(dataSets) < len(markers):
        markers = ["."]
        
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
        ax.set_title(t+" FEMI Index for "+ModelName+"("+str(epoch)+" Epochs training)")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Mutual Information")
        ax.grid()
        
        if len(dataSets) < len(markers):
            ax.legend()

        fig.colorbar(matplotlib.cm.ScalarMappable(cmap = colorMap),ax = ax,label="AUC Score")
        
        fig.set_figwidth(10)
        fig.set_figheight(3)
        
        plt.savefig(plotDir+t+" FEMI Index for "+ModelName+" "+str(epoch)+" Epochs.pdf")
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
            ax.set_title(t+" "+legendMapping[q]+" VS AUC for "+ModelName+"("+str(epoch)+" Epochs training)")
            ax.set_xlabel(legendMapping[q])
            ax.set_ylabel("AUC")
            ax.legend()
            ax.grid()

            fig.set_figwidth(8)
            fig.set_figheight(3)
            
            plt.savefig(plotDir+t+" "+legendMapping[q]+" VS AUC for "+ModelName+" "+str(epoch)+" Epochs.pdf")
            plt.close()


for m in range(0,16):
    for e in [0,20,40]:
        plotFEMIandAUC(m,e)


