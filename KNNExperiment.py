#The Experiment file for the KNN algorithm...

import pandas as pd
import numpy as np

csvFolder = "TestResults/AUCandFEMI/"
#This are the column names of the performance measure in the df
P = "AUC"
dP = "AUC_Error"
minPerformance = 0
maxPerformance = 1
performanceSpan = maxPerformance - minPerformance

#########################
## Utility

def loadCSV(modelID, epoch):
    return pd.read_csv(csvFolder+"Model No "+str(modelID)+" epoch "+str(epoch)+".csv",sep = '\t')

#########################
## Random Guessing Performance

#d Treshhold ... decency Treshhold
def getRandomEstimatorDCER(femiPerformanceData,dTreshhold):
    deltas = 2*femiPerformanceData[dP]
    deltas += 2*dTreshhold
    #deltas[deltas > maxPerformance] = maxPerformance
    return deltas.sum()/len(deltas.index)

##########################
## KNN

#A point in FEMI-Plain
class Point:
    
    def __init__(self,E,dE,MI,dMI,initialValues):
        self.E = E
        self.dE = dE
        self.MI = MI
        self.dMI = dMI
        self.values = initialValues

    def add(self,valueName,value):
        self.values[valueName] = value

    def getE(self):
        return self.E,self.dE
    
    def getMI(self):
        return self.MI,self.dMI

    def getValue(self,valueName):
        return self.values[valueName]

#You could say "thats a utility method." its here because its utility for knn
def pointCloudFromDF(df,FEMIType = "Polar"):
    pointCloud = [0] * len(df.index)
    
    for i,row in df.iterrows():
        initialPValues = {
                    P:row[P],
                    dP:row[dP]
                }

        p = Point(  row["E_"+FEMIType],
                    row["Delta_E_"+FEMIType],
                    row["MI_"+FEMIType],
                    row["Delta_MI_"+FEMIType],
                    initialPValues)

        pointCloud[i] = p

    return pointCloud

#################
# Diffrent distance Metrics

#2D Euclidic distance between E,MI and pE,pMI ("point E" and "point MI")
#All distance measures should implement this interface.
def euclidicD(E,MI,pE,dpE,pMI,dpMI):

    distance = np.sqrt(np.power(E-pE,2) + np.power(MI-pMI,2))
    
    #computing the Error
    eErrContrib = 2*(pE-E)*dpE
    miErrContrib = 2*(pMI-MI)*dpMI

    err = (1/(2*distance)) * np.sqrt(np.power(eErrContrib,2)+np.power(miErrContrib,2))

    return  distance,err

#################
# Diffrent Weights

#All Weight functions should implement this interface...
def sphericalWeight(Distance,dDistance,epsilon):
    
    weight = np.sqrt(np.power(epsilon,2)-np.power(Distance,2))/epsilon
    err = (Distance*dDistance)/(np.power(epsilon,2)*weight)

    return weight,err

##################
# THE MAIN KNN ALGORITHM:

def KNNPerformanceEstimate(E,MI,epsilon,pointCloud,
                                    d = euclidicD,
                                    w = sphericalWeight):
    pEstimate = 0
    
    #Here, the values of the points in the epsilon ball are stored
    #performance indicator
    ps = [] #value
    dps = [] #error
    #weight of this point
    ws = [] #value
    dws = [] #error

    #Selecting points that are in the epsilon ball
    for point in pointCloud:
        
        distance,dDistance = d(E,MI,*point.getE(),*point.getMI())
        
        if distance < epsilon:
            #Is in the epsilon ball
            
            
            ps.append(point.getValue(P))
            dps.append(point.getValue(dP))

            weight,dWeight = w(distance,dDistance,epsilon)
            ws.append(weight)
            dws.append(dWeight)
    
    if len(ps) == 0:
        #no points in the neighbourhood -> unknown terrain
        return float(minPerformance+maxPerformance)/2.0,performanceSpan*0.5

    ps = np.array(ps)
    dps = np.array(dps)
    ws = np.array(ws)
    dws = np.array(dws)

    #Begining the calculation
    
    #The KNN Performance
    weightedPs = ps*ws
    W = ws.sum()
    KNNPerformance = weightedPs.sum()/W
    
    #The Error of the KNN Performance
    
    pContrib = (ws*dps)/W
    pContrib = np.power(pContrib,2)
    pContrib = pContrib.sum()

    wContrib = (ps*dws)/W
    wContrib = np.power(wContrib,2)
    wContrib = wContrib.sum()

    dW = dws * dws
    dW = np.sqrt(dW.sum())
    WContrib = np.power((KNNPerformance*dW)/W,2)

    deltaKNNPerformance = np.sqrt(pContrib+wContrib+WContrib)

    deltaKNNPerformance = max(deltaKNNPerformance,ps.std())
    deltaKNNPerformance = min(deltaKNNPerformance,performanceSpan*0.5)

    return KNNPerformance,deltaKNNPerformance

#Alpha is the dececy threshhold...
def KNNEvaluation(epsilon,alpha,pointCloud,d = euclidicD,w = sphericalWeight):

    EstimateWasDecent = np.zeros(len(pointCloud))
    EstimateWasCorrect = np.zeros(len(pointCloud))
    
    predErrors = np.zeros(len(pointCloud))
    actualDistances = np.zeros(len(pointCloud))

    for i in range(0,len(pointCloud)):

        point = pointCloud[i]
        remainingCloud = pointCloud[:i]+pointCloud[i+1:]
        
        E,dE = point.getE()
        MI,dMI = point.getMI()
        
        Preal = point.getValue(P)
        dPreal = point.getValue(dP)
        
        Ppred,dPpred = KNNPerformanceEstimate(E,MI,epsilon,remainingCloud,d,w)
        
        predErrors[i] = dPpred
        actualDistances[i] = abs(Ppred-Preal)

        #print("Predicted: ",np.round(Ppred,2)," +/- ",np.round(dPpred,2),", Real:",np.round(Preal,2)," +/- ",np.round(dPreal,2))

        EstimateWasDecent[i] = dPpred < alpha

        if Ppred < (Preal+dPreal+dPpred) and Ppred > (Preal-dPreal-dPpred):
            EstimateWasCorrect[i] = 1

    DescentCorrectEstimates = EstimateWasDecent*EstimateWasCorrect #only one if descent and correct

    DER = float(EstimateWasDecent.sum())/float(len(pointCloud))
    CER = float(EstimateWasCorrect.sum())/float(len(pointCloud))
    DCER = float(DescentCorrectEstimates.sum())/float(len(pointCloud))
    
    return DCER,CER,DER,predErrors.mean(),predErrors.std(),actualDistances.mean(),actualDistances.std()

#Small Test:

def smallParameterEvaluation(modelID = 11,epoch = 40,
                                alphas = [0.05,0.1,0.2,0.25],
                                epsilons=[2,3,4,5,10],
                                FEMIType="Polar"):

    data = loadCSV(modelID,epoch)

    cloud = pointCloudFromDF(data,FEMIType)

    for alpha in alphas:
        randDCER =getRandomEstimatorDCER(data,alpha)
        for epsilon in epsilons:

            KNNDCER,KNNCER,KNNDER,merr,stderr,dist,disterr = KNNEvaluation(epsilon,alpha,cloud)

            print()
            print("Alpha = ",alpha," Epsilon = ",epsilon)
            print("=========================================")
            print("Random Performance\t",randDCER)
            print("KNN Performance\t",KNNDCER)
            print("(CER: ",KNNCER," DER: ",KNNDER,")")
            print("Mean KNN Err: ",merr,"+/-",stderr)
            print("Mean Dist, p vs. r",dist,"+/-",disterr)


import matplotlib.pyplot as plt
import matplotlib

def KNNFemiHeatmapPlot(modelID = 11,epoch = 40,epsilon=10,FEMIType="Polar",resolution=1):
    
    data = loadCSV(modelID,epoch)
    cloud = pointCloudFromDF(data,FEMIType)

    colorMap=plt.cm.plasma
    errColorMap=plt.cm.viridis
    figFolder = "TestPlots/KNN/" 

    def enterBenchmarkResults(fig,ax,makeColorBar = True,marker='.'):

        for point in cloud:
        
            E,dE = point.getE()
            MI,dMI = point.getMI()

            ax.errorbar([E],[MI],xerr=[dE],yerr=[dMI],
                    linestyle="None",
                    marker = marker,
                    markersize = 12,
                    color = 'k',)

            ax.errorbar([E],[MI],
                    marker = marker,
                    markersize = 8,
                    color = colorMap(point.getValue(P)))

        #ax.set_title("FEMI Index for "+str(modelID)+"("+str(epoch)+" Epochs training)")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Mutual Information")
        ax.grid()

        if makeColorBar:
            fig.colorbar(matplotlib.cm.ScalarMappable(cmap = colorMap),ax = ax,label="AUC Score")
    
    
    fig0,ax0 = plt.subplots()
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    axs = [ax1,ax2]

    enterBenchmarkResults(fig0,ax0)
    enterBenchmarkResults(fig1,axs[0])
    enterBenchmarkResults(fig2,axs[1],False)
    

    xMin,xMax = axs[0].get_xlim()
    yMin,yMax = axs[0].get_ylim() 
    
    nPointsX = int(np.round(xMax-xMin)*resolution)
    nPointsY = int(np.round(yMax-yMin)*resolution)
    
    KNNx = np.linspace(xMin,xMax,nPointsX)
    KNNy = np.linspace(yMin,yMax,nPointsY)

    KNNPred = np.zeros([nPointsX,nPointsY])
    KNNErr = np.zeros([nPointsX,nPointsY])

    for ix in range(0,len(KNNx)):
        for iy in range(0,len(KNNy)):

            Ppred,dPpred = KNNPerformanceEstimate(KNNx[ix],KNNy[iy],epsilon,cloud)
            KNNPred[ix,iy] = Ppred
            KNNErr[ix,iy]=dPpred
    
    ax0.set_title(FEMIType+" FEMI Index")
    fig0.set_figwidth(10)
    fig0.set_figheight(5)
    ax0.set_title("Generalisation of KNN Prediction for Epsilon = "+str(epsilon))
    fig1.set_figwidth(10)
    fig1.set_figheight(5)
    ax0.set_title("Uncertainty of KNN Prediction for Epsilon = "+str(epsilon))
    fig2.set_figwidth(10)
    fig2.set_figheight(5)

    a=.35
    axs[0].imshow(KNNPred.T,cmap = colorMap,extent=(xMin,xMax,yMin,yMax),origin = "lower",aspect = a) 
    img=axs[1].imshow(KNNErr.T,cmap = errColorMap,extent=(xMin,xMax,yMin,yMax),origin = "lower",aspect = a) 
    
    fig2.colorbar(img,ax = axs[1],label="Prediction Uncertenty")
    
    fig0.savefig(figFolder+FEMIType+" Model "+str(modelID)+" Epochs "+str(epoch)+" epsilon "+str(epsilon)+" FEMIIndex.pdf") 
    fig1.savefig(figFolder+FEMIType+" Model "+str(modelID)+" Epochs "+str(epoch)+" epsilon "+str(epsilon)+" Generalisation.pdf") 
    fig2.savefig(figFolder+FEMIType+" Model "+str(modelID)+" Epochs "+str(epoch)+" epsilon "+str(epsilon)+" Uncertainty.pdf") 
    plt.close()

#models = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#epsilons = [3,10]
#types = ["Polar","Component"]

#for m in models:
#    for e in epsilons:
#        for t in types:
#            KNNFemiHeatmapPlot(modelID = m,epoch = 40,epsilon=e,FEMIType=t,resolution=4)

def matrixEvaluation(modelID = 11,epoch = 40,
                                alphas = [0.05,0.1,0.2,0.25,0.5],
                                epsilons=[2,3,4,5,10],
                                FEMIType="Polar"):

    data = loadCSV(modelID,epoch)

    cloud = pointCloudFromDF(data,FEMIType)

    figFolder = "TestPlots/KNNMatrixPlots/" 
    
    KNNCER = np.zeros([len(alphas),len(epsilons)])
    KNNDER = np.zeros([len(alphas),len(epsilons)])
    KNNDCER = np.zeros([len(alphas),len(epsilons)])
    KNNdist = np.zeros([len(alphas),len(epsilons)])
    KNNBetterRandom = np.zeros([len(alphas),len(epsilons)])

    for ix,alpha in enumerate(alphas):
        randDCER =getRandomEstimatorDCER(data,alpha)
        for iy,epsilon in enumerate(epsilons):

            lKNNDCER,lKNNCER,lKNNDER,merr,stderr,dist,disterr = KNNEvaluation(epsilon,alpha,cloud)

            KNNCER[ix,iy] = lKNNCER
            KNNDER[ix,iy] = lKNNDER
            KNNDCER[ix,iy] = lKNNDCER
            KNNdist[ix,iy] = dist
            if lKNNDCER > randDCER:
                KNNBetterRandom[ix,iy] = 1

    def plotMatrix(data,plotBold,xLabel,yLabel,cmap,title):
        fig,ax = plt.subplots()

        print(data)
        ax.matshow(data,cmap=cmap)

        for ix in range(0,len(data)):
            for iy in range(0,len(data[0])):
                value = np.round(data[ix,iy],2)
                if(plotBold[ix,iy]):
                    ax.text(ix,iy,str(value),va="center",ha="center",fontweight="bold")
                else:
                    ax.text(ix,iy,str(value),va="center",ha="center")

        ax.set_xticklabels([0]+xLabel)
        ax.set_ylabel("Alpha")

        ax.set_yticklabels([0]+yLabel)
        ax.set_ylabel("Epsilon")

        plt.title(title)
        plt.show()
    
    plotMatrix(KNNDCER,KNNBetterRandom,alphas,epsilons,cmap=plt.cm.Wistia,title="Classifiers DCER")
    plotMatrix(KNNdist,KNNBetterRandom,alphas,epsilons,cmap=plt.cm.cool,title="Average distance between prediction real value")


def EpsilonVSRandomPlot(modelID = 11,epoch = 40,FEMIType="Polar"):

    figFolder = "TestPlots/KNNMatrixPlots/" 

    alphas = np.linspace(0,0.5,20)
    epsilons = epsilons=[2,4,10,15,30,100,200,10000,1000000]
    
    data = loadCSV(modelID,epoch)

    stdev = data[P].std()

    cloud = pointCloudFromDF(data,FEMIType)
    
    fig,ax = plt.subplots()

    #Random Baseline
    randDCER = [0]*len(alphas)

    for i,a in enumerate(alphas):
        randDCER[i] =getRandomEstimatorDCER(data,a)
    
    ax.plot(alphas,randDCER,linestyle = "dashed",color="gray",label = "Random Guessing")
    ax.plot([stdev,stdev],[0,1],linestyle = "dashed",color = "black",label="St.d. of AUC")
    
    #The KNN-Classifier for diffrent Epsilons
    
    for e in epsilons:

        KNNCER = 0
        KNNDCER = [0]*len(alphas)
        KNNDER = [0]*len(alphas)
        
        for i,a in enumerate(alphas):
            KNNDCER[i],KNNCER,KNNDER[i],merr,stderr,dist,disterr = KNNEvaluation(e,a,cloud)
        
        lines = ax.plot(alphas,KNNDCER,label = "KNN for Epsilon = "+str(e))
        ax.plot([alphas[0],alphas[-1]],[KNNCER,KNNCER],linestyle="dashed",color=lines[0].get_color())
        ax.plot(alphas,KNNDER,linestyle=":",color=lines[0].get_color())

    ax.set_xlabel("Alpha")
    ax.set_ylabel("DCER , DER")
    ax.set_title("Random Guessing- VS KNN- Classifier")
    plt.legend()
    plt.show()




def KNNMSE(epsilon,pointCloud,d = euclidicD,w = sphericalWeight):

    
    realPDistribution = np.zeros(len(pointCloud))
    dRealPDistribution = np.zeros(len(pointCloud))
    predPDistribution = np.zeros(len(pointCloud))
    dPredPDistribution = np.zeros(len(pointCloud))
    Errors = np.zeros(len(pointCloud))
    

    for i in range(0,len(pointCloud)):

        point = pointCloud[i]
        remainingCloud = pointCloud[:i]+pointCloud[i+1:]
        
        E,dE = point.getE()
        MI,dMI = point.getMI()
        
        Preal = point.getValue(P)
        dPreal = point.getValue(dP)
        
        Ppred,dPpred = KNNPerformanceEstimate(E,MI,epsilon,remainingCloud,d,w)
       
        realPDistribution[i] = Preal
        dRealPDistribution[i] = dPreal
        predPDistribution[i] = Ppred
        dPredPDistribution[i] = dPpred
        Errors[i] = np.abs(Ppred-Preal)

    sErrors = np.power(Errors,2)
    MSE = sErrors.mean()
    
    sumUncertainty = dRealPDistribution+dPredPDistribution
    uncCont = sErrors*np.power(sumUncertainty,2)
    dMSE = 2*np.sqrt(uncCont.mean())

    return realPDistribution,predPDistribution,MSE,dMSE

def randomMSE(data):
    singleMSE = np.power(data[P],2)-data[P] + (1/3)
    dSingleMSE = np.power((2*data[P]-1)*data[dP],2)
    return singleMSE.mean(),np.sqrt(dSingleMSE.mean())

def evaluateKNNMSE(modelID=11,epoch=40,FEMIType="Polar"):
    data = loadCSV(modelID,epoch)
    cloud = pointCloudFromDF(data,FEMIType)
    
    figFolder = "TestPlots/KNNBoxplots/"
    resultFolder = "TestResults/MSE/"

    mean =  data[P].mean()
    meanPredErr = np.array((data[P]-mean))
    meanMSE = np.power(meanPredErr,2).mean()
    
    epsilons = [1,1.5,2,3,5,10,100]

    fig,ax = plt.subplots()
    
    Labels = []
    results = []

    csvFile = open(resultFolder+FEMIType+" Model "+str(modelID)+" Epochs "+str(epoch)+"MSE.csv",'w')
    csvFile.write("Classifier\tMSE\n")
    csvFile.write("Mean MSE\t"+str(meanMSE)+"\n")
    csvFile.write("Random MSE\t"+str(randomMSE(data))+"\n")
    
    for i,e in enumerate(epsilons):
        #real DIstribution, predicted DIstribution
        rD,pD,MSE = KNNMSE(e,cloud)
        
        csvFile.write("KNN with R = "+str(e)+"\t"+str(MSE)+"\n")

        if i==0:
            results.append(rD)
            Labels.append("Real")
        results.append(pD)
        Labels.append("R = "+str(e))
            
    csvFile.close()

    ax.boxplot(results)
    ax.set_xticklabels(Labels)
    ax.set_ylabel("Distribution of AUC")
    ax.set_title("Distribution of MSE for diffrent R")
    fig.tight_layout()
    plt.savefig(figFolder+FEMIType+" Model "+str(modelID)+" Epochs "+str(epoch)+" Boxplot.pdf") 
    plt.close()


#models = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#Epochs = [0,20,40]

#for m in models:
#    for e in Epochs:
#        for t in ["Polar","Component"]:
#            evaluateKNNMSE(m,e,t)

def createLaTeXTable(tname,models = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], epsilons = [1,1.5,2,3,5,10,100], epoch = 40):
    
    roundDigits =3

    resultDict = {}

    for model in models:

        data = loadCSV(model,epoch)
        
        meanAUC =  data[P].mean()
        meanPredErr = np.array((data[P]-meanAUC))
        meanMSE = np.power(meanPredErr,2).mean()
        dMeanMSE = 2*np.sqrt((np.power(meanPredErr*data[dP],2)).mean())

        rMSE,dRMSE = randomMSE(data)

        modelResults = {"MeanAUC":np.round(meanAUC,roundDigits),
                        "Random":[np.round(rMSE,roundDigits),np.round(dRMSE,roundDigits)],
                        "MeanCErr":[np.round(meanMSE,roundDigits),np.round(dMeanMSE,roundDigits)]
                        }
        
        bestErr = 1
        dBestErr = 0
        pathToBestErr = []

        toMarkBold = []

        if meanMSE < bestErr:
            bestErr = meanMSE
            dBestErr = dMeanMSE
            pathToBestErr = ["MeanCErr"]
        if rMSE < bestErr:
            bestErr = rMSE
            dBestErr = dMSE
            pathToBestErr = ["Random"]

        NNResults = {}

        for FEMIType in ["Polar","Component"]:
            cloud = pointCloudFromDF(data,FEMIType)
            
            NNResults[FEMIType] = {}
            
            for e in epsilons:
                rD,pD,MSE,dMSE = KNNMSE(e,cloud)
                
                NNResults[FEMIType][e] = [np.round(MSE,roundDigits),np.round(dMSE,roundDigits)]

                if MSE < bestErr:
                    bestErr = MSE
                    pathToBestErr = ["NNResults",FEMIType,e]
        
        toMarkBold.append(pathToBestErr)
        
        def isInErrorInterval(x,dx,y,dy):
            if(x < y):
                return x+dx > y-dy
            else:
                return y+dy > x-dx

        for FEMIType in ["Polar","Component"]:
            for e in epsilons:
                
                err,dErr = NNResults[FEMIType][e][0],NNResults[FEMIType][e][1]
                if isInErrorInterval(err,dErr,bestErr,dBestErr):
                    toMarkBold.append(["NNResults",FEMIType,e])

        if(isInErrorInterval(rMSE,dRMSE,bestErr,dBestErr)):
            toMarkBold.append(["Random"])
        if(isInErrorInterval(meanMSE,dMeanMSE,bestErr,dBestErr)):
            toMarkBold.append(["MeanCErr"])
        

        modelResults["NNResults"] = NNResults
        modelResults["pathToBestErr"] = pathToBestErr
        modelResults["MarkBold"] = toMarkBold
        resultDict[model] = modelResults

    resultFolder = "TestResults/"

    texFile = open(resultFolder+tname+".tex",'w')

    texFile.write("\\begin{tabular}{r"+"|c"*(2*len(models))+"|}\n")
    
    def getEntry(model,path):
 
        pointer = resultDict[model]
            
        #going down the dict
        for key in path:
            pointer = pointer[key]
        return pointer
    
    def shouldBeMarkedBold(path):
        for p in resultDict[model]["MarkBold"]:
            if path == p:
                return True
        return False
   
   #fills digits after floating point
    def convertAndFill(value,numPositions = 3):
        value = str(value)
        while(len(value) < numPositions+2):
            value += "0"
        return value

    def writeBroadLine(lineName,errorKey):
        line = lineName+" & "
        path = [errorKey]

        for model in models:
            
            entry = getEntry(model,path)
            value = convertAndFill(entry[0])+" $\\pm$ "+convertAndFill(entry[1])

            if shouldBeMarkedBold(path):
                line += "\\multicolumn{2}{|c|}{\\bf{"+value+"}} "
            else:
                line += "\\multicolumn{2}{|c|}{"+value+"} "
            
            if not model == models[-1]:
                line += "& "
        
        line += " \\\\\n"
        texFile.write(line)

    def writeNormalLine(r):
        line = "R = "+str(r)+" & "
        
        for model in models:
            for FEMIType in ["Polar","Component"]:
                path = ["NNResults",FEMIType,r]
                entry = getEntry(model,path)
                value = convertAndFill(entry[0])+" $\\pm$ "+convertAndFill(entry[1])
                
                if shouldBeMarkedBold(path):
                    line += "\\bf{"+value+"} "
                else:
                    line += value+" "
                if not (model == models[-1] and  FEMIType == "Component"):
                    line += "& "

        line += " \\\\\n"
        texFile.write(line)
    
    headline = "Error/Model ID & "

    for model in models:
        headline += "\multicolumn{2}{|c|}{"+str(model)+"} "
        if not model == models[-1]:
            headline += "& "
    headline += "\\\\\n"

    texFile.write(headline)
    texFile.write("\\hline ")

    writeBroadLine("Random C. MSE","Random")
    writeBroadLine("Avg. C. MSE","MeanCErr")
    
    texFile.write("\\hline ")
    texFile.write("NN-C. MSE & "+" P: & C: & "*(len(models)-1)+" P: & C: \\\\\n")
    texFile.write("\\hline ")

    for r in epsilons:
        writeNormalLine(r)

    texFile.write("\\end{tabular}")
    texFile.close()    

            

#createLaTeXTable("MSETable")
#createLaTeXTable("MSE1",[0,1,2,3,4,5])
#createLaTeXTable("MSE2",[6,7,8,9,10,11])
#createLaTeXTable("MSE3",[12,13,14])
createLaTeXTable("MSE1_1",[0,1,2,3,4])
createLaTeXTable("MSE2_1",[5,6,7,8,9])
createLaTeXTable("MSE3_1",[10,11,12,13,14])
#createLaTeXTable("MSE1_2",[0,1,2])
#createLaTeXTable("MSE2_2",[3,4,5])
#createLaTeXTable("MSE3_2",[6,7,8])
#createLaTeXTable("MSE4_2",[9,10,11])
#createLaTeXTable("MSE5_2",[12,13,14])
createLaTeXTable("MSEResults",[3,10])
createLaTeXTable("MSEResultsRNN",[4,5])
createLaTeXTable("MSEResultsTransformer",[13,14])
