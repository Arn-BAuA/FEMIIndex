
###
# This class is sort of the "Prototype" for all experiments. It dictates, how the interface for an experiment should look like;
# The Experiment takes args in its init mehtod. If the first arg in args is "--plot" the file goes in a directory, which
# must be specified by the developer upon overwriting the interface and looks for the results, and plots them according to the arguments.
#If --plot is not specified, the experiment is conducted and stuff is filled in the Result forlder specified by the user.

import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import json


mainResultDir = "Results/" 
mainPlotDir = "Plots/"

class Experiment():

    def __init__(self,resultPath,args):
        
        self.resultPath = mainResultDir + resultPath
        self.plotPath = mainPlotDir + resultPath
        
        if not os.path.isdir(self.resultPath):
            print("Create Dir ",self.resultPath)
            os.mkdir(self.resultPath)
        if not os.path.isdir(self.plotPath):
            print("Create Dir ",self.plotPath)
            os.mkdir(self.plotPath)
        
        if len(args) > 1:
            if args[1] == "--plot":
                if len(args) > 2:
                    self._plot(args[2:])
                else:
                    raise ValueError("No path for results to plot specified.")
            else:
                self._conduct(args[1:])
        else:
            self._conduct([])

    def _conduct(self,args):
        pass

    def _plot(self,args):
        pass


    #################################
    # Some Experiment Utility Methods to use in the subclasses
    ##

    #################################
    # Some Plot Utility Methods to use in the subclasses
    ##

    def _fullPlotAgainstVariedQuantity(self,
                                        variedQuantity,
                                        quantityLabel,
                                        oneDPlotTitle,
                                        twoDPlotTitle,
                                        args,
                                        colorMap=plt.cm.viridis,
                                        autosave=True,
                                        showFig=False): 
        types = ["Component","Polar"]    
        quantitys = ["E","MI"]
        labelMapping = {"E":"Entropy","MI":"Mutual Information"}
        markers = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d",4,5,6,7,8,9,10,11]

        #Loading all the data
        data = [0]*len(args)
        metaData = [0] * len(args)
 
        for i,path in enumerate(args):

            #Remove Postfix
            split = path.split('.')
            if not len(split) == 1:
                split = split[:-1]
                
                path = ""
                for s in split:
                    path += s+'.'

                path = path[:-1] #rmoving extra dot at the end

            
            data[i] = pd.read_csv(path+".csv",sep = '\t')
            metaDataFile = open(path+".json",'r')
            metaData[i] = json.load(metaDataFile)
            metaDataFile.close()

        for t in types:
            #1D Plots
            for quantity in quantitys:
                fig,ax = plt.subplots() 
    
                for i in range(0,len(data)):

                           ax.errorbar(data[i][variedQuantity],
                                data[i][quantity+"_"+t],
                                yerr=data[i]["Delta_"+quantity+"_"+t],
                                label=metaData[i]["DataSetName"],
                                linestyle = ":",
                                marker = markers[i%len(markers)]
                                )
                    
                #plt.xticks(ticks = np.arange(len(data["Window Length"])),labels =data["Window Length"])
                ax.set_title(oneDPlotTitle)
                ax.legend(loc="center left",bbox_to_anchor=(1,0.5))
                ax.set_xlabel(quantityLabel)
                ax.set_ylabel(t+" "+labelMapping[quantity])
                
                pos = ax.get_position()
                ax.set_position([pos.x0,pos.y0,pos.width*0.8,pos.height])
                fig.set_figwidth(15)
                fig.set_figheight(10)


                if autosave:
                    plt.savefig(self.plotPath+oneDPlotTitle+" "+t+" "+labelMapping[quantity]+" VS "+quantityLabel+".pdf")
                if showFig:
                    plt.show()
                plt.close()
            
            fig,ax = plt.subplots()
            
            #2D Plot
            for i in range(0,len(data)):
                for j in range(1,len(data[i][variedQuantity])):
                    ax.plot(data[i]["E_"+t].iloc[j-1:j+1],
                            data[i]["MI_"+t].iloc[j-1:j+1],
                            marker = markers[i%len(markers)],
                            markersize = 12,
                            linestyle = ":",
                            color="k",)
                for j in range(0,len(data[i][variedQuantity])):
                    ax.errorbar(
                            [data[i]["E_"+t].iloc[j]],
                            [data[i]["MI_"+t].iloc[j]],
                            xerr = [data[i]["Delta_E_"+t].iloc[j]],
                            yerr = [data[i]["Delta_MI_"+t].iloc[j]],
                            #color = colorMap(data[i]["Window Length"].iloc[j])
                            color = colorMap(float(j)/float(len(data[i][variedQuantity]))),
                            marker = markers[i%len(markers)],
                            markersize = 8,
                        )
                
            ax.set_title(twoDPlotTitle+" for "+t+" FEMI-Index")
            ax.set_xlabel("Entropy")
            ax.set_ylabel("Mutual Information")

            colorLegendHandles = []
        
            #We assume its the same for all data points
            nWindows = len(data[0][variedQuantity])
            colorLegendHandles.append(mlines.Line2D([],[],marker="None",linestyle = "None",label = quantityLabel))
            for i in range(0,nWindows):
                colorLegendHandles.append(
                        mlines.Line2D([],[],
                                       color = colorMap(float(i)/float(nWindows)),
                                       marker="s",
                                       markersize = 10,
                                       linestyle = "None",
                                       label = str(data[0][variedQuantity].iloc[i])
                            )
                        )

            colorLegend = plt.legend(handles = colorLegendHandles,loc="upper left", bbox_to_anchor=(1,1))
            plt.gca().add_artist(colorLegend)
            #Data Set Legend
            dsLegendHandles = []
            #THis functions as headline for the legend. #jank
            dsLegendHandles.append(mlines.Line2D([],[],marker="None",linestyle = "None",label = "Data Set:"))
            for i in range(0,len(metaData)):
                dsLegendHandles.append(
                        mlines.Line2D([],[],
                                       color = 'k',
                                       marker=markers[i%len(markers)],
                                       markersize = 10,
                                       linestyle = "None",
                                       label = str(metaData[i]["DataSetName"])
                            )
                        )

            plt.legend(handles = dsLegendHandles,loc="lower left",bbox_to_anchor=(1,0))
           
            ax.set_position([pos.x0,pos.y0,pos.width*0.8,pos.height])
            fig.set_figwidth(15)
            fig.set_figheight(10)

            if autosave:
                plt.savefig(self.plotPath+twoDPlotTitle+" for "+t+" FEMI-Index w. diffrent "+quantityLabel+".pdf")
            if showFig:
                plt.show()
            plt.close()

