from .ExperimentClass import Experiment

#Describe your experiment

class NoiseExperiment(Experiment):

    def __init__(self,args):
        Experiment.__init__("NoiseExperiment",args)

    def _conduct(self,args):
        #Code to conduct experiment goes here.
        pass

    def _plot(self,args):
        #Code to plot experimental Results goes here.
        pass
