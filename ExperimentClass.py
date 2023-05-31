
###
# This class is sort of the "Prototype" for all experiments. It dictates, how the interface for an experiment should look like;
# The Experiment takes args in its init mehtod. If the first arg in args is "--plot" the file goes in a directory, which
# must be specified by the developer upon overwriting the interface and looks for the results, and plots them according to the arguments.
#If --plot is not specified, the experiment is conducted and stuff is filled in the Result forlder specified by the user.

import os

mainResultDir = "Results/" #../, since the instances of the Experiment are stored in a subfolder of the repo.

class Experiment():

    def __init__(self,resultPath,args):
        
        self.resultPath = mainResultDir + resultPath
        
        if not os.path.isdir(self.resultPath):
            print("Create Dir ",self.resultPath)
            os.mkdir(self.resultPath)
        
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
