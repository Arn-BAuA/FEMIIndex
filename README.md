# Fourier Entropy Mutual Information (FEMI)-Index

Todo: Banner Hier.

## About This Repostory and Basic Overview:

This repository contains the code to compute the FEMI-index along side experiment code for its validation. The FEMI-Index
is a way to mathematically assign and index to a time series data set. Ideally, this index gives a hint how 
difficult the calssification of the anomalies contained in the data set is for a given anomaly detection algorithm. <br>
this repository is structured as follow:

## On the computation of the FEMI-index

FEMI-index is a measure to cluster or odrer data sets for time series anomaly detection. The aim here is to create an index that is associated with the difficulty to find the anomalys in the data. Further constraints are, that this index has to be easy to compute and should not requere many assumptions on the data.<br>
To fulfill these needs, concepts of information theory are used. The index requires to sets of data. One containing only normal data, the other one containing normal data along side anomalys.<br> 
The entropy of the normal data is computed. By this, we are asking: "How much information is there to be learned to characterize the normal state."<br>
Than, the mutual information between the normal data and the data containing anomalies is created, asking: "How much informaion separates normal and abnormal states?".<br>
Since dooing these operations on the time domain samples would ignore the time series nature of the FEMI-index, the fourier transform of normal and abnormal data is taken and the fourier transformed data is used for the experimnts.<br>

Todo: Some Figures here.

## Contents of this repository

This repository uses some conventions and objects defined in our other repository: https://github.com/Arn-BAuA/TimeSeriesAEBenchmarkSuite.
It is used as a submodule to source data sets, models and the framework for the experimentation. <br>
The main computation of the FEMI-index is done in "FEMIDataIndex.py".<br>
### The Benchmark Experiments
The ModelComparisonBenchmark.py is a scipt that coordinates a huge benchmark of multiple models on multiple datasets. Be carefull when running it. Since extended information is logged for every experiment (e.g. examples where the trained algorithm performed good or bad on along side model weights), the data generated here can easily exceed hundrets of gigabytes storage.
### The ExperimentClass:
To Evalueate the FEMI-Index, several experiments can be run using this repository. Some of them rely on the abstract interface defined in "ExperimentClass.py". It provides functionality that handles the iteration over the benchmark data and the management of the experiments, in addition to a generic plot script to viusalize the results. To conduct the experiments using this interface, a "set dispenser" method. THat is a lambda function that needs to be specified when implementing the interface. It outputs the datasets for the experimentation.<br>
The experiment class also defines a default command line interface for calling experiments.
### Subclasses of the Experiment Class:
There are some subclasses implementing the interface provided by the experiment class. They are stored (along other experiments) in the Experiment subfolder. In addition, a blank experiment is stored in the folder in the file "blank.py" for You to implement your own once.<br>
#### Noise.py
Noise.py is an experiment that floods the data with noise and evaluates how the FEMI-Index changs with noise
#### Stability.py
This experiment averages the data with multible window lengths for the averaging. The influences on the FEMI-Index are evaluated.
### Other Experiments in the Repository
There are other experiments in the repository that don't conform to the interface. For the future, it is planned to migrate everything to be conform to a common interface, but for now we document them as is:
#### Experiments/AUCScorePlots
This experiments associates the AUC-Scores from the ModelComparisonBenchmark.py with the FEMI-Indices of the datasets using baselines that where recorded by the ExperimentClass. In addition, it saves these associataions as .csv.
#### Experimetns/FEMI-Evaluation.py and Experiments/NoiseOld.py
These are some of the first experiments which will eventually be deleted, since the things they do are also archivable using newer Experiments, that implement the "ExperimentClass" interface.
###The NN-Classifier:
To investigate the performance of the FEMI-Index, a NN-Classifier that uses the index for model performance estimation has been developed. That classifier is compared against two baselines. The classifier, the baselines and all the experiments and plots for these are in "KNNExperiment.py". Eventually this will be refactored, to be more modular and fit the interfaces.

## How to run the experiments
Since scoping in python and imports work the way they do, we decided to write some little helper scripts.
the conductExperiment.sh file is used to conduct an experiment that is stored in Experiments/\*. All that it does is copy the experiment to the root dir of the repo, run it in an interpreter and than delete the copy. This is done, so that no problems with the imports ariase when coding here. ALso meaning, that, when writing on a experiment file, all imports are specified from the root dir.


### Example Method calls:

To Plot AUC-Score form benchmark results and FEMI-Index Calculation:
<pre><code>
./conductExperiment.sh AUCScorePlots.py
</pre></code>


To create Plots and Baselines for the noise evaluation
</pre></code>
./conductExperiment.sh Noise.py
</pre></code>
## Shameless Plug
