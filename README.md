#Fourier Entropy Mutual Information (FEMI)-Index

Todo: Banner Hier.

##About This Repostory and Basic Overview:

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

## Contents of this repository

This repository uses some conventions and objects 
The main computation of the FEMI-index is done in "FEMIDataIndex.py".

## How to run the experiments

## Shameless Plug

Example Method calls:

To Plot AUC-Score form benchmark results and FEMI-Index Calculation:
./conductExperiment.sh AUCScorePlots.py


To create Plots and Baselines for the noise evaluation
./conductExperiment.sh Noise.py

