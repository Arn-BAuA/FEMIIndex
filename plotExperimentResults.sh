#!/bin/bash
cp Experiments/$1 .
python $1 --plot $2/*.csv; 
rm $1

