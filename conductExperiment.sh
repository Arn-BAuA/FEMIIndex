#!/bin/bash

cp Experiments/$1 .
python $1
rm $1
