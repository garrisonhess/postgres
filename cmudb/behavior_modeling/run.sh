#!/bin/bash

# generate data
./datagen.py

# perform differencing
./differencing.py 

# train-evaluate-serialize models
./train.py --config_name nodiff
./train.py --config_name diff
