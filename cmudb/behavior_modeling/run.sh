#!/bin/bash

# generate data, perform differencing, then train-evaluate-serialize models
./datagen.py

./differencing.py 

./train.py --config_name nodiff

./train.py --config_name diff