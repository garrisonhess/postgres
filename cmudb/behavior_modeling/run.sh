#!/bin/bash

# generate data, perform differencing, then train-evaluate-serialize models
./datagen.py && ./differencing.py && ./train.py
