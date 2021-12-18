#!/bin/bash

# Generate the data
./datagen/datagen.py

# Train, evaluate, and serialize models
./train.py
