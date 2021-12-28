#!/bin/bash

black .
isort *.py
flake8 .