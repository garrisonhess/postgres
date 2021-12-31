#!/bin/bash

black *.py
isort *.py
flake8 *.py
mypy differencing.py
pylint differencing.py
