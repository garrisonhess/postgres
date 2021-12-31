#!/bin/bash

black *.py
isort *.py
flake8 *.py
mypy plan_diff.py
pylint plan_diff.py
