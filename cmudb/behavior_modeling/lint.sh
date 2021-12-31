#!/bin/bash

black *.py
isort *.py
flake8 *.py
mypy *.py
# bandit *.py
pylint *.py
