#!/bin/bash

black .
isort .
flake8 .
mypy ./src/plans/
pylint ./src/plans/
