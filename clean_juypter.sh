#!/bin/bash

# Bash command to convert all jupyter notebooks to have no output
find . -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;
