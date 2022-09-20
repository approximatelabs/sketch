#!/bin/bash

# Bash command to convert all jupyter notebooks to have no output
# find . -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;

# Bash command to convert all jupyter notebooks in git diff to have no output
git diff --name-only --diff-filter=ACMRTUXB | grep ".ipynb" | xargs -I {} jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {}