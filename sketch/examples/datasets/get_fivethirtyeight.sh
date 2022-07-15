#/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

wget https://fivethirtyeight.datasettes.com/fivethirtyeight.db $SCRIPT_DIR