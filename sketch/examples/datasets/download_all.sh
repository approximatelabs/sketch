#/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo $SCRIPT_DIR
wget -nc -i $SCRIPT_DIR/sqlites.txt --directory-prefix=$SCRIPT_DIR