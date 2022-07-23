#!/bin/bash

# you should be in valid node version and stuff
(cd ./sketch/api/tailwindcss && npx tailwindcss -i ./src/input.css -o ../static/output.css --watch) &
PID1=$!

# You should be in a valid python environment
(cd ./sketch && DEBUG=True uvicorn sketch.api.main:app --reload) &
PID2=$!

wait $PID1
wait $PID2