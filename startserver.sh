#!/bin/bash

trap 'kill $(jobs -p)' SIGINT

# you should be in valid node version and stuff
(cd ./sketch/api/tailwindcss && npx tailwindcss -i ./src/input.css -o ../static/output.css --watch &)
PID1=$!

echo $PID1
# You should be in a valid python environment
DEBUG=True uvicorn sketch.api.main:app --reload &
PID2=$!

wait $PID1
wait $PID2