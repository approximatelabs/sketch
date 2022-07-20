#!/bin/bash

trap 'kill $(jobs -p)' SIGINT

kill $(lsof -t -i:8000)
# you should be in valid node version and stuff
(cd ./sketch/api/tailwindcss && npx tailwindcss -i ./src/input.css -o ../static/output.css --watch &)
# You should be in a valid python environment
DEBUG=True uvicorn sketch.api.main:app --reload &

