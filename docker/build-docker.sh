#!/bin/bash

source init-docker.sh

docker build -t "${IMAGENAME}" --build-arg USER_ID=$USERID -f Dockerfile ..
