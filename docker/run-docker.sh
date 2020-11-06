#!/bin/bash

source init-docker.sh

docker run \
	-h ${CONT_HOST} \
	-it \
	--rm \
	--runtime=nvidia \
	--name ${CONTAINER} \
	-p 8899:8899 \
	--net=host \
	--ipc=host \
	--mount type=bind,source="$(pwd)"/..,target=/home/trooper/src \
	--mount type=bind,source=${DATA_VOLM},target=/home/trooper/data \
        -v /etc/timezone:/etc/timezone:ro \
        -v /etc/localtime:/etc/localtime \
	${IMAGENAME}