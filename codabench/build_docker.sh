#!/bin/bash
IMAGE_NAME=lips:0.3
export DOCKER_HUB_ID=jeromepicault/lips:0.3
docker build --build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTPS_PROXY -t $IMAGE_NAME --build-arg HTTP_PROXY docker/
docker tag $IMAGE_NAME $DOCKER_HUB_ID

