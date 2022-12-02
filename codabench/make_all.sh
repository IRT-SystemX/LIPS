#!/bin/bash
./build_docker.sh
./publish_docker.sh
./make_bundle.sh
./make_submission.sh
