# LIPS Benchmark on Codabench

This document briefly describes how to publish the LIPS benchmark on Codabench (http://www.codabench.org).

## Codabench account
- create  a  Codabench account

## Prepare a docker image

This image is the one used to run the benchmark.
It shall contain the LIPS package and its dependencies.

### Dockerfile

See `docker/Dockerfile` for the details of the content of the image.

The main steps consists in:
- adding the necessary linux commands
- installing the required Python packages

The size of the image should be as small as possible - for ex. use the option `--no-cache-dir` when using `pip install`.


It has to be regenerated depending on LIPS code changes.

In order to automate the docker image creation, please use the script `build_docker.sh`

### Publication on Docker hub
Use the script `./publish_docker.sh`.

This requires an account on docker hub.

_TODO_: use an institutional account (or a shared account) for better docker image management.

## Preparation of Codabench bundle

The bun

### `competition.yaml` config file

Both programs ingestion and scoring communicates through a shared memory (file system).
We have to ensure that ingestion is finished (esp. if ingestion includes a training phase) before looking at results.

### Ingestion program

### Scoring progrom
