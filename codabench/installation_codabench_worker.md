# Installation codabench
## Prerequisite: installation of docker
https://docs.docker.com/engine/install/ubuntu/

## Installation of Codabench worker image

`sudo docker pull codalab/competitions-v2-compute-worker`

## Create a private queue on Codabench

Copy the broker URL and replace `@rabbit:5672` with `@codalabcomp-v2-prod.lri.fr:5672`,
for example: for the `lips_exaion` queue
`pyamqp://55efc5fd-dce7-4f3f-8e77-ed6903355312:4686f918-8e96-4e1f-9e48-23496b826ee9@codalabcomp-v2-prod.lri.fr:5672/90f820bb-c80c-4c58-bb8e-45cfd6a12e12`

## Make a file .env containing
```
# Queue URL (defined on codabench web site)
BROKER_URL=pyamqp://55efc5fd-dce7-4f3f-8e77-ed6903355312:4686f918-8e96-4e1f-9e48-23496b826ee9@codalabcomp-v2-prod.lri.fr:5672/90f820bb-c80c-4c58-bb8e-45cfd6a12e12

# Location to store submissions/cache -- absolute path!
HOST_DIRECTORY= /home/ubuntu/codabench/storage

# If SSL is enabled, then uncomment the following line
#BROKER_USE_SSL=True
```

## Creation of a script to start the docker container (CPU worker)
```
docker run \
    -v /home/ubuntu/codabench/storage:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:latest
```

## Optional: retrieve lips docker image (save time - otherwise retrieved during first execution of the benchmark)

`sudo docker pull jeromepicault/lips:0.1`

## Installation worker Codabench in GPU mode

### Installation of docker-nvidia
This requires docker-nvidia
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```
### Creation of a script to start the docker container (GPU worker)
```
sudo nvidia-docker run \
    -e NVIDIA_VISIBLE_DEVICES=3 \
    -v /home/ubuntu/codabench/storage:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /var/lib/nvidia-docker/nvidia-docker.sock:/var/lib/nvidia-docker/nvidia-docker.sock \
    -d \
    --env-file .env \
    --name codabench_gpu \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:nvidia
```

Note the option NVIDIA_VISIBLE_DEVICES to select which GPU to use.

The GPU selection can be checked for example like this:

`nvidia-docker run  -e NVIDIA_VISIBLE_DEVICES=3 nvidia/cuda:10.0-base nvidia-smi`



