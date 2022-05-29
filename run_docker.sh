#!/bin/bash
IMG="$1"
NICE_SLAM_REPO_PATH="$2"

if [ -z "$1" ]
then
      echo "Docker image cannot be empty"
      exit
fi

if [ -z "$2" ]
then
      echo "NICE_SLAM_REPO_PATH cannot be empty"
      exit
fi

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<< "$xauth_list")
    if [ ! -z "$xauth_list" ]
    then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi


DOCKER_OPTS="$DOCKER_OPTS --gpus all"
  	
docker run -it \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=$XAUTH \
  -v "$XAUTH:$XAUTH" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$NICE_SLAM_REPO_PATH:/home/developer/nice-slam-cpp" \
  --network host \
  --rm \
  --privileged \
  --security-opt seccomp=unconfined \
  $DOCKER_OPTS \
  $IMG \
  bash
