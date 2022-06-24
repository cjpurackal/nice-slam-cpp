# nice-slam-cpp(WIP)
A cpp implementation of NICE-SLAM
## Installation

```
mkdir docker_deps
cp libcudnn8_8.2.2.26-1+cuda10.2_amd64.deb libcudnn8-dev_8.2.2.26-1+cuda10.2_amd64.deb docker_deps/
sudo docker build --build-arg user_id=99 -t nscpp:dev .
```

## Run the container
```
./run_docker nscpp:dev /path/to/nice-slam-cpp
```

## Reference
https://github.com/cvg/nice-slam
