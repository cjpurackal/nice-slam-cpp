# Ubuntu 18.04 with nvidia-docker2 beta opengl support
FROM nvidia/cudagl:10.2-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Tools I find useful during development
RUN apt-get update  \
 && apt-get install -y  \
        build-essential \
        bwm-ng \
        cmake \
        cppcheck \
        gdb \
        git \
        libbluetooth-dev \
        libcwiid-dev \
        libgoogle-glog-dev \
        libspnav-dev \
        libusb-dev \
        lsb-release \
        python3-dbg \
        python3-empy \
        python3-numpy \
        python3-setuptools \
        python3-pip \
        python3-venv \
        software-properties-common \
        sudo \
        vim \
        wget \
        net-tools \
        iputils-ping \
 && apt-get clean 

# Add a user with the same user_id as the user outside the container
# Requires a docker build argument `user_id`
ARG user_id
ENV USERNAME developer
RUN useradd -U --uid ${user_id} -ms /bin/bash $USERNAME \
 && echo "$USERNAME:$USERNAME" | chpasswd \
 && adduser $USERNAME sudo \
 && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

# Commands below run as the developer user
USER $USERNAME

# Make a couple folders for organizing docker volumes
RUN mkdir ~/workspaces ~/other 
RUN mkdir ~/datasets


# When running a container start in the developer's home folder
WORKDIR /home/$USERNAME

RUN export DEBIAN_FRONTEND=noninteractive \
 && sudo apt-get update -qq \
 && sudo -E apt-get install -y -qq \
    tzdata \
 && sudo ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
 && sudo dpkg-reconfigure --frontend noninteractive tzdata \
 && sudo apt-get clean -qq


RUN sudo apt-get install -y -qq \
    apt-transport-https \
    curl \
    ca-certificates \
 && sudo apt-get clean -qq

 
RUN curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add - \
 && sudo add-apt-repository "deb https://download.sublimetext.com/ apt/stable/" \
 && sudo apt update -qq \
 && sudo apt install -y -qq sublime-text 


RUN sudo apt install -y libpython3-dev

WORKDIR /home/$USERNAME/

RUN sudo apt install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev freeglut3-dev libglew-dev libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev

RUN mkdir ~/deps \
&& cd ~/deps \
&& git clone --single-branch --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git \
&& cd Pangolin \
&& mkdir build \
&& cd build \
&& cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON \
&& make -j8 \
&& sudo make install 


RUN cd ~/deps/ \
&& wget https://hyperrealm.github.io/libconfig/dist/libconfig-1.7.3.tar.gz \
&& tar -xvf libconfig-1.7.3.tar.gz \
&& cd libconfig-1.7.3 \
&& ./configure \
&& sudo make install 

RUN cd ~/deps/ \
&& wget https://github.com/sharkdp/fd/releases/download/v8.3.2/fd-musl_8.3.2_amd64.deb \
&& sudo dpkg -i fd-musl_8.3.2_amd64.deb

RUN cd ~/deps \
&& git clone https://github.com/Itseez/opencv.git \
&& git clone https://github.com/Itseez/opencv_contrib.git \
&& cd opencv \
&& mkdir release \
&& cd release \
&& cmake -D BUILD_TIFF=ON -D WITH_CUDA=OFF -D ENABLE_AVX=OFF -D WITH_OPENGL=OFF -D WITH_OPENCL=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_TBB=ON -D WITH_EIGEN=OFF -D WITH_V4L=OFF -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D OPENCV_EXTRA_MODULES_PATH=/home/developer/deps/opencv_contrib/modules .. \
&& sudo  make -j4 \
&& sudo make install

#install cudnn
COPY ./docker_deps/libcudnn8_8.2.2.26-1+cuda10.2_amd64.deb /home/developer/deps/libcudnn8_8.2.2.26-1+cuda10.2_amd64.deb
COPY ./docker_deps/libcudnn8-dev_8.2.2.26-1+cuda10.2_amd64.deb /home/developer/deps/libcudnn8-dev_8.2.2.26-1+cuda10.2_amd64.deb

RUN cd ~/deps/ \
&& sudo dpkg -i ~/deps/libcudnn8_8.2.2.26-1+cuda10.2_amd64.deb \
&& sudo dpkg -i ~/deps/libcudnn8-dev_8.2.2.26-1+cuda10.2_amd64.deb

RUN sudo apt-get install unzip

#install libtorch
RUN cd ~/deps \
&& wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu102.zip \
&& sudo unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu102.zip

# install python opencv
# update pip
RUN sudo python3 -m pip install --upgrade pip \
&& sudo pip3 install -U --timeout 1000 opencv-python

#install pytorch
RUN sudo pip3 install -U --timeout 2000 torch torchvision

