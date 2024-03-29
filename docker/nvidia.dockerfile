FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME 
RUN groupmod -g $USER_GID $USER_NAME

WORKDIR /home/${USER_NAME}
ENV USER_HOME_DIR=/home/${USER_NAME}
ENV HOME=${USER_HOME_DIR}

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    make \
    cmake \
    curl \
    git \
    wget \
    gfortran \
    software-properties-common \
    net-tools \
    ffmpeg \
    unzip \
    vim \
    xserver-xorg-dev \
    python-tk \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends libboost-all-dev    

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg-dev \
    libprotobuf-dev \
    libqt5multimedia5 \
    libqt5x11extras5 \
    libtbb2 \
    libtheora0 \
    libyaml-cpp0.5v5 \
    libpng-dev \
    qtbase5-dev \
    zlib1g && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

### ROS stuff below
# install packages
RUN apt-get update && apt-get install -q -y  \
  dirmngr \
  gnupg2 \
  lsb-release \
  && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install ros packages
ENV ROS_DISTRO kinetic

RUN apt-get update && apt-get install -y  \
  ros-kinetic-ros-core=1.3.2-0* \
  && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update && apt-get install -y  \
  git-core \
  python-argparse \
  python-wstool \
  python-vcstools \
  python-rosdep \
  python-dev \
  python-numpy \
  python-pip \
  python-setuptools \
  python-scipy \
  ros-kinetic-control-msgs && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y  \
  gazebo7 \
  ros-kinetic-qt-build \
  ros-kinetic-gazebo-ros-control \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-ros-control \
  ros-kinetic-control-toolbox \
  ros-kinetic-realtime-tools \
  ros-kinetic-ros-controllers \
  ros-kinetic-xacro \
  ros-kinetic-tf-conversions \
  ros-kinetic-python-orocos-kdl \
  ros-kinetic-orocos-kdl \
  ros-kinetic-kdl-parser-py \
  ros-kinetic-kdl-parser \
  ros-kinetic-moveit-simple-controller-manager \
  ros-kinetic-trac-ik \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y  \
  ros-kinetic-robot-state-publisher \
  && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip==9.0.3

# Catkin
RUN  pip install rospkg
RUN  pip install -U catkin_tools

# bootstrap rosdep
RUN rosdep init \
  && rosdep update

RUN apt-get update && apt-get install -y  \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-gazebo-ros-control \
&& rm -rf /var/lib/apt/lists/*

# Replacing shell with bash for later docker build commands
RUN mv /bin/sh /bin/sh-old && \
  ln -s /bin/bash /bin/sh

# create catkin workspace
ENV CATKIN_WS=${USER_HOME_DIR}/catkin_ws
RUN source /opt/ros/kinetic/setup.bash
RUN mkdir -p $CATKIN_WS/src
WORKDIR ${CATKIN_WS} 
RUN catkin init
RUN catkin config --extend /opt/ros/$ROS_DISTRO \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=False 
WORKDIR $CATKIN_WS/src

RUN apt-get update && \
  apt-get install -y ros-${ROS_DISTRO}-moveit-* && \
  rm -rf /var/lib/apt/lists/*

# install dependencies for RGBD camera calibration
RUN apt-get update && \
    apt-get install -y \
    wget \
    software-properties-common \
    ros-kinetic-rgbd-launch \
    ros-kinetic-visp-hand2eye-calibration \
    ros-kinetic-ddynamic-reconfigure && \
    rm -rf /var/lib/apt/lists/*

# keys and installs for realsense SDK
RUN apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key && \
    add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

RUN apt-get update && \
    apt-get install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg && \
    rm -rf /var/lib/apt/lists/*

# clone repositories into workspace and build
WORKDIR ${CATKIN_WS}/src

RUN git clone https://github.com/IntelRealSense/realsense-ros.git && \
    cd realsense-ros/ && \
    git checkout `git tag | sort -V | grep -P "^\d+\.\d+\.\d+" | tail -1` && \
    cd .. && \
    git clone https://github.com/pal-robotics/aruco_ros.git 

# build
WORKDIR ${CATKIN_WS}
RUN catkin build

# copy over airobot repositoriy
WORKDIR ${USER_HOME_DIR}
RUN git clone -b qa https://github.com/Improbable-AI/airobot.git && \ 
    cd airobot && pip install -e . && cd ..

# bashrc ros source and CODE_BASE env variable for python imports
RUN echo 'source $HOME/catkin_ws/devel/setup.bash' >> ${HOME}/.bashrc
RUN echo 'export CODE_BASE="$HOME"' >> ${HOME}/.bashrc

RUN apt-get update && apt-get install -y \
    protobuf-compiler

COPY ./requirements.txt ${USER_HOME_DIR}
WORKDIR ${USER_HOME_DIR}
RUN pip install -r requirements.txt

WORKDIR ${USER_HOME_DIR}
RUN git clone --recurse -b master https://github.com/mcubelab/mpalm_affordances.git && \
    cd mpalm_affordances/catkin_ws/src && \
    cp -r * ${CATKIN_WS}/src

WORKDIR ${CATKIN_WS}
RUN catkin build config pykdl_utils

# Exposing the ports
EXPOSE 11311

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# change ownership of everything to our user
RUN cd ${USER_HOME_DIR} && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .

# setup entrypoint
COPY ./entrypoint.sh /
WORKDIR /
ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]
