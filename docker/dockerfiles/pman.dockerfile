FROM fhogan/mpalm-dev-cpu2

#get ssh access right
ARG SSH_PRIVATE_KEY
#RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN rm /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts


RUN apt-get update 
# download tactile dexterity dependencies
RUN apt-get install -y \
			python-pip \
			protobuf-compiler \
			protobuf-c-compiler \
			python-matplotlib 
			
RUN pip install pyftpdlib \
				scipy \
				shapely \
				numpy-stl \
				numba \
				numpy-quaternion \
				trimesh \
				python-fcl \
				dubins \
				sympy \
				quadpy
				
#download pman dependencies
RUN apt-get install -y \
			default-jdk \
			python-gobject \
			python-gtk2 \
			libcanberra-gtk-module \
			terminator \
			snapd 
			
#Install PyCharm Editor
			
RUN /bin/bash -c  "add-apt-repository ppa:mystic-mirage/pycharm && \
	apt-get update && \
	apt-get install pycharm-community"
			
# download ros dependencies
RUN apt-get install -y \
			ros-kinetic-control-toolbox \
			ros-kinetic-controller-interface \
			ros-kinetic-controller-manager \
			ros-kinetic-effort-controllers \
			ros-kinetic-force-torque-sensor-controller \
			ros-kinetic-gazebo-ros-control \
			ros-kinetic-joint-limits-interface \
			ros-kinetic-robot-state-publisher \
			ros-kinetic-joint-state-publisher \
			ros-kinetic-joint-state-controller \
			ros-kinetic-joint-trajectory-controller \
			ros-kinetic-moveit-commander \
			ros-kinetic-moveit-core \
			ros-kinetic-moveit-planners \
			ros-kinetic-moveit-ros-move-group \
			ros-kinetic-moveit-ros-planning \
			ros-kinetic-moveit-ros-visualization \
			ros-kinetic-moveit-simple-controller-manager \
			ros-kinetic-position-controllers \
			ros-kinetic-rqt-joint-trajectory-controller \
			ros-kinetic-transmission-interface \
			ros-kinetic-velocity-controllers \
			ros-kinetic-trac-ik \
			ros-kinetic-simple-message 

# Fixing pyassimp bug (https://github.com/assimp/assimp/issues/1209) updating to newer version
# But not latest, because of another bug (https://github.com/assimp/assimp/issues/2343)
RUN pip install pyassimp==4.1.3 --upgrade

# For running and debugging remotely from host
RUN apt-get install -y ssh net-tools
RUN sed -i 's/prohibit-password/yes/' /etc/ssh/sshd_config

#download software for pman
#RUN /bin/bash -c "cd /tmp/ && git clone https://github.com/lcm-proj/lcm && cd lcm && mkdir build && cd build && cmake .. && make && make install"
#RUN /bin/bash -c "cd /tmp/ && git clone https://github.com/mcubelab/libbot.git && cd libbot && make BUILD_PREFIX=/usr"
