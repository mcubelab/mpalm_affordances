FROM mpalm-dev-cpu

ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN rm /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts

RUN /bin/bash -c "cd /tmp/ && git clone https://github.com/lcm-proj/lcm && cd lcm && mkdir build && cd build && cmake .. && make && make install"
RUN /bin/bash -c "cd /tmp/ && git clone https://github.com/mcubelab/libbot.git && cd libbot && make BUILD_PREFIX=/usr"
