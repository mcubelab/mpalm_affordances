#!/bin/bash

thisFile=$_
if [ $BASH ]
then
  # may be a relative or absolute path
  thisFile=${BASH_SOURCE[0]}
fi

set_CODE_BASE()
{
  export ROS_MASTER_URI=http://localhost:11311
  export ROS_HOSTNAME=192.168.100.170
  export ROS_IP=192.168.100.170
  export ROSLAUNCH_SSH_UNKNOWN=1

  # use cd and pwd to get an absolute path
  configParentDir="$(cd "$(dirname "$thisFile")/.." && pwd)"

  # different cases for software/config or software/build/config
  case "$(basename $configParentDir)" in
    "software") export CODE_BASE=$(dirname $configParentDir);;
    "build") export CODE_BASE=$(dirname $(dirname $configParentDir));;
    *) echo "Warning: mpalm_affordances environment file is stored in unrecognized location: $thisFile";;
  esac
  export PATH=$PATH:$CODE_BASE/software/build/bin
}

setup_mpalm_affordances()
{
  export PATH=$PATH:$HOME/software/libbot/build/bin  # for lcm and libbot install
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CODE_BASE/software/build/lib:$CODE_BASE/software/build/lib64:$LD_LIBRARY_PATH
  export PYTHONPATH=$PYTHONPATH:$CODE_BASE/software/build/lib/python2.7/site-packages:$CODE_BASE/software/build/lib/python2.7/dist-packages
  # enable some warnings by default
  export CXXFLAGS="$CXXFLAGS -Wreturn-type -Wuninitialized"
  export CFLAGS="$CFLAGS -Wreturn-type -Wuninitialized"
  export PATH=$PATH:$HOME/software/ffmpeg-2.4.2-64bit-static # for ffmpeg software
}

set_ros()
{
  if [ -f $CODE_BASE/catkin_ws/devel/setup.bash ]; then
    echo "setting mpalm_affordances environment"
    source $CODE_BASE/catkin_ws/devel/setup.bash
  else
    source /opt/ros/*/setup.bash
  fi
  export ROS_PACKAGE_PATH=$HOME/mpalm_affordances/catkin_ws/:$ROS_PACKAGE_PATH
}

# some useful commands
alias cdmain='cd $CODE_BASE'
alias gitsub='git submodule update --init --recursive'
alias gitpull='git -C $CODE_BASE pull'
alias rebash='source ~/.bashrc'
alias open='gnome-open'

alias yolo='rosservice call /robot2_SetSpeed 1600 180'
alias faster='rosservice call /robot2_SetSpeed 200 50'
alias fast='rosservice call /robot2_SetSpeed 100 30'
alias slow='rosservice call /robot2_SetSpeed 50 15'

alias pman='bot-procman-sheriff -l $CODE_BASE/software/config/mpalm_affordances.pmd'

alias roslocal='export ROS_MASTER_URI=http://localhost:11311'

alias getjoint='rosservice call -- robot2_GetJoints'
alias getcart='rosservice call -- robot2_GetCartesian'
alias setjoint='rosservice call -- robot2_SetJoints'
alias setcart='rosservice call -- robot2_SetCartesian'
alias setspeed='rosservice call /robot2_SetSpeed'
alias zeroft='rosservice call zero'

alias lcmlocal='sudo ifconfig lo multicast; sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo'
alias sshraspi='ssh -X raspi@192.168.5.240'
alias sshraspi2='ssh -X raspi2@192.168.5.200'
alias catmake='cd $CODE_BASE/catkin_ws; catkin_make; cd -;'

function set_bash {
   PROMPT_COMMAND='history -a'
   history -a

   # sorting in old style
   LC_COLLATE="C"
   export LC_COLLATE

   ulimit -c unlimited
   export HISTTIMEFORMAT="%d/%m/%y %T "
}

function compress_video {
	ffmpeg -i $1.avi -b 10000000 $1.avi_compressed.avi
}


function kill_main {
   kill -9 $(ps aux | grep -v 'grep' | grep main.py | awk '{print $2;}')
}

set_CODE_BASE
setup_mpalm_affordances
set_ros
set_bash
