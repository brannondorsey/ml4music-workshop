FROM nvidia/cuda:8.0-devel-ubuntu16.04

# create a non-root user named "docker"
# we will use this when the user is logged in
RUN useradd -ms /bin/bash docker
USER root
ENV DOCKER_HOME /home/docker

# install apt dependencies and clone the repo
RUN apt-get update
RUN apt-get install -y git python python-pip
WORKDIR $DOCKER_HOME
RUN git clone https://github.com/brannondorsey/ml4music-workshop.git
WORKDIR $DOCKER_HOME/ml4music-workshop
RUN git submodule init
RUN git submodule update

# remove unnecessary files
RUN rm setup_wavenet_venv.sh
RUN rm Dockerfile
RUN rm -rf notebooks

# instal pip and pip dependencies
RUN pip install --upgrade pip
RUN pip install virtualenv
RUN pip install tensorflow==1.0.1

WORKDIR $DOCKER_HOME/ml4music-workshop/midi-rnn
RUN pip install -r requirements.txt

# install and setup a virtual env for tensorflow-wavenet
# as it requires a conflicting version of tensorflow from char-rnn-tensorflow
WORKDIR $DOCKER_HOME/ml4music-workshop/tensorflow-wavenet
RUN virtualenv venv
COPY setup_wavenet_venv.sh $DOCKER_HOME/ml4music-workshop/tensorflow-wavenet/setup_wavenet_venv.sh
RUN $DOCKER_HOME/ml4music-workshop/tensorflow-wavenet/setup_wavenet_venv.sh
RUN rm setup_wavenet_venv.sh

WORKDIR $DOCKER_HOME/ml4music-workshop

RUN chown -R docker:docker ./
USER docker

# expose tensorboard ports
# 7006 - char-rnn
# 7007 - midi-rnn
# 7008 - wavene7

EXPOSE 7006
EXPOSE 7007
EXPOSE 7008

CMD tensorboard --port 7006 --logdir $DOCKER_HOME/ml4music-workshop/char-rnn-tensorflow/logs &> /dev/null && \
    tensorboard --port 7007 --logdir $DOCKER_HOME/ml4music-workshop/midi-rnn/experiments &> /dev/null && \
    tensorboard --port 7008 --logdir $DOCKER_HOME/ml4music-workshop/tensorflow-wavenet/logdir &> /dev/null && bash
