FROM nvidia/cuda:8.0-devel-ubuntu16.04

# install apt dependencies and clone the repo
RUN apt-get update
RUN apt-get install -y git python python-pip
WORKDIR /root
RUN git clone https://github.com/brannondorsey/ml4music-workshop.git
WORKDIR /root/ml4music-workshop
RUN git submodule init
RUN git submodule update

# instal pip and pip dependencies
RUN pip install --upgrade pip
RUN pip install virtualenv
RUN pip install tensorflow==1.0.1

WORKDIR /root/ml4music-workshop/midi-rnn
RUN pip install -r requirements.txt

# install and setup a virtual env for tensorflow-wavenet
# as it requires a conflicting version of tensorflow from char-rnn-tensorflow
WORKDIR /root/ml4music-workshop/tensorflow-wavenet
RUN virtualenv venv
COPY setup_wavenet_venv.sh /root/ml4music-workshop/tensorflow-wavenet/setup_wavenet_venv.sh
RUN /root/ml4music-workshop/tensorflow-wavenet/setup_wavenet_venv.sh
RUN rm setup_wavenet_venv.sh

WORKDIR /root/ml4music-workshop

# expose tensorboard ports
# 7006 - char-rnn
# 7007 - midi-rnn
# 7008 - wavene7

EXPOSE 7006
EXPOSE 7007 
EXPOSE 7008

CMD tensorboard --port 7006 --logdir /root/ml4music-workshop/char-rnn-tensorflow/logs &> /dev/null && \
    tensorboard --port 7007 --logdir /root/ml4music-workshop/midi-rnn/experiments &> /dev/null && \
    tensorboard --port 7008 --logdir /root/ml4music-workshop/tensorflow-wavenet/logdir &> /dev/null && bash
