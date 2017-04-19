# ml4music-workshop
Machine Learning for Music, Audio, and Sound Synthesis workshop @ the School of the Art Institute of Chicago, April 23rd, 2017.

## Overview

In this workshop we will learn about and use generative machine learning models to create new musical experiments. Generative models learn patterns from example data, and once trained, can be used to create new content similar to the data they were trained on. We will go through this full pipeline, from training, to generating, using three different models, each in a different music/audio domain:

- Song lyric generation (using a tensorflow implementation of [char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow/tree/5029173fc6ad527545082abcaf6c267061825484))
- Symbolic music generation (think MIDI, or sheet music) (using [midi-rnn](https://github.com/brannondorsey/midi-rnn/tree/cf6d1b6ceca80fe3fcf978b3a273fba6fceaed41))
- Raw audio generation (actual audio files, .wav and .mp3, etc...) (using a tensorflow implementation of [WaveNet](https://github.com/ibab/tensorflow-wavenet/tree/6e0b27d6c934d6beca0c4b3c2972c21bd86f5499))

We will begin by training a basic Recurrent Neural Network (RNN) on a corpus (large text dataset) of lyrics from thousands of pop songs. Next, we will use a similar model to generate new MIDI tracks in the style of ~100 MIDI files that we will train it with. Finally, we will dable in brand-new research into generating raw audio files at the sample level!

## Getting Started

We are going to be using a different tool for each of the three 1) text, 2) music, and 3) sound experiments. I've done my best to choose/build tools for each of these domains that are very similar to one another, to help flatten the learning curve. Once we develope an understanding for the first tool, char-rnn, you will find that the others are very similar.

I've created a Docker image that has everything needed to train our generative models pre-installed. If you would prefer not to use Docker, and instead install the dependencies yourself, see the manual installation instructions at the bottom of this README.

### Downloading this Repository

First, lets get you a copy of this here code. There should be a "Clone or Download" button on the right side of this page. Click that select "Download ZIP". Unzip :)

If you are comfortable using git, you can instead clown this repo with:

```
git clone https://github.com/brannondorsey/ml4music-workshop.git
```

### Installing with Docker

Next, download and install Docker CE (Community Edition) for your platform: [MacOS](https://download.docker.com/mac/stable/Docker.dmg) | [Windows](https://download.docker.com/win/stable/InstallDocker.msi) | [Linux](https://docs.docker.com/engine/installation/). Once unzipped, open the downloaded Docker app and follow the on-screen instructions. If you have an issue installing Docker on your computer, see the [Docker installation page](https://docs.docker.com/engine/installation/) for more info.

Once you've got docker installed, lets make sure it is working properly. Open your Terminal application (on MacOS, type "terminal" into your application search bar) and run the following command (copy + past the below text, then press ENTER).

```bash
docker --version
```

If you see a version number that means everything has been installed correctly!

Inside this repository, I've included a helpful little script called `start.sh`. This script will automagically log us into the workshop's Docker container. To run it, you must navigate your Terminal to this repository's folder (the one you downloaded earlier).

```bash
# the "cd" command stands for "change directory". we use it to 
# navigate around our terminal, just like you normally navigate
# a filebrowser to view the files on your computer
cd /path/to/this/folder

# once we are in this repository's folder we can execute the
# start.sh script like so
./start.sh
```

The first time you run it it needs to download our Docker Image from the internet (~2GB), so this may take a while. Once complete, you should see the message:

```
Starting TensorBoard 41 on port 7006.
(You can navigate to http://172.17.0.2:7006)
``` 

That means everything worked great! Press ENTER. You will now find yourself logged into our docker container.

