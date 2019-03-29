#!/usr/bin/env bash

mkdir -p res/data/archives
mkdir -p res/data/extract
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz -P res/data/archives
tar xzvf res/data/archives/MPIIGaze.tar.gz -C res/data/extract
rm -r res/data/archives
python3 gaze-preprocess.py
python3 gaze-cnn.py