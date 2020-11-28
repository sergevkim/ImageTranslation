#!/bin/sh

mkdir data
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz
mv cityscapes.tar.gz data
cd data
tar -xvzf cityscapes.tar.gz
rm cityscapes.tar.gz
cd ..

