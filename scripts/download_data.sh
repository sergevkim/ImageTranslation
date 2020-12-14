#!/bin/sh

mkdir data
cd data
#wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
#tar -xvzf cityscapes.tar.gz
tar -xvzf facades.tar.gz
#rm cityscapes.tar.gz
rm facades.tar.gz
cd ..

