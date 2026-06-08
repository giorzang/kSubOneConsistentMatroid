#!/bin/bash

wget http://snap.stanford.edu/data/ca-AstroPh.txt.gz
gunzip ca-AstroPh.txt.gz
../../../preproc ca-AstroPh.txt astro.bin

wget http://snap.stanford.edu/data/web-Google.txt.gz
gunzip web-Google.txt.gz
../../../preproc web-Google.txt google.bin

wget http://snap.stanford.edu/data/ca-GrQc.txt.gz
gunzip ca-GrQc.txt.gz
../../../preproc ca-GrQc.txt grqc.bin