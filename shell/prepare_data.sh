#!/bin/bash -l

#PBS -P xv83 
#PBS -q express
#PBS -l walltime=06:00:00
#PBS -l mem=192gb
#PBS -l ncpus=48
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -l storage=gdata/xv83+gdata/oi10+gdata/ua8
#PBS -j oe

conda activate squire_2022_correlation

python src/prepare_data.py

