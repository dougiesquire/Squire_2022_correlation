#!/bin/bash -l

#PBS -P xv83 
##PBS -q hugemem
#PBS -q express
#PBS -l walltime=03:00:00
##PBS -l mem=1TB
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -l storage=gdata/xv83+gdata/oi10+gdata/ua8
#PBS -j oe

conda activate squire_2022_correlation

python src/prepare_data.py

