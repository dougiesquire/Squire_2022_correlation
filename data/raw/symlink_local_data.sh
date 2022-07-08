#!/bin/bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast ${SCRIPT_DIR}/EC-Earth3_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppB-forecast ${SCRIPT_DIR}/EC-Earth3_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical ${SCRIPT_DIR}/EC-Earth3_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast ${SCRIPT_DIR}/HadGEM3-GC31-MM_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast ${SCRIPT_DIR}/HadGEM3-GC31-MM_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical ${SCRIPT_DIR}/HadGEM3-GC31-MM_historical

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl ${SCRIPT_DIR}/HadGEM3-GC31-MM_piControl

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast ${SCRIPT_DIR}/CanESM5_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppB-forecast ${SCRIPT_DIR}/CanESM5_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/CCCma/CanESM5/historical ${SCRIPT_DIR}/CanESM5_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/NCAR/CESM1-1-CAM5-CMIP5/dcppA-hindcast ${SCRIPT_DIR}/CESM1-1-CAM5-CMIP5_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/NCAR/CESM2/historical ${SCRIPT_DIR}/CESM2_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MIROC/MIROC6/dcppA-hindcast ${SCRIPT_DIR}/MIROC6_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MIROC/MIROC6/historical ${SCRIPT_DIR}/MIROC6_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MPI-M/MPI-ESM1-2-HR/dcppA-hindcast ${SCRIPT_DIR}/MPI-ESM1-2-HR_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical ${SCRIPT_DIR}/MPI-ESM1-2-HR_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppA-hindcast ${SCRIPT_DIR}/IPSL-CM6A-LR_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical ${SCRIPT_DIR}/IPSL-CM6A-LR_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/NCC/NorCPM1/dcppA-hindcast ${SCRIPT_DIR}/NorCPM1_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/NCC/NorCPM1/historical ${SCRIPT_DIR}/NorCPM1_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CMCC/CMCC-CM2-SR5/dcppA-hindcast ${SCRIPT_DIR}/CMCC-CM2-SR5_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CMCC/CMCC-CM2-SR5/dcppB-forecast ${SCRIPT_DIR}/CMCC-CM2-SR5_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/CMCC/CMCC-CM2-SR5/historical ${SCRIPT_DIR}/CMCC-CM2-SR5_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MRI/MRI-ESM2-0/dcppA-hindcast ${SCRIPT_DIR}/MRI-ESM2-0_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MRI/MRI-ESM2-0/historical ${SCRIPT_DIR}/MRI-ESM2-0_historical
