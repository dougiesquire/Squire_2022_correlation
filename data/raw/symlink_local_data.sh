#!/bin/bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast ${SCRIPT_DIR}/EC-Earth3_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppB-forecast ${SCRIPT_DIR}/EC-Earth3_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical ${SCRIPT_DIR}/EC-Earth3_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/ ${SCRIPT_DIR}/HadGEM3-GC31-MM_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast/ ${SCRIPT_DIR}/HadGEM3-GC31-MM_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl ${SCRIPT_DIR}/HadGEM3-GC31-MM_piControl

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast/ ${SCRIPT_DIR}/CanESM5_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppB-forecast/ ${SCRIPT_DIR}/CanESM5_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/CCCma/CanESM5/historical ${SCRIPT_DIR}/CanESM5_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/NCAR/CESM1-1-CAM5-CMIP5/dcppA-hindcast/ ${SCRIPT_DIR}/CESM1-1-CAM5-CMIP5_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/NCAR/CESM2/historical ${SCRIPT_DIR}/CESM2_historical
