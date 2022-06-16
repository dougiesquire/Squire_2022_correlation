#!/bin/bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

echo ${SCRIPT_DIR}

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast ${SCRIPT_DIR}/EC-Earth3_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppB-forecast ${SCRIPT_DIR}/EC-Earth3_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical ${SCRIPT_DIR}/EC-Earth3_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/ ${SCRIPT_DIR}/HadGEM3-GC31-MM_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast/ ${SCRIPT_DIR}/HadGEM3-GC31-MM_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl ${SCRIPT_DIR}/HadGEM3-GC31-MM_piControl
