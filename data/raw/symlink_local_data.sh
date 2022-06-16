#!/bin/bash

set -e

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast EC-Earth3_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppB-forecast EC-Earth3_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical EC-Earth3_historical

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/ HadGEM3-GC31-MM_dcppA-hindcast

ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast/ HadGEM3-GC31-MM_dcppB-forecast

ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl HadGEM3-GC31-MM_piControl
