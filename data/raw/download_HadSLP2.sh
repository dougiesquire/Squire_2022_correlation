#!/bin/bash

set -e

# f="https://www.metoffice.gov.uk/hadobs/hadslp2/data/HadSLP2r_lowvar_200501-201212.nc"
f="ftp://ftp.cdc.noaa.gov/Datasets.other/hadslp2/slp.mnmean.real.nc"

rm -f "slp.mnmean.real.nc"

wget "$f"
chmod 775 slp.mnmean.real.nc
