#!/bin/bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# f="https://www.metoffice.gov.uk/hadobs/hadslp2/data/HadSLP2r_lowvar_200501-201212.nc"
f="ftp://ftp.cdc.noaa.gov/Datasets.other/hadslp2/slp.mnmean.real.nc"

rm -f ${SCRIPT_DIR}/"slp.mnmean.real.nc"
wget "$f" -P ${SCRIPT_DIR}
chmod 775 ${SCRIPT_DIR}/slp.mnmean.real.nc
