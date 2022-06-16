#!/bin/bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

f="https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz"

rm -f ${SCRIPT_DIR}/"HadISST_sst.nc.gz"
 
wget "$f" -P ${SCRIPT_DIR}
zip_file=`basename $f`
gzip -f -d ${SCRIPT_DIR}/$zip_file
chmod 775 ${SCRIPT_DIR}/HadISST_sst.nc
