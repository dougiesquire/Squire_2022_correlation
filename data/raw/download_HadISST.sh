#!/bin/bash

set -e

f="https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz"

rm -f "HadISST_sst.nc.gz"
 
wget "$f"
zip_file=`basename $f`
gzip -d $zip_file
chmod 775 HadISST_sst.nc
