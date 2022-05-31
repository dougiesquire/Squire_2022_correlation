#!/bin/bash

set -e

f="https://climatedataguide.ucar.edu/sites/default/files/nao_station_annual.txt"

rm -f "nao_station_annual.txt"

wget "$f"
chmod 775 nao_station_annual.txt
