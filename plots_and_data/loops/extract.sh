#!/bin/bash
log=$1
csv="${log/log/csv}"
echo $csv
grep Transforming $log | cut -d' ' -f6 > $csv
