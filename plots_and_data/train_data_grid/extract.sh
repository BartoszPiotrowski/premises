#!/bin/bash
log=$1
csv="${log/log/csv}"
echo $csv
grep Percentage $log | cut -d' ' -f11 > $csv
