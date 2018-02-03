#!/bin/bash
log=$1
csv="${log/log/all.csv}"
echo $csv
grep "all p" $log | cut -d' ' -f7 > $csv
