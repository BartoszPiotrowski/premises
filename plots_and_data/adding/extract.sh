#!/bin/bash
log=$1
csv="${log/log/csv}"
echo $csv
grep "all t" $log | cut -d' ' -f9 > $csv
