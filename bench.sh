#!/bin/sh

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: ./bench.sh PATH_TO_PARAMS.cfg"
    exit 1
fi


binary_path=$(realpath $(dirname $0))/bin/Release
binary_name=main
config_path=../../config/
config_name=$(basename $1) #Single paramter of script: data set config file
log_file=log/3dns.log

# Stuff only works if we are in the directory of the binary, apparently
cd ${binary_path}

# Remove old log file so we can wait on the new one being created
rm -f ${log_file}

# Start rendering process in background
export CUDA_VISIBLE_DEVICES=1
./${binary_name} ${config_path}/${config_name}&
PID=$!

# Wait for log file to be created
until [ -f ${log_file} ]
do
    sleep 1
done

# Scan log file, kill rendering process when last non-full request batch is encountered (which means we are done with rendering)
tail -f ${log_file} | perl -ne 'if (/(\d+)\/(\d+)$/ && $2 == $1 + 1 && $2 != 50) { print; kill 9,'${PID}'; exit }'


#Final evaluation: time difference between first and last brick request
awk '
/^[0-9]{4}\/[0-9]{2}\/[0-9]{2} @ [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+/ && /Get brick/ {
    split($3, parts, /[:.]+/)
    hour = parts[1]
    min  = parts[2]
    sec  = parts[3]
    usec = parts[4]
    t = hour*3600 + min*60 + sec + usec/1000000
    if(first == "") first = t
    last = t
}
END {
    if (first) {
        diff = last - first
        printf("First time: %.6f\nLast time: %.6f\nDifference (seconds): %.6f\n", first, last, diff)
    }
}' ${log_file}
