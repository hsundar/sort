#!/bin/bash

make clean
make -j10
mkdir -p tmp
rm -r -f tmp/*

#Submit Job
qsub run.pbs

#Wait for job to finish and display output
while [ 1 ]
do

  FILE_OUT='tmp/main.out'
  FILE_ERR='tmp/main.err'

    if [ -f $FILE_OUT ]; then
      if [ -f $FILE_ERR ]; then
        cat tmp/main.out
        cat tmp/main.err
        break;
      fi
    fi
    #clear
    #ls -l tmp/
    sleep 1;

done


