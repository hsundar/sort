#!/bin/bash

export EXEC=bin/main
export WORK_DIR=$(dirname ${PWD}/$0)/..
cd ${WORK_DIR}

mkdir -p tmp
make clean
make ${EXEC} -j8

export NODES=3;            # Number of compute nodes.
export CORES=12;           # Number of cores per node.
export MPI_PROC=32;        # Number of MPI processes.
export THREADS=1;          # Number of threads per MPI process.
export NUM_PTS=1000000;    # Number of point sources/samples.

FILE_OUT='tmp/main.out'
FILE_ERR='tmp/main.err'

#Submit Job
qsub -l nodes=${NODES}:ppn=$((${MPI_PROC}/${NODES})) -o ${FILE_OUT} -e ${FILE_ERR} ./scripts/run.pbs

#Wait for job to finish and display output
while [ 1 ]
do
    if [ -f $FILE_OUT ]; then
      if [ -f $FILE_ERR ]; then
        cat tmp/main.out
        cat tmp/main.err
        break;
      fi
    fi
    sleep 1;
done

