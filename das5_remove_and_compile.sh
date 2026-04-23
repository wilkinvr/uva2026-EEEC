#!/bin/bash
# Remove the pcgov and dvfsTSP files we uploaded - DAS-5 doesn't have their dependencies
rm -f ~/projects/HotSniper/common/scheduler/policies/pcgov.cc
rm -f ~/projects/HotSniper/common/scheduler/policies/pcgov.h
rm -f ~/projects/HotSniper/common/scheduler/policies/dvfsTSP.cc
rm -f ~/projects/HotSniper/common/scheduler/policies/dvfsTSP.h
echo "Removed. Compiling..."
singularity exec /var/scratch/spolstra/e3c/ubuntu16.sif bash -c 'source ~/hotsniper.env; cd ~/projects/HotSniper; make 2>&1 | tail -20'
