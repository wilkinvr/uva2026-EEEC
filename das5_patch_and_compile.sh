#!/bin/bash
# Remove includes that don't exist in the DAS-5 version
sed -i '/#include "policies\/dvfsTSP.h"/d' ~/projects/HotSniper/common/scheduler/scheduler_open.cc
sed -i '/#include "policies\/pcgov.h"/d' ~/projects/HotSniper/common/scheduler/scheduler_open.cc
# Also remove PCGov instantiation if present
sed -i '/PCGov/d' ~/projects/HotSniper/common/scheduler/scheduler_open.cc
echo "Patched. Now compiling..."
singularity exec /var/scratch/spolstra/e3c/ubuntu16.sif bash -c 'source ~/hotsniper.env; cd ~/projects/HotSniper; make 2>&1 | tail -20'
