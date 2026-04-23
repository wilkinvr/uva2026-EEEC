#!/bin/bash
singularity exec /var/scratch/spolstra/e3c/ubuntu16.sif bash -c 'source ~/hotsniper.env; cd ~/projects/HotSniper; make 2>&1 | tail -30'
