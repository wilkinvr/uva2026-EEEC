#!/bin/bash
# Create the expected directory structure in scratch
mkdir -p /var/scratch/eeec2625/e3c

# Copy our compiled HotSniper to scratch (where the job script expects it)
echo "Copying HotSniper to scratch (this may take a minute)..."
cp -r ~/projects/HotSniper /var/scratch/eeec2625/e3c/HotSniper

# Also copy our run.py to scratch location
cp ~/projects/HotSniper/simulationcontrol/run.py /var/scratch/eeec2625/e3c/HotSniper/simulationcontrol/run.py
cp ~/projects/HotSniper/config/base.cfg /var/scratch/eeec2625/e3c/HotSniper/config/base.cfg

echo "Done. Submitting job..."
sbatch ~/simulation.job
echo "Check status with: squeue"
