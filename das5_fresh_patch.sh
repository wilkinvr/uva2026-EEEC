#!/bin/bash
# Restore the original scheduler_open.cc from the professor's copy
cp /var/scratch/spolstra/e3c/HotSniper/common/scheduler/scheduler_open.cc ~/projects/HotSniper/common/scheduler/scheduler_open.cc

# Add our migration policy include after the last existing policy include
sed -i 's|#include "policies/mapFirstUnused.h"|#include "policies/mapFirstUnused.h"\n#include "policies/migrationHeatAndRun.h"|' ~/projects/HotSniper/common/scheduler/scheduler_open.cc

# Register heatAndRun in initMigrationPolicy (replace the comment placeholder line)
sed -i 's|} \/\/else if (policyName ="XYZ") {|} else if (policyName == "heatAndRun") {\n\t\tmigrationPolicy = new MigrationHeatAndRun(performanceCounters, numberOfCores);\n\t} \/\/else if (policyName ="XYZ") {|' ~/projects/HotSniper/common/scheduler/scheduler_open.cc

echo "Done patching. Compiling..."
singularity exec /var/scratch/spolstra/e3c/ubuntu16.sif bash -c 'source ~/hotsniper.env; cd ~/projects/HotSniper; make 2>&1 | tail -20'
