#ifndef __DVFS_PREDICTIVE_H
#define __DVFS_PREDICTIVE_H

#include "dvfspolicy.h"
#include "performance_counters.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sys/types.h>

/**
 * DVFSPredictive
 *
 * Proactive DVFS governor backed by the ML forecast model in
 * prediction/inference.py.  At each DVFS epoch it asks the model for the
 * predicted temperature change at horizon h (configurable via
 * scheduler/open/dvfs/predictive/horizon in base.cfg) and selects the highest
 * frequency that keeps the predicted temperature below the thermal limit.
 *
 * Communication: the constructor spawns prediction/dvfs_server.py as a
 * subprocess and exchanges single-line CSV messages over stdin/stdout pipes.
 * No changes to the build system are required.
 */
class DVFSPredictive : public DVFSPolicy {
public:
    DVFSPredictive(
        const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
        int minFrequency,
        int maxFrequency,
        int frequencyStepSize,
        int horizon,
        double thermalLimit);

    virtual ~DVFSPredictive();

    virtual std::vector<int> getFrequencies(
        const std::vector<int> &oldFrequencies,
        const std::vector<bool> &activeCores);

private:
    const PerformanceCounters *performanceCounters;
    unsigned int coreRows;
    unsigned int coreColumns;
    int minFrequency;
    int maxFrequency;
    int frequencyStepSize;
    int horizon;
    double thermalLimit;

    pid_t  childPid;
    FILE  *toServer;
    FILE  *fromServer;
    bool   serverAlive;

    void spawnServer();
    std::string serverScriptPath() const;
};

#endif // __DVFS_PREDICTIVE_H
