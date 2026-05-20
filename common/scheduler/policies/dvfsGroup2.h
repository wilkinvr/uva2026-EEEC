#ifndef __DVFS_GROUP2_H
#define __DVFS_GROUP2_H

#include "dvfspolicy.h"
#include "performance_counters.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sys/types.h>

class DVFSGroup2 : public DVFSPolicy {
public:
    DVFSGroup2(
        const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
        int minFrequency,
        int maxFrequency,
        int frequencyStepSize,
        int horizon,
        double thermalLimit);

    virtual ~DVFSGroup2();

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

    // prediction server
    pid_t  childPid;
    FILE  *toServer;
    FILE  *fromServer;
    bool   serverAlive;

    void spawnServer();
    std::string serverScriptPath() const;

    std::vector<int>    fineStep;
    std::vector<int>    lastDir;

    std::vector<double> prevDeltaTemp;


    int bandLogicCore(unsigned int c, double effectiveT, int oldFreq, bool isActive);

    static constexpr double T_LOW      = 65.0;
    static constexpr double T_GOAL     = 80.0;
    static constexpr double T_HIGH     = 85.0;
    static constexpr double T_CRITICAL = 95.0;
    static constexpr int    F_STEP        = 250;
    static constexpr int    MIN_FINE_STEP = 50;

    static constexpr double UNCERTAINTY_THRESHOLD = 0.5;
    static constexpr double FINE_SUPPRESS_BAND    = 2.0;
};

#endif // __DVFS_GROUP2_H
