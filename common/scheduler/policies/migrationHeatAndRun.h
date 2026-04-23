/**
 * "Heat and Run" thread migration policy for Assignment 2 characterization.
 * Strategy: periodically find the hottest active core and migrate its thread
 * to the coolest active core. This models the classical thermal-aware
 * migration technique described in the course notes.
 */

#ifndef __MIGRATION_HEAT_AND_RUN_H
#define __MIGRATION_HEAT_AND_RUN_H

#include "migrationpolicy.h"
#include "performance_counters.h"
#include <vector>
#include <iostream>
#include <limits>

class MigrationHeatAndRun : public MigrationPolicy {
public:
    MigrationHeatAndRun(const PerformanceCounters *performanceCounters, int numberOfCores)
        : performanceCounters(performanceCounters), numberOfCores(numberOfCores) {}

    /**
     * Called every migration epoch.
     * Finds the hottest core and migrates its thread to the coolest core.
     * Returns an empty list, if no beneficial migration is found.
     */
    virtual std::vector<migration> migrate(SubsecondTime time,
                                           const std::vector<int> &taskIds,
                                           const std::vector<bool> &activeCores) override {
        std::vector<migration> migrations;

        int hottestCore = -1;
        int coolestCore = -1;
        double maxTemp = std::numeric_limits<double>::lowest();
        double minTemp = std::numeric_limits<double>::max();

        for (int core = 0; core < numberOfCores; core++) {
            double temp = performanceCounters->getTemperatureOfCore(core);

            if (activeCores[core] && taskIds[core] != -1) {
                // This core has an active thread: candidate to migrate FROM
                if (temp > maxTemp) {
                    maxTemp = temp;
                    hottestCore = core;
                }
            }

            if (!activeCores[core] || taskIds[core] == -1) {
                // This core is idle — candidate to migrate TO
                if (temp < minTemp) {
                    minTemp = temp;
                    coolestCore = core;
                }
            }
        }

        // Only migrate if there is a meaningful temperature difference (>= 2°C)
        // and both a hot source core and a cool destination core exist
        if (hottestCore != -1 && coolestCore != -1 && (maxTemp - minTemp) >= 2.0) {
            std::cout << "[MigrationHeatAndRun] Migrating thread from core "
                      << hottestCore << " (" << maxTemp << "°C) to core "
                      << coolestCore << " (" << minTemp << "°C)" << std::endl;
            migration m;
            m.fromCore = hottestCore;
            m.toCore   = coolestCore;
            m.swap     = false;
            migrations.push_back(m);
        }

        return migrations;
    }

private:
    const PerformanceCounters *performanceCounters;
    int numberOfCores;
};

#endif
