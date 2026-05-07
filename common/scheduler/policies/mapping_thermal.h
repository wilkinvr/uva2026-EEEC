/**
 * Reactive thermal migration policy.
 * At each migration epoch, the hottest and coldest active cores are found.
 * If the hottest core exceeds T_MIGRATE and the spatial gradient exceeds
 * DELTA_T_MIN, the tasks on those two cores are swapped to redistribute load.
 */

#ifndef __MAPPING_THERMAL_H
#define __MAPPING_THERMAL_H

#include "migrationpolicy.h"
#include "performance_counters.h"

class MappingThermal : public MigrationPolicy {
public:
  MappingThermal(const PerformanceCounters *performanceCounters,
                 int numberOfCores);
  virtual std::vector<migration> migrate(SubsecondTime time,
                                         const std::vector<int> &taskIds,
                                         const std::vector<bool> &activeCores);

private:
  const PerformanceCounters *performanceCounters;
  int numberOfCores;

  static constexpr double T_MIGRATE = 68.0;
  static constexpr double DELTA_T_MIN = 8.0;
};

#endif
