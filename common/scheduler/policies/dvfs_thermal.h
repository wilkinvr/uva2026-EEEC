#ifndef __DVFS_THERMAL_H
#define __DVFS_THERMAL_H

#include "dvfspolicy.h"
#include "performance_counters.h"
#include <vector>

class DVFSThermal : public DVFSPolicy {
public:
  DVFSThermal(const PerformanceCounters *performanceCounters, int coreRows,
              int coreColumns, int minFrequency, int maxFrequency);
  virtual std::vector<int>
  getFrequencies(const std::vector<int> &oldFrequencies,
                 const std::vector<bool> &activeCores);

private:
  const PerformanceCounters *performanceCounters;
  unsigned int coreRows;
  unsigned int coreColumns;
  int minFrequency;
  int maxFrequency;

  std::vector<int> fineStep;
  std::vector<int> lastDir;

  static constexpr double T_LOW = 65.0;      // below this: step up (Celsius)
  static constexpr double T_HIGH = 85.0;     // above this: step down
  static constexpr double T_CRITICAL = 95.0; // above this: emergency drop
  static constexpr double T_GOAL =
      80.0; // target inside the band, start fine-tuing with steps towards this
  static constexpr int F_STEP = 250;       // start step size (MHz)
  static constexpr int MIN_FINE_STEP = 50; // stop fine-tuning below this
};

#endif
