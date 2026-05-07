#include "mapping_thermal.h"
#include <cfloat>
#include <iomanip>
#include <iostream>

using namespace std;

MappingThermal::MappingThermal(const PerformanceCounters *performanceCounters,
                               int numberOfCores)
    : performanceCounters(performanceCounters), numberOfCores(numberOfCores) {
  cout << "[Scheduler][MIGRATION_THERMAL]: T_migrate=" << T_MIGRATE
       << " C, delta_T_min=" << DELTA_T_MIN << " C" << endl;
}

std::vector<migration>
MappingThermal::migrate(SubsecondTime time, const std::vector<int> &taskIds,
                        const std::vector<bool> &activeCores) {
  std::vector<migration> migrations;

  int hotCore = -1, coldCore = -1;
  double T_hot = -1.0, T_cold = DBL_MAX;

  for (int c = 0; c < numberOfCores; c++) {
    // Check if core has task
    if (taskIds.at(c) == -1)
      continue;

    double T = performanceCounters->getTemperatureOfCore(c);
    if (T > T_hot) {
      T_hot = T;
      hotCore = c;
    }
    if (T < T_cold) {
      T_cold = T;
      coldCore = c;
    }
  }

  if (hotCore == -1 || coldCore == -1 || hotCore == coldCore) {
    // not enough cores found
    return migrations;
  }

  if (T_hot > T_MIGRATE && (T_hot - T_cold) > DELTA_T_MIN) {
    cout << "[Scheduler][MIGRATION_THERMAL]: Swapping core " << hotCore
         << " (T=" << fixed << setprecision(1) << T_hot
         << " C, task=" << taskIds.at(hotCore) << ")"
         << " with core " << coldCore << " (T=" << fixed << setprecision(1)
         << T_cold << " C, task=" << taskIds.at(coldCore) << ")" << endl;

    migrations.push_back({(unsigned int)hotCore, (unsigned int)coldCore, true});
  }

  return migrations;
}
