#include "dvfs_thermal.h"
#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace std;

DVFSThermal::DVFSThermal(const PerformanceCounters *performanceCounters,
                         int coreRows, int coreColumns, int minFrequency,
                         int maxFrequency)
    : performanceCounters(performanceCounters), coreRows(coreRows),
      coreColumns(coreColumns), minFrequency(minFrequency),
      maxFrequency(maxFrequency), fineStep(coreRows * coreColumns, F_STEP),
      lastDir(coreRows * coreColumns, 0) {
  cout << "[Scheduler][DVFS_THERMAL]: Initialized"
       << " f_min=" << minFrequency << " MHz"
       << " f_max=" << maxFrequency << " MHz"
       << " T_low=" << T_LOW << " C"
       << " T_goal=" << T_GOAL << " C"
       << " T_high=" << T_HIGH << " C"
       << " T_critical=" << T_CRITICAL << " C" << endl;
}

std::vector<int>
DVFSThermal::getFrequencies(const std::vector<int> &oldFrequencies,
                            const std::vector<bool> &activeCores) {
  std::vector<int> frequencies(oldFrequencies);

  for (unsigned int c = 0; c < coreRows * coreColumns; c++) {
    if (!activeCores.at(c)) {
      frequencies.at(c) = maxFrequency;
      continue;
    }

    double T = performanceCounters->getTemperatureOfCore(c);
    int oldFreq = oldFrequencies.at(c);

    if (T > T_CRITICAL) {
      // Emergency: jump to f_min immediately to maximise cooling rate.
      frequencies.at(c) = minFrequency;
      if (oldFreq != minFrequency)
        cout << "[Scheduler][DVFS_THERMAL]: Core " << setw(2) << c
             << " T=" << fixed << setprecision(1) << T << " C > " << T_CRITICAL
             << " C -> EMERGENCY f_min=" << minFrequency << " MHz" << endl;
      fineStep.at(c) = F_STEP;
      lastDir.at(c) = 0;

    } else if (T > T_HIGH) {
      // Coarse: step down toward the band.
      int newFreq = max(oldFreq - F_STEP, minFrequency);
      frequencies.at(c) = newFreq;
      if (newFreq != oldFreq)
        cout << "[Scheduler][DVFS_THERMAL]: Core " << setw(2) << c
             << " T=" << fixed << setprecision(1) << T << " C > " << T_HIGH
             << " C -> step down to " << newFreq << " MHz" << endl;
      fineStep.at(c) = F_STEP;
      lastDir.at(c) = 0;

    } else if (T < T_LOW) {
      // Coarse: step up toward the band.
      int newFreq = min(oldFreq + F_STEP, maxFrequency);
      frequencies.at(c) = newFreq;
      if (newFreq != oldFreq)
        cout << "[Scheduler][DVFS_THERMAL]: Core " << setw(2) << c
             << " T=" << fixed << setprecision(1) << T << " C < " << T_LOW
             << " C -> step up to " << newFreq << " MHz" << endl;
      fineStep.at(c) = F_STEP;
      lastDir.at(c) = 0;

    } else {
      // Inside: bisect toward T_GOAL
      if (fineStep.at(c) < MIN_FINE_STEP) {
        continue;
      }

      int dir = (T > T_GOAL) ? -1 : (T < T_GOAL) ? +1 : 0;
      if (dir == 0) {
        // at goal, hold!
        continue;
      }

      // Halve the step on direction reversal (overshoot detected).
      if (lastDir.at(c) != 0 && dir != lastDir.at(c))
        fineStep.at(c) = max(fineStep.at(c) / 2, MIN_FINE_STEP);

      int newFreq = oldFreq + dir * fineStep.at(c);
      newFreq = max(newFreq, minFrequency);
      newFreq = min(newFreq, maxFrequency);
      frequencies.at(c) = newFreq;
      lastDir.at(c) = dir;

      cout << "[Scheduler][DVFS_THERMAL]: Core " << setw(2) << c
           << " T=" << fixed << setprecision(1) << T << " C (goal=" << T_GOAL
           << " C) -> fine " << (dir > 0 ? "+" : "") << dir * fineStep.at(c)
           << " MHz to " << newFreq << " MHz (step=" << fineStep.at(c) << ")"
           << endl;
    }
  }

  return frequencies;
}
