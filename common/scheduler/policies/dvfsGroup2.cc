#include "dvfsGroup2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

using namespace std;

static string sniperRootGroup2() {
    string p(__FILE__);
    for (int i = 0; i < 4; ++i) {
        size_t slash = p.rfind('/');
        if (slash == string::npos) break;
        p = p.substr(0, slash);
    }
    return p;
}


DVFSGroup2::DVFSGroup2(
        const PerformanceCounters *performanceCounters,
        int coreRows,
        int coreColumns,
        int minFrequency,
        int maxFrequency,
        int frequencyStepSize,
        int horizon,
        double thermalLimit)
    : performanceCounters(performanceCounters),
      coreRows(coreRows),
      coreColumns(coreColumns),
      minFrequency(minFrequency),
      maxFrequency(maxFrequency),
      frequencyStepSize(frequencyStepSize),
      horizon(horizon),
      thermalLimit(thermalLimit),
      childPid(-1),
      toServer(nullptr),
      fromServer(nullptr),
      serverAlive(false),
      fineStep(coreRows * coreColumns, F_STEP),
      lastDir(coreRows * coreColumns, 0),
      prevDeltaTemp(coreRows * coreColumns, 0.0)
{
    cout << "[Scheduler][DVFSGroup2]: Initialized"
         << " f_min=" << minFrequency << " MHz"
         << " f_max=" << maxFrequency << " MHz"
         << " step=" << frequencyStepSize << " MHz"
         << " horizon=" << horizon
         << " thermal_limit=" << thermalLimit << " C"
         << " T_low=" << T_LOW << " C"
         << " T_goal=" << T_GOAL << " C"
         << " T_high=" << T_HIGH << " C"
         << " T_critical=" << T_CRITICAL << " C"
         << endl;

    spawnServer();
}

DVFSGroup2::~DVFSGroup2() {
    if (toServer)   { fclose(toServer);   toServer   = nullptr; }
    if (fromServer) { fclose(fromServer); fromServer = nullptr; }
    if (childPid > 0) {
        waitpid(childPid, nullptr, 0);
        childPid = -1;
    }
}


string DVFSGroup2::serverScriptPath() const {
    return sniperRootGroup2() + "/prediction/dvfs_server_group2.py";
}

void DVFSGroup2::spawnServer() {
    int pipe_to[2];
    int pipe_from[2];

    if (pipe(pipe_to) != 0 || pipe(pipe_from) != 0) {
        cerr << "[DVFSGroup2] pipe() failed: " << strerror(errno) << endl;
        return;
    }

    string script = serverScriptPath();

    childPid = fork();
    if (childPid < 0) {
        cerr << "[DVFSGroup2] fork() failed: " << strerror(errno) << endl;
        return;
    }

    if (childPid == 0) {
        close(pipe_to[1]);
        close(pipe_from[0]);

        dup2(pipe_to[0],   STDIN_FILENO);
        dup2(pipe_from[1], STDOUT_FILENO);

        close(pipe_to[0]);
        close(pipe_from[1]);

        const char *args[] = { "python3.10", script.c_str(), nullptr };
        execvp("python3.10", const_cast<char **>(args));

        cerr << "[DVFSGroup2] execvp failed: " << strerror(errno) << endl;
        _exit(1);
    }

    close(pipe_to[0]);
    close(pipe_from[1]);

    toServer   = fdopen(pipe_to[1],   "w");
    fromServer = fdopen(pipe_from[0], "r");

    if (!toServer || !fromServer) {
        cerr << "[DVFSGroup2] fdopen() failed" << endl;
        return;
    }

    serverAlive = true;
    cout << "[DVFSGroup2] Server started (pid=" << childPid
         << "): " << script << endl;
}


int DVFSGroup2::bandLogicCore(unsigned int c, double effectiveT, int oldFreq, bool isActive) {
    if (!isActive) {
        fineStep[c] = F_STEP;
        lastDir[c]  = 0;
        return maxFrequency;
    }

    if (effectiveT > T_CRITICAL) {
        fineStep[c] = F_STEP;
        lastDir[c]  = 0;
        return minFrequency;
    }

    if (effectiveT > T_HIGH) {
        fineStep[c] = F_STEP;
        lastDir[c]  = 0;
        return max(oldFreq - F_STEP, minFrequency);
    }

    if (effectiveT < T_LOW) {
        fineStep[c] = F_STEP;
        lastDir[c]  = 0;
        return min(oldFreq + F_STEP, maxFrequency);
    }

    // Inside band: bisect toward T_GOAL.
    bool suppress = (fabs(prevDeltaTemp[c]) < UNCERTAINTY_THRESHOLD &&
                     fabs(effectiveT - T_GOAL) < FINE_SUPPRESS_BAND);
    if (suppress || fineStep[c] < MIN_FINE_STEP) {
        return oldFreq;
    }

    int dir = (effectiveT > T_GOAL) ? -1 : +1;

    if (lastDir[c] != 0 && dir != lastDir[c])
        fineStep[c] = max(fineStep[c] / 2, MIN_FINE_STEP);

    lastDir[c] = dir;

    int newFreq = oldFreq + dir * fineStep[c];
    newFreq = max(newFreq, minFrequency);
    newFreq = min(newFreq, maxFrequency);
    return newFreq;
}

vector<int> DVFSGroup2::getFrequencies(
        const vector<int> &oldFrequencies,
        const vector<bool> &activeCores)
{
    unsigned int nCores = coreRows * coreColumns;

    vector<double> measuredT(nCores);
    for (unsigned int c = 0; c < nCores; ++c)
        measuredT[c] = performanceCounters->getTemperatureOfCore(c);

    vector<int>    candidateFreq(nCores);
    vector<double> effectiveT_band(nCores);   // saved for logging
    for (unsigned int c = 0; c < nCores; ++c) {
        effectiveT_band[c] = measuredT[c] + prevDeltaTemp[c];
        candidateFreq[c]   = bandLogicCore(c, effectiveT_band[c], oldFrequencies[c], activeCores[c]);
    }

    vector<double> newDeltaTemp(nCores, 0.0);

    if (serverAlive) {
        double peakTemp     = *max_element(measuredT.begin(), measuredT.end());
        double minTemp      = *min_element(measuredT.begin(), measuredT.end());
        double tempGradient = peakTemp - minTemp;

        ostringstream req;
        req << fixed
            << coreRows << ","
            << coreColumns << ","
            << thermalLimit << ","
            << horizon << ","
            << minFrequency << ","
            << maxFrequency << ","
            << frequencyStepSize;

        for (unsigned int c = 0; c < nCores; ++c) {
            double T    = measuredT[c];
            double P    = performanceCounters->getPowerOfCore(c);
            double fGHz = candidateFreq[c] / 1000.0;   // candidate, not old freq
            double util = performanceCounters->getUtilizationOfCore(c);
            double cpiTotal  = performanceCounters->getCPIOfCore(c);
            double cpiBase   = performanceCounters->getCPIStackPartOfCore(c, "base");
            double cpiDram   = performanceCounters->getCPIStackPartOfCore(c, "mem-dram");
            double cpiBranch = performanceCounters->getCPIStackPartOfCore(c, "branch");
            double cpiIfetch = performanceCounters->getCPIStackPartOfCore(c, "ifetch");
            double cpiL1d    = performanceCounters->getCPIStackPartOfCore(c, "mem-l1d");
            double ips       = performanceCounters->getIPSOfCore(c);
            double active    = activeCores[c] ? 1.0 : 0.0;

            req << "," << T
                << "," << P
                << "," << fGHz
                << "," << util
                << "," << cpiTotal
                << "," << cpiBase
                << "," << cpiDram
                << "," << cpiBranch
                << "," << cpiIfetch
                << "," << cpiL1d
                << "," << ips
                << "," << active
                << "," << peakTemp
                << "," << tempGradient;
        }
        req << "\n";
        string reqStr = req.str();

        if (fputs(reqStr.c_str(), toServer) == EOF || fflush(toServer) != 0) {
            cerr << "[DVFSGroup2] write to server failed" << endl;
            serverAlive = false;
        } else {
            char buf[1024] = {};
            if (!fgets(buf, sizeof(buf), fromServer)) {
                cerr << "[DVFSGroup2] read from server failed" << endl;
                serverAlive = false;
            } else {
                istringstream resp(buf);
                for (unsigned int c = 0; c < nCores; ++c) {
                    double dt;
                    if (!(resp >> dt)) break;
                    newDeltaTemp[c] = dt;
                }
            }
        }
    }

    // Persist for next epoch before any overrides
    prevDeltaTemp = newDeltaTemp;

    vector<int> finalFreq = candidateFreq;

    for (unsigned int c = 0; c < nCores; ++c) {
        if (!activeCores[c]) continue;

        double effT_verify = measuredT[c] + newDeltaTemp[c];

        if (effT_verify > T_CRITICAL) {
            finalFreq[c]   = minFrequency;
            fineStep[c]    = F_STEP;
            lastDir[c]     = 0;
        } else if (effT_verify > T_HIGH && candidateFreq[c] > oldFrequencies[c]) {
            // We were stepping up but it'll be too hot
            finalFreq[c]   = oldFrequencies[c];
            fineStep[c]    = F_STEP;
            lastDir[c]     = 0;
        }
    }

    for (unsigned int c = 0; c < nCores; ++c) {
        double effT_verify = measuredT[c] + newDeltaTemp[c];
        bool   overridden  = (finalFreq[c] != candidateFreq[c]);

        cout << "[Scheduler][DVFSGroup2]: Core " << setw(2) << c
             << " T="     << fixed << setprecision(1) << measuredT[c]       << " C"
             << " effT="  << setprecision(1) << effectiveT_band[c] << " C"
             << " vfyT="  << setprecision(1) << effT_verify         << " C"
             << "  cur="  << setw(4) << oldFrequencies[c]  << " MHz"
             << "  cand=" << setw(4) << candidateFreq[c]   << " MHz"
             << "  final="<< setw(4) << finalFreq[c]       << " MHz"
             << (overridden ? " [override]" : "")
             << (!activeCores[c] ? " [idle]" : "")
             << endl;
    }

    return finalFreq;
}
