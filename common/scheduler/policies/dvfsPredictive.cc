#include "dvfsPredictive.h"

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Compute the absolute path to dvfs_server.py relative to this source file.
// __FILE__ is the full path at compile time:
//   /hotsniper/common/scheduler/policies/dvfsPredictive.cc
// Going up four path components reaches /hotsniper.
static string sniperRoot() {
    string p(__FILE__);
    for (int i = 0; i < 4; ++i) {
        size_t slash = p.rfind('/');
        if (slash == string::npos) break;
        p = p.substr(0, slash);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

DVFSPredictive::DVFSPredictive(
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
      serverAlive(false)
{
    cout << "[Scheduler][DVFSPredictive]: Initialized"
         << " f_min=" << minFrequency << " MHz"
         << " f_max=" << maxFrequency << " MHz"
         << " step=" << frequencyStepSize << " MHz"
         << " horizon=" << horizon
         << " thermal_limit=" << thermalLimit << " C"
         << endl;

    spawnServer();
}

DVFSPredictive::~DVFSPredictive() {
    if (toServer)   { fclose(toServer);   toServer   = nullptr; }
    if (fromServer) { fclose(fromServer); fromServer = nullptr; }
    if (childPid > 0) {
        waitpid(childPid, nullptr, 0);
        childPid = -1;
    }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

string DVFSPredictive::serverScriptPath() const {
    return sniperRoot() + "/prediction/dvfs_server.py";
}

void DVFSPredictive::spawnServer() {
    int pipe_to[2];    // C++ → Python
    int pipe_from[2];  // Python → C++

    if (pipe(pipe_to) != 0 || pipe(pipe_from) != 0) {
        cerr << "[DVFSPredictive] pipe() failed: " << strerror(errno) << endl;
        return;
    }

    string script = serverScriptPath();

    childPid = fork();
    if (childPid < 0) {
        cerr << "[DVFSPredictive] fork() failed: " << strerror(errno) << endl;
        return;
    }

    if (childPid == 0) {
        // ---- Child process (Python server) ----
        close(pipe_to[1]);    // close write end of to-pipe
        close(pipe_from[0]);  // close read end of from-pipe

        dup2(pipe_to[0],   STDIN_FILENO);
        dup2(pipe_from[1], STDOUT_FILENO);

        close(pipe_to[0]);
        close(pipe_from[1]);

        const char *args[] = { "python3.10", script.c_str(), nullptr };
        execvp("python3.10", const_cast<char **>(args));

        // execvp only returns on failure
        cerr << "[DVFSPredictive] execvp failed: " << strerror(errno) << endl;
        _exit(1);
    }

    // ---- Parent process (C++ scheduler) ----
    close(pipe_to[0]);    // close read end of to-pipe
    close(pipe_from[1]);  // close write end of from-pipe

    toServer   = fdopen(pipe_to[1],   "w");
    fromServer = fdopen(pipe_from[0], "r");

    if (!toServer || !fromServer) {
        cerr << "[DVFSPredictive] fdopen() failed" << endl;
        return;
    }

    serverAlive = true;
    cout << "[DVFSPredictive] Server started (pid=" << childPid
         << "): " << script << endl;
}

vector<int> DVFSPredictive::getFrequencies(
        const vector<int> &oldFrequencies,
        const vector<bool> &activeCores)
{
    unsigned int nCores = coreRows * coreColumns;

    // Fallback to current frequencies when server is unavailable
    if (!serverAlive) {
        return vector<int>(oldFrequencies);
    }

    // Gather system-wide features once
    double peakTemp    = performanceCounters->getPeakTemperature();
    double tempGradient = 0.0;
    {
        double minT = peakTemp;
        for (unsigned int c = 0; c < nCores; ++c)
            minT = min(minT, performanceCounters->getTemperatureOfCore(c));
        tempGradient = peakTemp - minT;
    }

    // Build CSV request
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
        double T    = performanceCounters->getTemperatureOfCore(c);
        double P    = performanceCounters->getPowerOfCore(c);
        double fMHz = oldFrequencies.at(c);
        double fGHz = fMHz / 1000.0;
        double util = performanceCounters->getUtilizationOfCore(c);
        double cpiTotal  = performanceCounters->getCPIOfCore(c);
        double cpiBase   = performanceCounters->getCPIStackPartOfCore(c, "base");
        double cpiDram   = performanceCounters->getCPIStackPartOfCore(c, "mem-dram");
        double cpiBranch = performanceCounters->getCPIStackPartOfCore(c, "branch");
        double cpiIfetch = performanceCounters->getCPIStackPartOfCore(c, "ifetch");
        double cpiL1d    = performanceCounters->getCPIStackPartOfCore(c, "mem-l1d");
        double ips       = performanceCounters->getIPSOfCore(c);
        double active    = activeCores.at(c) ? 1.0 : 0.0;

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
        cerr << "[DVFSPredictive] write to server failed" << endl;
        serverAlive = false;
        return vector<int>(oldFrequencies);
    }

    // Read response: space-separated frequencies in MHz, one per core
    char buf[1024] = {};
    if (!fgets(buf, sizeof(buf), fromServer)) {
        cerr << "[DVFSPredictive] read from server failed" << endl;
        serverAlive = false;
        return vector<int>(oldFrequencies);
    }

    vector<int> frequencies(nCores, maxFrequency);
    istringstream resp(buf);
    for (unsigned int c = 0; c < nCores; ++c) {
        int f;
        if (!(resp >> f)) break;
        // Clamp to valid range
        f = max(f, minFrequency);
        f = min(f, maxFrequency);
        frequencies.at(c) = f;
    }

    // Log every core every epoch
    for (unsigned int c = 0; c < nCores; ++c) {
        double T = performanceCounters->getTemperatureOfCore(c);
        cout << "[Scheduler][DVFSPredictive]: Core " << setw(2) << c
             << " T=" << fixed << setprecision(1) << T << " C"
             << "  cur=" << setw(4) << oldFrequencies.at(c) << " MHz"
             << "  model=" << setw(4) << frequencies.at(c) << " MHz"
             << (activeCores.at(c) ? "" : " [idle]")
             << endl;
    }

    return frequencies;
}
