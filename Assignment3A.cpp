/*
 * Assignment 3A : Intelligent Traffic Control under uncertain traffic conditions
 *
 * 4-way intersection  (North, South, East, West).
 *   Phase 0 = North-South green      Phase 1 = East-West green
 * Uncertainty  : vehicle arrivals are random (Bernoulli per second) and the traffic
 *                pattern changes over time (unknown to the controller);
 *                sensors give NOISY queue counts and sometimes FAIL (drop-out).
 * Adaptive controller : keeps a belief (exponential moving average) of the queue on
 *                every approach and splits the green time of the next cycle in
 *                proportion to the expected demand of each phase.
 * Baseline     : fixed-time controller (equal green for both phases).
 * Both controllers are simulated on exactly the same random arrivals.
 *
 * Compile : g++ -std=c++17 -O2 Assignment3A.cpp -o a3a
 */
#include <bits/stdc++.h>
using namespace std;

const int APPROACHES = 4;                 // 0=N 1=S 2=E 3=W
const char *NAME[4] = {"North", "South", "East", "West"};
const int SIM_TIME = 3600;                // one hour, 1 step = 1 second
const int LOST_TIME = 4;                  // yellow + all-red after every green
const int TOTAL_GREEN = 60;               // green seconds shared by two phases per cycle
const int MIN_GREEN = 10, MAX_GREEN = 50;
const double SAT_FLOW = 0.5;              // vehicles / second discharged on green

// arrival[t][a] = 1 if a vehicle arrives on approach a at second t
vector<array<int, 4>> arrivals;

double arrivalRate(int t, int a) {
    // three unknown traffic regimes (each 20 minutes)
    static const double R[3][4] = {
        {0.25, 0.22, 0.06, 0.07},         // morning : North-South heavy
        {0.06, 0.05, 0.26, 0.23},         // midday  : East-West heavy
        {0.14, 0.14, 0.14, 0.14}          // evening : balanced
    };
    return R[min(2, t / 1200)][a];
}

struct Metrics {
    double sumQueue = 0; int maxQueue = 0; long served = 0; int finalQueue = 0;
};

struct Controller {
    string name;
    bool adaptive;
    array<double, 4> belief{};            // estimated queue length per approach
    mt19937 rng{7};
    normal_distribution<double> noise{0.0, 2.0};
    uniform_real_distribution<double> uni{0.0, 1.0};

    // choose green times (g_NS, g_EW) from noisy observations
    pair<int, int> planCycle(const array<int, 4> &q) {
        if (!adaptive) return {TOTAL_GREEN / 2, TOTAL_GREEN / 2};
        for (int a = 0; a < 4; a++) {
            if (uni(rng) < 0.05) continue;                         // sensor drop-out: keep old belief
            double obs = max(0.0, q[a] + noise(rng));              // noisy measurement
            belief[a] = 0.5 * belief[a] + 0.5 * obs;               // belief update (EMA filter)
        }
        double dNS = max(belief[0], belief[1]) + 1.0;              // expected demand of phase
        double dEW = max(belief[2], belief[3]) + 1.0;
        double share = dNS / (dNS + dEW);
        int gNS = (int)lround(MIN_GREEN + (TOTAL_GREEN - 2 * MIN_GREEN) * share);
        gNS = max(MIN_GREEN, min(MAX_GREEN, gNS));
        return {gNS, TOTAL_GREEN - gNS};
    }
};

Metrics simulate(Controller &ctl, bool verbose) {
    Metrics m;
    array<int, 4> q = {0, 0, 0, 0};
    array<double, 4> credit = {0, 0, 0, 0};
    int t = 0, cycle = 0;
    if (verbose) {
        cout << "\nCycle  Time  Queues(N S E W)   Belief(N S E W)        Green NS/EW\n";
    }
    while (t < SIM_TIME) {
        auto g = ctl.planCycle(q);
        if (verbose && cycle < 12) {
            cout << setw(4) << cycle + 1 << setw(7) << t << "   "
                 << setw(3) << q[0] << setw(3) << q[1] << setw(3) << q[2] << setw(3) << q[3]
                 << "      " << fixed << setprecision(1)
                 << setw(6) << ctl.belief[0] << setw(6) << ctl.belief[1]
                 << setw(6) << ctl.belief[2] << setw(6) << ctl.belief[3]
                 << "        " << g.first << "/" << g.second << "\n";
        }
        for (int phase = 0; phase < 2 && t < SIM_TIME; phase++) {
            int green = phase == 0 ? g.first : g.second;
            for (int s = 0; s < green + LOST_TIME && t < SIM_TIME; s++, t++) {
                for (int a = 0; a < 4; a++) q[a] += arrivals[t][a];
                if (s < green) {
                    for (int a = 0; a < 4; a++) {
                        bool hasGreen = (phase == 0) ? (a < 2) : (a >= 2);
                        if (!hasGreen) continue;
                        credit[a] += SAT_FLOW;
                        while (credit[a] >= 1.0 && q[a] > 0) { q[a]--; credit[a] -= 1.0; m.served++; }
                        if (q[a] == 0) credit[a] = 0;
                    }
                }
                int total = q[0] + q[1] + q[2] + q[3];
                m.sumQueue += total;
                m.maxQueue = max(m.maxQueue, *max_element(q.begin(), q.end()));
            }
        }
        cycle++;
    }
    m.finalQueue = q[0] + q[1] + q[2] + q[3];
    return m;
}

int main() {
    mt19937 gen(2024);
    uniform_real_distribution<double> uni(0.0, 1.0);
    arrivals.assign(SIM_TIME, {0, 0, 0, 0});
    long totalArr = 0;
    for (int t = 0; t < SIM_TIME; t++)
        for (int a = 0; a < 4; a++) { arrivals[t][a] = uni(gen) < arrivalRate(t, a); totalArr += arrivals[t][a]; }

    cout << "======= ADAPTIVE TRAFFIC SIGNAL CONTROL (uncertain traffic) =======\n";
    cout << "Simulated time : " << SIM_TIME / 60 << " minutes, vehicles generated : " << totalArr << "\n";
    cout << "Regimes : 0-20 min North-South heavy, 20-40 min East-West heavy, 40-60 min balanced\n";

    Controller fixedCtl{"Fixed-time (30s/30s)", false};
    Controller adaptCtl{"Adaptive (belief based)", true};

    cout << "\n--- Adaptive controller : first 12 cycles ---";
    Metrics ma = simulate(adaptCtl, true);
    Metrics mf = simulate(fixedCtl, false);

    cout << "\n\n================ PERFORMANCE COMPARISON ================\n";
    cout << left << setw(26) << "Controller" << right << setw(14) << "Avg queue"
         << setw(11) << "Max queue" << setw(10) << "Served" << setw(16) << "Left in queue" << "\n";
    for (auto pr : {make_pair(&fixedCtl, &mf), make_pair(&adaptCtl, &ma)}) {
        cout << left << setw(26) << pr.first->name << right << fixed << setprecision(2)
             << setw(14) << pr.second->sumQueue / SIM_TIME << setw(11) << pr.second->maxQueue
             << setw(10) << pr.second->served << setw(16) << pr.second->finalQueue << "\n";
    }
    double imp = 100.0 * (mf.sumQueue - ma.sumQueue) / mf.sumQueue;
    cout << "\nAdaptive control reduced the average number of waiting vehicles by "
         << setprecision(1) << imp << "%\n";
    cout << "It handles uncertainty by filtering noisy sensor data and re-allocating green time\n"
         << "to the approach that is expected to have the larger demand.\n";
    return 0;
}
