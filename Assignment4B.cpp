/*
 * Assignment 4B : AI agent that learns to play a simple game using Reinforcement Learning
 *
 * Game  : "Treasure Hunt" grid world (5 x 7)
 *         S = start, T = treasure (+100, episode ends), P = pit (-100, episode ends),
 *         # = wall (cannot enter), . = empty cell (every step costs -1)
 * Agent : Q-learning (off-policy temporal-difference control)
 *         Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
 * Policy: epsilon-greedy with decaying epsilon.
 *
 * Compile : g++ -std=c++17 -O2 Assignment4B.cpp -o a4b
 */
#include <bits/stdc++.h>
using namespace std;

const int R = 5, C = 7;
const char *MAP[R] = {
    "S..#...",
    ".#.#.#.",
    ".#...#P",
    ".#.#.#.",
    "...#..T"
};
const int dr[4] = {-1, 1, 0, 0};
const int dc[4] = {0, 0, -1, 1};
const char ARROW[4] = {'^', 'v', '<', '>'};
const char *ACTION[4] = {"Up", "Down", "Left", "Right"};

struct Env {
    int sr = 0, sc = 0, r, c;
    Env() {
        for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) if (MAP[i][j] == 'S') { sr = i; sc = j; }
        reset();
    }
    int reset() { r = sr; c = sc; return state(); }
    int state() const { return r * C + c; }
    // returns reward, sets done
    double step(int a, bool &done) {
        int nr = r + dr[a], nc = c + dc[a];
        done = false;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C || MAP[nr][nc] == '#') return -2;   // bump into wall
        r = nr; c = nc;
        if (MAP[r][c] == 'T') { done = true; return 100; }
        if (MAP[r][c] == 'P') { done = true; return -100; }
        return -1;
    }
};

vector<array<double, 4>> Q(R * C, {0, 0, 0, 0});

int bestAction(int s) {
    int b = 0;
    for (int a = 1; a < 4; a++) if (Q[s][a] > Q[s][b]) b = a;
    return b;
}

void printMap() {
    cout << "Game map:\n";
    for (int i = 0; i < R; i++) cout << "   " << MAP[i] << "\n";
    cout << "   (S start, T treasure +100, P pit -100, # wall, step cost -1)\n";
}

void printPolicy() {
    cout << "\nLearned policy (best action in every cell):\n";
    for (int i = 0; i < R; i++) {
        cout << "   ";
        for (int j = 0; j < C; j++) {
            char m = MAP[i][j];
            if (m == '#' || m == 'T' || m == 'P') cout << m << ' ';
            else cout << ARROW[bestAction(i * C + j)] << ' ';
        }
        cout << "\n";
    }
}

int main() {
    mt19937 rng(42);
    uniform_real_distribution<double> U(0, 1);
    Env env;

    const int EPISODES = 5000, MAX_STEPS = 200;
    const double alpha = 0.1, gamma_ = 0.95;
    double eps = 1.0;
    const double epsMin = 0.05, epsDecay = 0.999;

    cout << "======= Q-LEARNING AGENT : TREASURE HUNT GAME =======\n";
    printMap();
    cout << "\nTraining for " << EPISODES << " episodes (alpha=" << alpha << ", gamma=" << gamma_
         << ", epsilon 1.0 -> " << epsMin << ")\n\n";
    cout << " Episodes      Avg reward   Treasure reached   Epsilon\n";

    double sumReward = 0; int wins = 0;
    for (int ep = 1; ep <= EPISODES; ep++) {
        int s = env.reset();
        double total = 0; bool done = false; bool won = false;
        for (int t = 0; t < MAX_STEPS && !done; t++) {
            int a = (U(rng) < eps) ? (int)(rng() % 4) : bestAction(s);
            double rew = env.step(a, done);
            int s2 = env.state();
            double target = rew + (done ? 0.0 : gamma_ * *max_element(Q[s2].begin(), Q[s2].end()));
            Q[s][a] += alpha * (target - Q[s][a]);
            total += rew; s = s2;
            if (done && rew == 100) won = true;
        }
        sumReward += total; if (won) wins++;
        eps = max(epsMin, eps * epsDecay);
        if (ep % 500 == 0) {
            cout << setw(5) << ep - 499 << " - " << setw(5) << ep << fixed << setprecision(1)
                 << setw(12) << sumReward / 500 << setw(14) << 100.0 * wins / 500 << " %"
                 << setw(12) << setprecision(3) << eps << "\n";
            sumReward = 0; wins = 0;
        }
    }

    printPolicy();

    // play one greedy game using the learned Q-table
    cout << "\nGreedy play-through after training:\n";
    int s = env.reset(); bool done = false; double total = 0; int steps = 0;
    cout << "   Start at (" << env.r << "," << env.c << ")\n";
    while (!done && steps < 50) {
        int a = bestAction(s);
        double rew = env.step(a, done);
        total += rew; steps++; s = env.state();
        cout << "   Step " << setw(2) << steps << ": " << setw(5) << left << ACTION[a] << right
             << " -> (" << env.r << "," << env.c << ")  reward " << setw(4) << (int)rew << "\n";
    }
    cout << "   Game finished in " << steps << " steps, total reward = " << (int)total << "\n";
    cout << (MAP[env.r][env.c] == 'T' ? "   The agent found the treasure!\n" : "   The agent failed.\n");

    // evaluation without exploration
    int success = 0; const int TESTS = 100;
    for (int i = 0; i < TESTS; i++) {
        s = env.reset(); done = false;
        for (int t = 0; t < 50 && !done; t++) { env.step(bestAction(s), done); s = env.state(); }
        if (done && MAP[env.r][env.c] == 'T') success++;
    }
    cout << "\nEvaluation (greedy policy, " << TESTS << " games) : success rate = " << success << " %\n";
    return 0;
}
