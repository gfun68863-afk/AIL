/*
 * Assignment 1B : N x N sliding tile puzzle (8-puzzle, 15-puzzle, ...)
 * Algorithm : A* search  f(n) = g(n) + h(n)
 * Heuristic : Manhattan distance (compared with Misplaced Tiles)
 * Tiles are numbered 1..N^2-1, 0 denotes the blank space.
 * Goal state : 1 2 3 ... N^2-1 0
 *
 * Compile : g++ -std=c++17 -O2 Assignment1B.cpp -o a1b
 */
#include <bits/stdc++.h>
using namespace std;

int N;
string goalState;

struct Stats { bool solved = false; int moves = 0; long long expanded = 0, generated = 0; };

int manhattan(const string &s) {
    int d = 0;
    for (int i = 0; i < N * N; i++) {
        int v = s[i];
        if (v == 0) continue;
        int gi = v - 1;
        d += abs(i / N - gi / N) + abs(i % N - gi % N);
    }
    return d;
}

int misplaced(const string &s) {
    int c = 0;
    for (int i = 0; i < N * N; i++) if (s[i] != 0 && s[i] != goalState[i]) c++;
    return c;
}

int heuristic(const string &s, int type) { return type == 0 ? manhattan(s) : misplaced(s); }

bool isSolvable(const string &s) {
    int inv = 0;
    for (int i = 0; i < N * N; i++)
        for (int j = i + 1; j < N * N; j++)
            if (s[i] && s[j] && s[i] > s[j]) inv++;
    if (N % 2 == 1) return inv % 2 == 0;
    int blankRow = (int)(s.find((char)0)) / N;
    int fromBottom = N - blankRow;               // 1-based row counted from the bottom
    return (inv + fromBottom) % 2 == 1;
}

void printBoard(const string &s) {
    for (int r = 0; r < N; r++) {
        cout << "   ";
        for (int c = 0; c < N; c++) {
            int v = s[r * N + c];
            if (v == 0) cout << setw(3) << "_";
            else cout << setw(3) << v;
        }
        cout << "\n";
    }
}

const int dr[4] = {-1, 1, 0, 0};
const int dc[4] = {0, 0, -1, 1};
const char *moveName[4] = {"Down", "Up", "Right", "Left"};  // direction the TILE slides

struct QNode {
    int f, g; string s;
};
struct Cmp {
    bool operator()(const QNode &a, const QNode &b) const {
        if (a.f != b.f) return a.f > b.f;
        return a.g < b.g;                         // prefer deeper nodes on ties
    }
};

Stats solve(const string &start, int hType, bool show) {
    Stats st;
    unordered_map<string, int> best;
    unordered_map<string, pair<string,int>> parent;   // state -> (previous state, move)
    priority_queue<QNode, vector<QNode>, Cmp> pq;
    best[start] = 0;
    pq.push({heuristic(start, hType), 0, start});
    const long long LIMIT = 3000000;
    string found;

    while (!pq.empty()) {
        QNode cur = pq.top(); pq.pop();
        if (cur.g > best[cur.s]) continue;                // stale queue entry
        st.expanded++;
        if (cur.s == goalState) { found = cur.s; st.solved = true; st.moves = cur.g; break; }
        if (st.expanded > LIMIT) break;
        int z = (int)cur.s.find((char)0), r = z / N, c = z % N;
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            string nxt = cur.s;
            swap(nxt[z], nxt[nr * N + nc]);
            int ng = cur.g + 1;
            auto it = best.find(nxt);
            if (it == best.end() || ng < it->second) {
                best[nxt] = ng;
                parent[nxt] = {cur.s, k};
                pq.push({ng + heuristic(nxt, hType), ng, nxt});
                st.generated++;
            }
        }
    }

    if (st.solved && show) {
        vector<pair<string,int>> steps;
        string s = found;
        while (s != start) { auto p = parent[s]; steps.push_back({s, p.second}); s = p.first; }
        reverse(steps.begin(), steps.end());
        cout << "\nStart state:\n"; printBoard(start);
        int i = 1;
        for (auto &p : steps) {
            int z = (int)p.first.find((char)0);
            // blank moved in direction k, so the tile that moved is now where the blank was
            int k = p.second;
            int tr = z / N - dr[k] , tc = z % N - dc[k];
            cout << "\nStep " << i++ << ": slide tile " << (int)p.first[tr * N + tc] << " "
                 << moveName[k] << "\n";
            printBoard(p.first);
        }
    }
    return st;
}

int main() {
    cout << "=========== N x N SLIDING PUZZLE (A* SEARCH) ===========\n";
    cout << "Enter board size N (2 - 4, e.g. 3 for 8-puzzle) : ";
    cin >> N;
    if (!cin || N < 2 || N > 4) { cout << "Invalid N\n"; return 1; }

    goalState.assign(N * N, 0);
    for (int i = 0; i < N * N - 1; i++) goalState[i] = (char)(i + 1);
    goalState[N * N - 1] = 0;

    cout << "\n1. Enter initial board manually\n2. Generate random board\nChoice : ";
    int choice; cin >> choice;
    string start(N * N, 0);

    if (choice == 1) {
        cout << "Enter " << N * N << " numbers (0 = blank) row by row:\n";
        vector<bool> seen(N * N, false);
        for (int i = 0; i < N * N; i++) {
            int v; cin >> v;
            if (!cin || v < 0 || v >= N * N || seen[v]) { cout << "Invalid board!\n"; return 1; }
            seen[v] = true; start[i] = (char)v;
        }
    } else {
        mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
        start = goalState;
        int prev = -1, z = N * N - 1;
        int steps = (N == 4) ? 22 : 40;
        for (int i = 0; i < steps; i++) {
            vector<int> opts;
            for (int k = 0; k < 4; k++) {
                int nr = z / N + dr[k], nc = z % N + dc[k];
                if (nr >= 0 && nr < N && nc >= 0 && nc < N && k != (prev ^ 1)) opts.push_back(k);
            }
            int k = opts[rng() % opts.size()];
            int nz = (z / N + dr[k]) * N + (z % N + dc[k]);
            swap(start[z], start[nz]); z = nz; prev = k;
        }
    }

    cout << "\nInitial board:\n"; printBoard(start);
    cout << "\nGoal board:\n";    printBoard(goalState);

    if (!isSolvable(start)) {
        cout << "\nThis configuration is NOT solvable (parity of inversions is wrong).\n";
        return 0;
    }

    Stats a = solve(start, 0, true);
    if (!a.solved) { cout << "\nSearch limit reached without a solution.\n"; return 0; }

    Stats b = solve(start, 1, false);
    cout << "\n================ RESULT ================\n";
    cout << "Optimal number of moves : " << a.moves << "\n\n";
    cout << left << setw(26) << "Heuristic" << right << setw(10) << "Moves"
         << setw(12) << "Expanded" << setw(12) << "Generated" << "\n";
    cout << left << setw(26) << "Manhattan distance" << right << setw(10) << a.moves
         << setw(12) << a.expanded << setw(12) << a.generated << "\n";
    cout << left << setw(26) << "Misplaced tiles" << right << setw(10) << b.moves
         << setw(12) << b.expanded << setw(12) << b.generated << "\n";
    cout << "\nManhattan distance dominates misplaced tiles, so A* expands fewer nodes\n"
         << "while still returning the optimal (shortest) solution.\n";
    return 0;
}
