/*
 * Assignment 3B : Map Coloring of Indian states as a Constraint Satisfaction Problem
 *
 * Variables   : the 28 states of India
 * Domain      : a set of K colours (default 4, by the Four Colour Theorem 4 always suffice)
 * Constraints : two neighbouring states must not get the same colour
 *
 * Two solvers are run and compared:
 *   1) plain chronological backtracking
 *   2) backtracking + MRV + Degree heuristic + Forward Checking
 *
 * Compile : g++ -std=c++17 -O2 Assignment3B.cpp -o a3b
 */
#include <bits/stdc++.h>
using namespace std;

vector<string> states = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal"
};

vector<pair<string, string>> borders = {
    {"Andhra Pradesh", "Telangana"}, {"Andhra Pradesh", "Odisha"}, {"Andhra Pradesh", "Chhattisgarh"},
    {"Andhra Pradesh", "Karnataka"}, {"Andhra Pradesh", "Tamil Nadu"},
    {"Arunachal Pradesh", "Assam"}, {"Arunachal Pradesh", "Nagaland"},
    {"Assam", "Nagaland"}, {"Assam", "Manipur"}, {"Assam", "Mizoram"}, {"Assam", "Tripura"},
    {"Assam", "Meghalaya"}, {"Assam", "West Bengal"},
    {"Bihar", "Uttar Pradesh"}, {"Bihar", "Jharkhand"}, {"Bihar", "West Bengal"},
    {"Chhattisgarh", "Madhya Pradesh"}, {"Chhattisgarh", "Maharashtra"}, {"Chhattisgarh", "Telangana"},
    {"Chhattisgarh", "Odisha"}, {"Chhattisgarh", "Jharkhand"}, {"Chhattisgarh", "Uttar Pradesh"},
    {"Goa", "Maharashtra"}, {"Goa", "Karnataka"},
    {"Gujarat", "Rajasthan"}, {"Gujarat", "Madhya Pradesh"}, {"Gujarat", "Maharashtra"},
    {"Haryana", "Punjab"}, {"Haryana", "Himachal Pradesh"}, {"Haryana", "Uttarakhand"},
    {"Haryana", "Uttar Pradesh"}, {"Haryana", "Rajasthan"},
    {"Himachal Pradesh", "Punjab"}, {"Himachal Pradesh", "Uttarakhand"}, {"Himachal Pradesh", "Uttar Pradesh"},
    {"Jharkhand", "West Bengal"}, {"Jharkhand", "Odisha"}, {"Jharkhand", "Uttar Pradesh"},
    {"Karnataka", "Maharashtra"}, {"Karnataka", "Telangana"}, {"Karnataka", "Tamil Nadu"},
    {"Karnataka", "Kerala"}, {"Kerala", "Tamil Nadu"},
    {"Madhya Pradesh", "Rajasthan"}, {"Madhya Pradesh", "Uttar Pradesh"}, {"Madhya Pradesh", "Maharashtra"},
    {"Maharashtra", "Telangana"},
    {"Manipur", "Nagaland"}, {"Manipur", "Mizoram"}, {"Mizoram", "Tripura"},
    {"Odisha", "West Bengal"}, {"Punjab", "Rajasthan"}, {"Rajasthan", "Uttar Pradesh"},
    {"Sikkim", "West Bengal"}, {"Uttar Pradesh", "Uttarakhand"}
};

const vector<string> colourNames = {"Red", "Green", "Blue", "Yellow", "Orange", "Purple"};

int n, K;
vector<vector<int>> adj;
long long assignments;

bool consistent(int v, int c, const vector<int> &col) {
    for (int u : adj[v]) if (col[u] == c) return false;
    return true;
}

// ---------- 1) plain backtracking ----------
bool plainBT(int idx, vector<int> &col) {
    if (idx == n) return true;
    for (int c = 0; c < K; c++) {
        if (consistent(idx, c, col)) {
            col[idx] = c; assignments++;
            if (plainBT(idx + 1, col)) return true;
            col[idx] = -1;
        }
    }
    return false;
}

// ---------- 2) MRV + Degree + Forward Checking ----------
bool smartBT(vector<int> &col, vector<vector<bool>> &dom, int assigned) {
    if (assigned == n) return true;
    int v = -1, bestSize = INT_MAX, bestDeg = -1;
    for (int i = 0; i < n; i++) {
        if (col[i] != -1) continue;
        int sz = count(dom[i].begin(), dom[i].end(), true);
        int deg = 0;
        for (int u : adj[i]) if (col[u] == -1) deg++;
        if (sz < bestSize || (sz == bestSize && deg > bestDeg)) { v = i; bestSize = sz; bestDeg = deg; }
    }
    for (int c = 0; c < K; c++) {
        if (!dom[v][c]) continue;
        col[v] = c; assignments++;
        vector<pair<int,int>> removed;                 // forward checking
        bool ok = true;
        for (int u : adj[v]) {
            if (col[u] == -1 && dom[u][c]) {
                dom[u][c] = false; removed.push_back({u, c});
                if (count(dom[u].begin(), dom[u].end(), true) == 0) { ok = false; break; }
            }
        }
        if (ok && smartBT(col, dom, assigned + 1)) return true;
        for (auto &r : removed) dom[r.first][r.second] = true;   // undo
        col[v] = -1;
    }
    return false;
}

bool verify(const vector<int> &col) {
    for (int v = 0; v < n; v++)
        for (int u : adj[v]) if (col[u] == col[v] || col[v] < 0) return false;
    return true;
}

int main() {
    n = states.size();
    map<string, int> id;
    for (int i = 0; i < n; i++) id[states[i]] = i;
    adj.assign(n, {});
    for (auto &b : borders) {
        int u = id.at(b.first), v = id.at(b.second);
        adj[u].push_back(v); adj[v].push_back(u);
    }

    cout << "======= MAP COLORING OF INDIAN STATES (CSP) =======\n";
    cout << "States (variables) : " << n << "\nBorders (constraints): " << borders.size() << "\n";
    cout << "Enter number of colours to use (3-6, 4 recommended) : ";
    cin >> K;
    if (!cin || K < 2 || K > 6) { cout << "Invalid number of colours.\n"; return 1; }

    vector<int> col1(n, -1);
    assignments = 0;
    bool ok1 = plainBT(0, col1);
    long long a1 = assignments;

    vector<int> col2(n, -1);
    vector<vector<bool>> dom(n, vector<bool>(K, true));
    assignments = 0;
    bool ok2 = smartBT(col2, dom, 0);
    long long a2 = assignments;

    if (!ok2) {
        cout << "\nNo valid colouring exists with " << K << " colours.\n";
        cout << "(plain backtracking agrees: " << (ok1 ? "solution found" : "no solution") << ")\n";
        return 0;
    }

    cout << "\nColouring found (MRV + Degree + Forward Checking):\n\n";
    cout << left << setw(20) << "State" << setw(10) << "Colour" << "Neighbours\n";
    cout << string(75, '-') << "\n";
    for (int i = 0; i < n; i++) {
        cout << left << setw(20) << states[i] << setw(10) << colourNames[col2[i]];
        for (size_t j = 0; j < adj[i].size(); j++)
            cout << states[adj[i][j]] << (j + 1 < adj[i].size() ? ", " : "");
        cout << "\n";
    }

    cout << "\nColour groups:\n";
    for (int c = 0; c < K; c++) {
        cout << "  " << setw(7) << left << colourNames[c] << ": ";
        bool first = true;
        for (int i = 0; i < n; i++)
            if (col2[i] == c) { cout << (first ? "" : ", ") << states[i]; first = false; }
        cout << "\n";
    }

    cout << "\nVerification (no two neighbours share a colour) : "
         << (verify(col2) ? "PASSED" : "FAILED") << "\n";
    cout << "\nSolver comparison (number of variable assignments tried):\n";
    cout << "  Plain backtracking        : " << a1 << (ok1 ? "" : "  (no solution)") << "\n";
    cout << "  MRV + Degree + FwdCheck   : " << a2 << "\n";
    return 0;
}
