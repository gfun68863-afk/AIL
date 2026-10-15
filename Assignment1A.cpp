/*
 * Assignment 1A : Shortest path between two cities (Google-Maps like system)
 * Uninformed search : BFS, DFS, Uniform Cost Search (UCS)
 * Informed search   : Greedy Best-First Search, A*
 * Edge cost   = road distance (km)
 * Heuristic   = straight-line (haversine) distance to goal (km)
 *
 * Compile : g++ -std=c++17 -O2 Assignment1A.cpp -o a1a
 */
#include <bits/stdc++.h>
using namespace std;

struct City { string name; double lat, lon; };

vector<City> cities = {
    {"Pune",       18.52, 73.86}, {"Mumbai",    19.08, 72.88},
    {"Nashik",     19.99, 73.79}, {"Aurangabad",19.88, 75.34},
    {"Nagpur",     21.15, 79.09}, {"Hyderabad", 17.39, 78.49},
    {"Solapur",    17.66, 75.91}, {"Kolhapur",  16.70, 74.24},
    {"Belgaum",    15.85, 74.50}, {"Goa",       15.49, 73.83},
    {"Bangalore",  12.97, 77.59}
};

struct Edge { int u, v; double w; };
vector<Edge> edgeList = {
    {0,1,150}, {0,2,210}, {0,3,235}, {0,6,250}, {0,7,230},
    {1,2,170}, {1,9,590}, {2,3,180}, {3,4,480}, {3,5,590},
    {4,5,500}, {5,6,305}, {5,10,570}, {7,8,115}, {8,9,150},
    {8,10,510}
};

int n;
vector<vector<pair<int,double>>> adj;   // adjacency list (neighbour, cost)

double haversine(int a, int b) {
    const double R = 6371.0;
    auto rad = [](double d) { return d * M_PI / 180.0; };
    double dlat = rad(cities[b].lat - cities[a].lat);
    double dlon = rad(cities[b].lon - cities[a].lon);
    double h = sin(dlat / 2) * sin(dlat / 2) +
               cos(rad(cities[a].lat)) * cos(rad(cities[b].lat)) *
               sin(dlon / 2) * sin(dlon / 2);
    return 2 * R * asin(sqrt(h));
}

struct Result {
    string algo;
    bool found = false;
    vector<int> path;
    double cost = 0;
    int expanded = 0;
    vector<int> order;     // order in which nodes were expanded
};

double edgeCost(int u, int v) {
    for (auto &e : adj[u]) if (e.first == v) return e.second;
    return 0;
}

void finish(Result &r, const vector<int> &parent, int src, int dst) {
    if (parent[dst] == -1 && src != dst) return;
    r.found = true;
    for (int x = dst; x != -1; x = parent[x]) r.path.push_back(x);
    reverse(r.path.begin(), r.path.end());
    for (size_t i = 1; i < r.path.size(); i++)
        r.cost += edgeCost(r.path[i - 1], r.path[i]);
}

// ---------------- Uninformed searches ----------------
Result bfs(int src, int dst) {
    Result r; r.algo = "BFS";
    vector<int> parent(n, -1); vector<bool> vis(n, false);
    queue<int> q; q.push(src); vis[src] = true;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        r.expanded++; r.order.push_back(u);
        if (u == dst) break;
        for (auto &e : adj[u])
            if (!vis[e.first]) { vis[e.first] = true; parent[e.first] = u; q.push(e.first); }
    }
    finish(r, parent, src, dst);
    return r;
}

bool dfsRec(int u, int dst, vector<bool> &vis, vector<int> &parent, Result &r) {
    vis[u] = true; r.expanded++; r.order.push_back(u);
    if (u == dst) return true;
    for (auto &e : adj[u])
        if (!vis[e.first]) {
            parent[e.first] = u;
            if (dfsRec(e.first, dst, vis, parent, r)) return true;
        }
    return false;
}

Result dfs(int src, int dst) {
    Result r; r.algo = "DFS";
    vector<int> parent(n, -1); vector<bool> vis(n, false);
    dfsRec(src, dst, vis, parent, r);
    finish(r, parent, src, dst);
    return r;
}

Result ucs(int src, int dst) {
    Result r; r.algo = "UCS (Dijkstra)";
    vector<double> g(n, 1e18); vector<int> parent(n, -1); vector<bool> done(n, false);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    g[src] = 0; pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (done[u]) continue;
        done[u] = true; r.expanded++; r.order.push_back(u);
        if (u == dst) break;
        for (auto &e : adj[u])
            if (d + e.second < g[e.first]) {
                g[e.first] = d + e.second; parent[e.first] = u;
                pq.push({g[e.first], e.first});
            }
    }
    finish(r, parent, src, dst);
    return r;
}

// ---------------- Informed searches ----------------
Result greedy(int src, int dst, const vector<double> &H) {
    Result r; r.algo = "Greedy Best-First";
    vector<int> parent(n, -1); vector<bool> vis(n, false);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    pq.push({H[src], src}); vis[src] = true;
    while (!pq.empty()) {
        int u = pq.top().second; pq.pop();
        r.expanded++; r.order.push_back(u);
        if (u == dst) break;
        for (auto &e : adj[u])
            if (!vis[e.first]) {
                vis[e.first] = true; parent[e.first] = u;
                pq.push({H[e.first], e.first});
            }
    }
    finish(r, parent, src, dst);
    return r;
}

Result astar(int src, int dst, const vector<double> &H) {
    Result r; r.algo = "A* Search";
    vector<double> g(n, 1e18); vector<int> parent(n, -1); vector<bool> done(n, false);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    g[src] = 0; pq.push({H[src], src});
    while (!pq.empty()) {
        int u = pq.top().second; pq.pop();
        if (done[u]) continue;
        done[u] = true; r.expanded++; r.order.push_back(u);
        if (u == dst) break;
        for (auto &e : adj[u]) {
            double ng = g[u] + e.second;
            if (ng < g[e.first]) {
                g[e.first] = ng; parent[e.first] = u;
                pq.push({ng + H[e.first], e.first});
            }
        }
    }
    finish(r, parent, src, dst);
    return r;
}

void printResult(const Result &r) {
    cout << "\n[" << r.algo << "]\n";
    if (!r.found) { cout << "  No path found.\n"; return; }
    cout << "  Path : ";
    for (size_t i = 0; i < r.path.size(); i++)
        cout << cities[r.path[i]].name << (i + 1 < r.path.size() ? " -> " : "\n");
    cout << "  Total distance : " << r.cost << " km\n";
    cout << "  Hops           : " << r.path.size() - 1 << "\n";
    cout << "  Nodes expanded : " << r.expanded << "\n";
    cout << "  Expansion order: ";
    for (size_t i = 0; i < r.order.size(); i++)
        cout << cities[r.order[i]].name << (i + 1 < r.order.size() ? ", " : "\n");
}

int main() {
    n = cities.size();
    adj.assign(n, {});
    for (auto &e : edgeList) {
        adj[e.u].push_back({e.v, e.w});
        adj[e.v].push_back({e.u, e.w});
    }
    for (auto &a : adj) sort(a.begin(), a.end());

    cout << "=========== CITY ROUTE FINDER ===========\n";
    cout << "Cities (" << n << " nodes) :\n";
    for (int i = 0; i < n; i++) cout << "  " << setw(2) << i << " : " << cities[i].name << "\n";
    cout << "\nRoads (" << edgeList.size() << " edges, cost in km) :\n";
    for (auto &e : edgeList)
        cout << "  " << setw(10) << left << cities[e.u].name << " <-> "
             << setw(10) << cities[e.v].name << right << setw(5) << e.w << " km\n";

    char again = 'y';
    while (again == 'y' || again == 'Y') {
        int src, dst;
        cout << "\nEnter source city number      : "; cin >> src;
        cout << "Enter destination city number : "; cin >> dst;
        if (!cin || src < 0 || src >= n || dst < 0 || dst >= n) {
            cout << "Invalid city number.\n"; return 1;
        }

        vector<double> H(n);
        for (int i = 0; i < n; i++) H[i] = haversine(i, dst);
        bool admissible = true;
        for (auto &e : edgeList)
            if (H[e.u] > e.w + H[e.v] + 1e-9 || H[e.v] > e.w + H[e.u] + 1e-9) admissible = false;

        cout << "\nHeuristic h(n) = straight-line distance to " << cities[dst].name << " :\n";
        for (int i = 0; i < n; i++)
            cout << "  " << setw(10) << left << cities[i].name << right
                 << fixed << setprecision(1) << setw(8) << H[i] << " km\n";
        cout << "Heuristic consistent for this goal : " << (admissible ? "yes" : "no") << "\n";
        cout << setprecision(0);

        vector<Result> res = { bfs(src, dst), dfs(src, dst), ucs(src, dst),
                               greedy(src, dst, H), astar(src, dst, H) };
        for (auto &r : res) printResult(r);

        cout << "\n================ COMPARISON ================\n";
        cout << left << setw(20) << "Algorithm" << right << setw(12) << "Distance(km)"
             << setw(8) << "Hops" << setw(10) << "Expanded" << setw(10) << "Optimal?" << "\n";
        double best = 1e18;
        for (auto &r : res) if (r.found) best = min(best, r.cost);
        for (auto &r : res) {
            cout << left << setw(20) << r.algo << right << setw(12) << r.cost
                 << setw(8) << (r.found ? (int)r.path.size() - 1 : 0)
                 << setw(10) << r.expanded
                 << setw(10) << (r.found && r.cost <= best + 1e-9 ? "Yes" : "No") << "\n";
        }
        cout << "\nAnalysis:\n"
             << " * BFS finds the path with fewest hops, but not necessarily the shortest distance.\n"
             << " * DFS may return a very long route; it is neither optimal nor complete in general.\n"
             << " * UCS is optimal but expands many nodes (no knowledge about the goal direction).\n"
             << " * Greedy is fast but can be misled by the heuristic and is not optimal.\n"
             << " * A* is optimal (admissible heuristic) and expands fewer nodes than UCS.\n";

        cout << "\nFind another route? (y/n) : "; cin >> again;
        if (!cin) break;
    }
    return 0;
}
