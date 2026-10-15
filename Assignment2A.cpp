/*
 * Assignment 2A : Smart Parking Allocation (Smart-City style)
 * Slots have : floor, size (S/M/L), EV charger, handicap reserved, distance from entrance.
 * Hard constraints  : slot must be free, slot size >= vehicle size,
 *                     EV vehicle needs a charger slot,
 *                     handicap-reserved slot only for handicapped drivers.
 * Soft cost (minimised) : walking distance + wasted-space penalty
 *                         + penalty for blocking a scarce EV / handicap slot.
 * Vehicles are served by priority (handicap > EV > larger vehicle > first come).
 *
 * Compile : g++ -std=c++17 -O2 Assignment2A.cpp -o a2a
 */
#include <bits/stdc++.h>
using namespace std;

struct Slot {
    int id, floor;
    char size;           // 'S' small, 'M' medium, 'L' large
    bool ev, handicap;
    int dist;            // metres from entrance
    bool occupied = false;
    string plate;
};

struct Vehicle {
    string plate;
    char size;
    bool needEV, needHandicap;
    int arrival;
};

vector<Slot> lot;

int rankOf(char s) { return s == 'S' ? 1 : (s == 'M' ? 2 : 3); }

void initLot() {
    // id, floor, size, ev, handicap, dist
    lot = {
        {1, 0, 'S', false, false, 10}, {2, 0, 'M', false, false, 12},
        {3, 0, 'L', false, false, 15}, {4, 0, 'M', false, true,   5},
        {5, 0, 'M', true,  false, 20}, {6, 0, 'L', true,  false, 25},
        {7, 1, 'S', false, false, 40}, {8, 1, 'M', false, false, 42},
        {9, 1, 'M', true,  false, 45}, {10,1, 'L', false, false, 50},
        {11,1, 'S', false, false, 55}, {12,1, 'L', true,  false, 60}
    };
}

bool feasible(const Slot &s, const Vehicle &v) {
    if (s.occupied) return false;
    if (rankOf(s.size) < rankOf(v.size)) return false;
    if (v.needEV && !s.ev) return false;
    if (s.handicap && !v.needHandicap) return false;
    return true;
}

int cost(const Slot &s, const Vehicle &v) {
    int c = s.dist;
    c += 8 * (rankOf(s.size) - rankOf(v.size));       // wasted space
    if (s.ev && !v.needEV) c += 15;                    // keep EV slots for EVs
    if (s.handicap && v.needHandicap) c -= 5;          // slightly prefer reserved slot
    c += 20 * s.floor;                                 // stairs / ramp time
    return c;
}

int allocate(const Vehicle &v) {
    int best = -1, bestCost = INT_MAX;
    for (int i = 0; i < (int)lot.size(); i++)
        if (feasible(lot[i], v)) {
            int c = cost(lot[i], v);
            if (c < bestCost) { bestCost = c; best = i; }
        }
    if (best != -1) { lot[best].occupied = true; lot[best].plate = v.plate; }
    return best;
}

void showLot() {
    cout << "\n Slot Floor Size  EV  Handicap  Dist  Status\n";
    cout << " ------------------------------------------------------\n";
    int freeCnt = 0;
    for (auto &s : lot) {
        cout << setw(5) << s.id << setw(6) << s.floor << setw(5) << s.size
             << setw(6) << (s.ev ? "Yes" : "No") << setw(8) << (s.handicap ? "Yes" : "No")
             << setw(8) << s.dist << "  "
             << (s.occupied ? "Occupied (" + s.plate + ")" : string("Free")) << "\n";
        if (!s.occupied) freeCnt++;
    }
    cout << " Free slots : " << freeCnt << " / " << lot.size() << "\n";
}

bool priorityOrder(const Vehicle &a, const Vehicle &b) {
    if (a.needHandicap != b.needHandicap) return a.needHandicap;
    if (a.needEV != b.needEV) return a.needEV;
    if (a.size != b.size) return rankOf(a.size) > rankOf(b.size);
    return a.arrival < b.arrival;
}

void describe(const Vehicle &v) {
    cout << v.plate << " (size " << v.size << (v.needEV ? ", EV" : "")
         << (v.needHandicap ? ", handicap" : "") << ")";
}

void runBatch(vector<Vehicle> vs) {
    stable_sort(vs.begin(), vs.end(), priorityOrder);
    cout << "\nProcessing " << vs.size() << " vehicles in priority order...\n";
    int ok = 0, totalDist = 0;
    for (auto &v : vs) {
        int idx = allocate(v);
        cout << "  "; describe(v);
        if (idx < 0) cout << "  -> NO SUITABLE SLOT AVAILABLE\n";
        else {
            cout << "  -> Slot " << lot[idx].id << " (floor " << lot[idx].floor
                 << ", " << lot[idx].dist << " m)\n";
            ok++; totalDist += lot[idx].dist;
        }
    }
    cout << "Allocated " << ok << "/" << vs.size() << ", average walking distance = "
         << (ok ? (double)totalDist / ok : 0.0) << " m\n";
}

int main() {
    initLot();
    int counter = 0;
    int choice;
    do {
        cout << "\n======= SMART PARKING SYSTEM =======\n"
             << "1. Show parking lot\n2. Park a vehicle\n3. Release a vehicle\n"
             << "4. Run demo batch of 9 vehicles\n0. Exit\nChoice : ";
        if (!(cin >> choice)) break;
        if (choice == 1) showLot();
        else if (choice == 2) {
            Vehicle v; int ev, hc;
            cout << "Plate number            : "; cin >> v.plate;
            cout << "Size (S/M/L)            : "; cin >> v.size; v.size = toupper(v.size);
            cout << "Needs EV charger (0/1)  : "; cin >> ev;
            cout << "Handicap permit (0/1)   : "; cin >> hc;
            if (v.size != 'S' && v.size != 'M' && v.size != 'L') { cout << "Invalid size\n"; continue; }
            v.needEV = ev; v.needHandicap = hc; v.arrival = ++counter;
            int idx = allocate(v);
            if (idx < 0) cout << "Sorry, no slot satisfies the constraints.\n";
            else cout << "Allocated slot " << lot[idx].id << " on floor " << lot[idx].floor
                      << " (" << lot[idx].dist << " m from entrance)\n";
        } else if (choice == 3) {
            string p; cout << "Plate number : "; cin >> p;
            bool done = false;
            for (auto &s : lot)
                if (s.occupied && s.plate == p) { s.occupied = false; s.plate = ""; done = true;
                    cout << "Slot " << s.id << " released.\n"; break; }
            if (!done) cout << "Vehicle not found.\n";
        } else if (choice == 4) {
            vector<Vehicle> demo = {
                {"MH12-AB-1001", 'S', false, false, 1}, {"MH12-CD-2002", 'M', true,  false, 2},
                {"MH14-EF-3003", 'L', false, false, 3}, {"MH12-GH-4004", 'M', false, true,  4},
                {"MH09-IJ-5005", 'S', true,  false, 5}, {"MH12-KL-6006", 'L', true,  false, 6},
                {"MH12-MN-7007", 'M', false, false, 7}, {"MH12-OP-8008", 'L', true,  false, 8},
                {"MH12-QR-9009", 'M', false, true,  9}
            };
            runBatch(demo);
            showLot();
        }
    } while (choice != 0);
    cout << "Goodbye!\n";
    return 0;
}
