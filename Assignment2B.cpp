/*
 * Assignment 2B : Tic-Tac-Toe agent using Minimax with Alpha-Beta Pruning
 * The human plays 'X', the AI plays 'O'.
 * Utility (from AI's point of view):  +10 - depth  for AI win,
 *                                     depth - 10   for human win,  0 for draw.
 * The program also reports how many nodes plain Minimax and Alpha-Beta visit
 * for every AI move, to show the benefit of pruning.
 *
 * Compile : g++ -std=c++17 -O2 Assignment2B.cpp -o a2b
 */
#include <bits/stdc++.h>
using namespace std;

const char AI = 'O', HUMAN = 'X';
long long nodeCount;

char winner(const array<char, 9> &b) {
    static const int L[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    for (auto &l : L)
        if (b[l[0]] != ' ' && b[l[0]] == b[l[1]] && b[l[1]] == b[l[2]]) return b[l[0]];
    return ' ';
}

bool full(const array<char, 9> &b) {
    for (char c : b) if (c == ' ') return false;
    return true;
}

void printBoard(const array<char, 9> &b) {
    cout << "\n";
    for (int r = 0; r < 3; r++) {
        cout << "  ";
        for (int c = 0; c < 3; c++) {
            char ch = b[r * 3 + c];
            cout << (ch == ' ' ? char('1' + r * 3 + c) : ch);
            if (c < 2) cout << " | ";
        }
        cout << "\n";
        if (r < 2) cout << " ---+---+---\n";
    }
    cout << "\n";
}

int minimax(array<char, 9> &b, int depth, bool maximizing) {
    nodeCount++;
    char w = winner(b);
    if (w == AI) return 10 - depth;
    if (w == HUMAN) return depth - 10;
    if (full(b)) return 0;
    int best = maximizing ? INT_MIN : INT_MAX;
    for (int i = 0; i < 9; i++) {
        if (b[i] != ' ') continue;
        b[i] = maximizing ? AI : HUMAN;
        int val = minimax(b, depth + 1, !maximizing);
        b[i] = ' ';
        best = maximizing ? max(best, val) : min(best, val);
    }
    return best;
}

int alphaBeta(array<char, 9> &b, int depth, int alpha, int beta, bool maximizing) {
    nodeCount++;
    char w = winner(b);
    if (w == AI) return 10 - depth;
    if (w == HUMAN) return depth - 10;
    if (full(b)) return 0;
    if (maximizing) {
        int best = INT_MIN;
        for (int i = 0; i < 9; i++) {
            if (b[i] != ' ') continue;
            b[i] = AI;
            best = max(best, alphaBeta(b, depth + 1, alpha, beta, false));
            b[i] = ' ';
            alpha = max(alpha, best);
            if (beta <= alpha) break;                  // beta cut-off
        }
        return best;
    } else {
        int best = INT_MAX;
        for (int i = 0; i < 9; i++) {
            if (b[i] != ' ') continue;
            b[i] = HUMAN;
            best = min(best, alphaBeta(b, depth + 1, alpha, beta, true));
            b[i] = ' ';
            beta = min(beta, best);
            if (beta <= alpha) break;                  // alpha cut-off
        }
        return best;
    }
}

int bestMove(array<char, 9> &b) {
    // plain minimax (only to count nodes for comparison)
    nodeCount = 0;
    for (int i = 0; i < 9; i++)
        if (b[i] == ' ') { b[i] = AI; minimax(b, 1, false); b[i] = ' '; }
    long long plainNodes = nodeCount;

    nodeCount = 0;
    int bestVal = INT_MIN, move = -1;
    for (int i = 0; i < 9; i++) {
        if (b[i] != ' ') continue;
        b[i] = AI;
        int val = alphaBeta(b, 1, INT_MIN, INT_MAX, false);
        b[i] = ' ';
        if (val > bestVal) { bestVal = val; move = i; }
    }
    cout << "AI statistics -> Minimax nodes: " << plainNodes
         << " | Alpha-Beta nodes: " << nodeCount
         << " | pruned: " << fixed << setprecision(1)
         << (plainNodes ? 100.0 * (plainNodes - nodeCount) / plainNodes : 0.0) << "%\n";
    return move;
}

int main() {
    cout << "========= TIC-TAC-TOE : Minimax + Alpha-Beta =========\n";
    cout << "You are X, computer is O. Enter 1-9 to place your mark.\n";
    char again = 'y';
    while (again == 'y' || again == 'Y') {
        array<char, 9> b; b.fill(' ');
        char first;
        cout << "\nDo you want to play first? (y/n) : "; cin >> first;
        bool humanTurn = (first == 'y' || first == 'Y');
        printBoard(b);
        while (true) {
            if (humanTurn) {
                int pos;
                cout << "Your move (1-9) : ";
                if (!(cin >> pos)) return 0;
                if (pos < 1 || pos > 9 || b[pos - 1] != ' ') { cout << "Invalid move, try again.\n"; continue; }
                b[pos - 1] = HUMAN;
            } else {
                int m = bestMove(b);
                b[m] = AI;
                cout << "Computer plays at position " << m + 1 << "\n";
            }
            printBoard(b);
            char w = winner(b);
            if (w == HUMAN) { cout << "You win!\n"; break; }
            if (w == AI)    { cout << "Computer wins!\n"; break; }
            if (full(b))    { cout << "It's a draw!\n"; break; }
            humanTurn = !humanTurn;
        }
        cout << "Play again? (y/n) : "; cin >> again;
        if (!cin) break;
    }
    cout << "Thanks for playing.\n";
    return 0;
}
