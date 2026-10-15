/*
 * Assignment 4A : Customer Churn Prediction - Supervised, Semi-supervised, Ensemble
 *
 * Dataset  : 1500 synthetic customer records (generated inside the program so that
 *            the code is self-contained). Features:
 *            tenure(months), monthly charges, support calls, contract type, fibre internet.
 *            Target: churn (1 = customer leaves, 0 = stays)
 * Models   :
 *   Supervised      : Logistic Regression, k-Nearest Neighbours, Decision Tree
 *   Semi-supervised : Self-training Logistic Regression (only 10% of training data labelled)
 *   Ensemble        : Bagging of trees (Random-Forest style), Majority-Vote of LR+KNN+DT
 * Evaluation : Accuracy, Precision, Recall, F1-score and confusion matrix on a hold-out test set.
 *
 * Compile : g++ -std=c++17 -O2 Assignment4A.cpp -o a4a
 */
#include <bits/stdc++.h>
using namespace std;

typedef vector<double> Vec;
struct Data { vector<Vec> X; vector<int> y; };

mt19937 rng(12345);

// ------------------------------------------------------------------ dataset
Data generateData(int n) {
    Data d;
    uniform_real_distribution<double> U(0, 1);
    normal_distribution<double> noise(0, 1.0);
    for (int i = 0; i < n; i++) {
        double tenure = 1 + U(rng) * 71;
        double monthly = 20 + U(rng) * 100;
        double calls = floor(U(rng) * 9);
        double contract = floor(U(rng) * 3);
        double fibre = U(rng) < 0.5 ? 1 : 0;
        double z = -1.2 - 0.05 * tenure + 0.03 * monthly + 0.35 * calls - 0.8 * contract + 0.5 * fibre + noise(rng);
        d.X.push_back({tenure, monthly, calls, contract, fibre});
        d.y.push_back(z > 0 ? 1 : 0);
    }
    return d;
}

void standardize(Data &train, vector<Vec> &other1, vector<Vec> &other2) {
    int f = train.X[0].size();
    for (int j = 0; j < f; j++) {
        double mu = 0, sd = 0;
        for (auto &x : train.X) mu += x[j];
        mu /= train.X.size();
        for (auto &x : train.X) sd += (x[j] - mu) * (x[j] - mu);
        sd = sqrt(sd / train.X.size()); if (sd < 1e-9) sd = 1;
        for (auto &x : train.X) x[j] = (x[j] - mu) / sd;
        for (auto &x : other1) x[j] = (x[j] - mu) / sd;
        for (auto &x : other2) x[j] = (x[j] - mu) / sd;
    }
}

// ------------------------------------------------------------------ classifier interface
struct Classifier {
    virtual void fit(const vector<Vec> &X, const vector<int> &y) = 0;
    virtual double proba(const Vec &x) const = 0;          // P(class = 1)
    int predict(const Vec &x) const { return proba(x) >= 0.5 ? 1 : 0; }
    virtual ~Classifier() {}
};

// ------------------------------------------------------------------ Logistic Regression
struct LogisticRegression : Classifier {
    Vec w; double b = 0; int epochs = 600; double lr = 0.1, l2 = 1e-3;
    static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
    void fit(const vector<Vec> &X, const vector<int> &y) override {
        int n = X.size(), f = X[0].size();
        w.assign(f, 0); b = 0;
        for (int e = 0; e < epochs; e++) {
            Vec gw(f, 0); double gb = 0;
            for (int i = 0; i < n; i++) {
                double err = sigmoid(inner_product(w.begin(), w.end(), X[i].begin(), b)) - y[i];
                for (int j = 0; j < f; j++) gw[j] += err * X[i][j];
                gb += err;
            }
            for (int j = 0; j < f; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
            b -= lr * gb / n;
        }
    }
    double proba(const Vec &x) const override {
        return sigmoid(inner_product(w.begin(), w.end(), x.begin(), b));
    }
};

// ------------------------------------------------------------------ k-NN
struct KNN : Classifier {
    vector<Vec> X; vector<int> y; int k;
    KNN(int k = 7) : k(k) {}
    void fit(const vector<Vec> &Xt, const vector<int> &yt) override { X = Xt; y = yt; }
    double proba(const Vec &q) const override {
        vector<pair<double, int>> d;
        for (size_t i = 0; i < X.size(); i++) {
            double s = 0;
            for (size_t j = 0; j < q.size(); j++) s += (X[i][j] - q[j]) * (X[i][j] - q[j]);
            d.push_back({s, y[i]});
        }
        int kk = min<int>(k, d.size());
        partial_sort(d.begin(), d.begin() + kk, d.end());
        int ones = 0;
        for (int i = 0; i < kk; i++) ones += d[i].second;
        return (double)ones / kk;
    }
};

// ------------------------------------------------------------------ Decision Tree (Gini)
struct DecisionTree : Classifier {
    struct Node { int feat = -1; double thr = 0, prob = 0; int l = -1, r = -1; };
    vector<Node> nodes; int maxDepth, minSamples, mtry;
    DecisionTree(int depth = 5, int minS = 5, int mtry = 0) : maxDepth(depth), minSamples(minS), mtry(mtry) {}

    static double gini(int ones, int total) {
        if (total == 0) return 0;
        double p = (double)ones / total; return 2 * p * (1 - p);
    }
    int build(const vector<Vec> &X, const vector<int> &y, vector<int> idx, int depth) {
        Node nd; int ones = 0;
        for (int i : idx) ones += y[i];
        nd.prob = (double)ones / idx.size();
        int id = nodes.size(); nodes.push_back(nd);
        if (depth >= maxDepth || (int)idx.size() < minSamples || ones == 0 || ones == (int)idx.size()) return id;

        int f = X[0].size();
        vector<int> feats(f); iota(feats.begin(), feats.end(), 0);
        if (mtry > 0 && mtry < f) { shuffle(feats.begin(), feats.end(), rng); feats.resize(mtry); }

        double bestScore = gini(ones, idx.size()) - 1e-9; int bf = -1; double bt = 0;
        for (int j : feats) {
            vector<pair<double,int>> v;
            for (int i : idx) v.push_back({X[i][j], y[i]});
            sort(v.begin(), v.end());
            int total = v.size(), leftOnes = 0;
            for (int s = 0; s + 1 < total; s++) {
                leftOnes += v[s].second;
                if (v[s].first == v[s + 1].first) continue;
                int ln = s + 1, rn = total - ln;
                double score = (ln * gini(leftOnes, ln) + rn * gini(ones - leftOnes, rn)) / total;
                if (score < bestScore) { bestScore = score; bf = j; bt = (v[s].first + v[s + 1].first) / 2; }
            }
        }
        if (bf == -1) return id;
        vector<int> L, R;
        for (int i : idx) (X[i][bf] <= bt ? L : R).push_back(i);
        if (L.empty() || R.empty()) return id;
        nodes[id].feat = bf; nodes[id].thr = bt;
        int l = build(X, y, L, depth + 1);
        int r = build(X, y, R, depth + 1);
        nodes[id].l = l; nodes[id].r = r;
        return id;
    }
    void fit(const vector<Vec> &X, const vector<int> &y) override {
        nodes.clear();
        vector<int> idx(X.size()); iota(idx.begin(), idx.end(), 0);
        build(X, y, idx, 0);
    }
    double proba(const Vec &x) const override {
        int cur = 0;
        while (nodes[cur].feat != -1) cur = x[nodes[cur].feat] <= nodes[cur].thr ? nodes[cur].l : nodes[cur].r;
        return nodes[cur].prob;
    }
};

// ------------------------------------------------------------------ Ensembles
struct Bagging : Classifier {
    vector<DecisionTree> trees; int B;
    Bagging(int B = 25) : B(B) {}
    void fit(const vector<Vec> &X, const vector<int> &y) override {
        trees.clear();
        int n = X.size();
        for (int b = 0; b < B; b++) {
            vector<Vec> bx; vector<int> by;
            for (int i = 0; i < n; i++) { int k = rng() % n; bx.push_back(X[k]); by.push_back(y[k]); }
            DecisionTree t(6, 5, 3);               // random subset of 3 features at every split
            t.fit(bx, by);
            trees.push_back(t);
        }
    }
    double proba(const Vec &x) const override {
        double s = 0; for (auto &t : trees) s += t.proba(x);
        return s / trees.size();
    }
};

struct MajorityVote : Classifier {
    vector<Classifier *> models;
    void fit(const vector<Vec> &X, const vector<int> &y) override { for (auto m : models) m->fit(X, y); }
    double proba(const Vec &x) const override {
        int votes = 0; for (auto m : models) votes += m->predict(x);
        return (double)votes / models.size();
    }
};

// ------------------------------------------------------------------ Self-training
struct SelfTraining {
    int rounds = 0, pseudoAdded = 0, pseudoCorrect = 0;
    LogisticRegression model;
    void fit(const vector<Vec> &XL, const vector<int> &yL, const vector<Vec> &XU, const vector<int> &trueU) {
        vector<Vec> X = XL; vector<int> y = yL;
        vector<bool> used(XU.size(), false);
        for (int r = 0; r < 10; r++) {
            model.fit(X, y);
            int added = 0;
            for (size_t i = 0; i < XU.size(); i++) {
                if (used[i]) continue;
                double p = model.proba(XU[i]);
                if (p > 0.90 || p < 0.10) {                 // confident prediction -> pseudo label
                    int lab = p > 0.5 ? 1 : 0;
                    X.push_back(XU[i]); y.push_back(lab); used[i] = true;
                    added++; pseudoAdded++; if (lab == trueU[i]) pseudoCorrect++;
                }
            }
            rounds = r + 1;
            if (added == 0) break;
        }
        model.fit(X, y);
    }
};

// ------------------------------------------------------------------ evaluation
struct Metrics { int tp = 0, fp = 0, tn = 0, fn = 0; double acc, prec, rec, f1; };

Metrics evaluate(const function<int(const Vec &)> &pred, const vector<Vec> &X, const vector<int> &y) {
    Metrics m;
    for (size_t i = 0; i < X.size(); i++) {
        int p = pred(X[i]);
        if (p == 1 && y[i] == 1) m.tp++; else if (p == 1 && y[i] == 0) m.fp++;
        else if (p == 0 && y[i] == 0) m.tn++; else m.fn++;
    }
    int n = X.size();
    m.acc = (double)(m.tp + m.tn) / n;
    m.prec = (m.tp + m.fp) ? (double)m.tp / (m.tp + m.fp) : 0;
    m.rec = (m.tp + m.fn) ? (double)m.tp / (m.tp + m.fn) : 0;
    m.f1 = (m.prec + m.rec) ? 2 * m.prec * m.rec / (m.prec + m.rec) : 0;
    return m;
}

void row(const string &name, const Metrics &m) {
    cout << left << setw(38) << name << right << fixed << setprecision(3)
         << setw(9) << m.acc << setw(11) << m.prec << setw(9) << m.rec << setw(9) << m.f1
         << setw(6) << m.tp << setw(5) << m.fp << setw(5) << m.tn << setw(5) << m.fn << "\n";
}

int main() {
    cout << "======= CUSTOMER CHURN PREDICTION =======\n";
    Data all = generateData(1500);
    int churn = accumulate(all.y.begin(), all.y.end(), 0);
    cout << "Dataset : " << all.X.size() << " records, 5 features, churn rate = "
         << fixed << setprecision(1) << 100.0 * churn / all.y.size() << "%\n";

    // shuffle and split 70/30
    vector<int> idx(all.X.size()); iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    int ntrain = 0.7 * idx.size();
    Data train, test;
    for (int i = 0; i < (int)idx.size(); i++) {
        Data &d = i < ntrain ? train : test;
        d.X.push_back(all.X[idx[i]]); d.y.push_back(all.y[idx[i]]);
    }
    vector<Vec> dummy;
    standardize(train, test.X, dummy);
    cout << "Train = " << train.X.size() << ", Test = " << test.X.size() << "\n";

    // labelled / unlabelled split for semi-supervised learning (10% labelled)
    int nl = train.X.size() / 10;
    vector<Vec> XL(train.X.begin(), train.X.begin() + nl), XU(train.X.begin() + nl, train.X.end());
    vector<int> yL(train.y.begin(), train.y.begin() + nl), yU(train.y.begin() + nl, train.y.end());
    cout << "Semi-supervised setting : " << nl << " labelled, " << XU.size() << " unlabelled samples\n\n";

    cout << left << setw(38) << "Model" << right << setw(9) << "Accuracy" << setw(11) << "Precision"
         << setw(9) << "Recall" << setw(9) << "F1" << setw(6) << "TP" << setw(5) << "FP"
         << setw(5) << "TN" << setw(5) << "FN" << "\n" << string(96, '-') << "\n";

    // ---- supervised (fully labelled training data)
    LogisticRegression lr; lr.fit(train.X, train.y);
    KNN knn(7);            knn.fit(train.X, train.y);
    DecisionTree dt(5, 5); dt.fit(train.X, train.y);
    Metrics mLR  = evaluate([&](const Vec &x) { return lr.predict(x); },  test.X, test.y);
    Metrics mKNN = evaluate([&](const Vec &x) { return knn.predict(x); }, test.X, test.y);
    Metrics mDT  = evaluate([&](const Vec &x) { return dt.predict(x); },  test.X, test.y);
    row("Supervised: Logistic Regression", mLR);
    row("Supervised: k-NN (k=7)", mKNN);
    row("Supervised: Decision Tree", mDT);

    // ---- semi-supervised
    LogisticRegression onlyL; onlyL.fit(XL, yL);
    Metrics mOL = evaluate([&](const Vec &x) { return onlyL.predict(x); }, test.X, test.y);
    row("LR on 10% labelled data only", mOL);
    SelfTraining st; st.fit(XL, yL, XU, yU);
    Metrics mST = evaluate([&](const Vec &x) { return st.model.predict(x); }, test.X, test.y);
    row("Semi-supervised: Self-training LR", mST);

    // ---- ensembles
    Bagging bag(25); bag.fit(train.X, train.y);
    LogisticRegression l2; KNN k2(7); DecisionTree d2(5, 5);
    MajorityVote mv; mv.models = {&l2, &k2, &d2}; mv.fit(train.X, train.y);
    Metrics mBag = evaluate([&](const Vec &x) { return bag.predict(x); }, test.X, test.y);
    Metrics mMV  = evaluate([&](const Vec &x) { return mv.predict(x); },  test.X, test.y);
    row("Ensemble: Bagged Trees (25)", mBag);
    row("Ensemble: Majority Vote (LR+KNN+DT)", mMV);

    cout << "\nSelf-training details : " << st.rounds << " rounds, " << st.pseudoAdded
         << " pseudo-labels added, pseudo-label accuracy = " << setprecision(1)
         << (st.pseudoAdded ? 100.0 * st.pseudoCorrect / st.pseudoAdded : 0.0) << "%\n";

    vector<pair<string, Metrics>> all_models = {
        {"Logistic Regression", mLR}, {"k-NN", mKNN}, {"Decision Tree", mDT},
        {"Self-training LR", mST}, {"Bagged Trees", mBag}, {"Majority Vote", mMV}};
    auto best = max_element(all_models.begin(), all_models.end(),
                            [](auto &a, auto &b) { return a.second.f1 < b.second.f1; });
    cout << "Best model by F1-score : " << best->first << " (F1 = " << setprecision(3) << best->second.f1 << ")\n";
    cout << "\nObservations:\n";
    cout << " * Self-training F1 = " << setprecision(3) << mST.f1 << " vs " << mOL.f1
         << " for LR trained on the 10% labelled data only (full supervision: " << mLR.f1 << ").\n";
    cout << " * Bagged trees F1 = " << mBag.f1 << " vs single decision tree F1 = " << mDT.f1
         << (mBag.f1 >= mDT.f1 ? " -> bagging reduces the variance of one tree.\n"
                               : " -> no gain from bagging on this split.\n");
    cout << " * Majority-vote F1 = " << mMV.f1 << " (best single base model F1 = "
         << max(mLR.f1, max(mKNN.f1, mDT.f1)) << ").\n";
    return 0;
}
