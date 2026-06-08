// algs/k_greedy_ts.h
#ifndef K_GREEDY_TS_H
#define K_GREEDY_TS_H

#include <vector>
#include <limits>
#include <chrono>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <cstdint>

#include "mygraph.h"
#include "kfunctions.h"
#include "matroid.h"
#include "algs/result.h"
#include "algs/runtime_seed.h"

namespace algs {

using Clock = std::chrono::high_resolution_clock;

inline void build_random_max_independent_set_V1_ts(
    const mygraph::tinyGraph &g,
    const matroid::Cap &cap,
    std::vector<mygraph::node_id> &V1_out,
    std::vector<std::uint8_t> &inV1_out 
) {
    const std::size_t n = g.n;
    const std::size_t p = matroid::derive_p(g);

    if (cap.size() != p) {
        throw std::invalid_argument("kGreedyTS: cap.size() != derive_p(g)");
    }

    V1_out.clear();
    inV1_out.assign(n, 0);

    std::vector<std::vector<mygraph::node_id>> bucket(p);
    for (std::size_t u = 0; u < n; ++u) {
        const std::size_t pid = static_cast<std::size_t>(g.part_id[u]);
        if (pid >= p) continue;
        bucket[pid].push_back(static_cast<mygraph::node_id>(u));
    }

    std::mt19937_64 rng(algs::runtime_seed64());

    for (std::size_t pid = 0; pid < p; ++pid) {
        auto &vec = bucket[pid];
        if (vec.empty()) continue;

        std::shuffle(vec.begin(), vec.end(), rng);

        const std::size_t take = std::min<std::size_t>(cap[pid], vec.size());
        for (std::size_t t = 0; t < take; ++t) {
            const mygraph::node_id u = vec[t];
            const std::size_t uu = static_cast<std::size_t>(u);
            if (uu < n && !inV1_out[uu]) {
                inV1_out[uu] = 1;
                V1_out.push_back(u);
            }
        }
    }
}


inline Result run_k_greedy_ts(
    const mygraph::tinyGraph &g,
    const matroid::Cap &cap,
    const std::vector<std::size_t> &u_upper 
) {
    Result res;
    res.algo = "kGreedyTS";

    const std::size_t n = g.n;
    const std::size_t K = g.K;

    if (u_upper.size() != K) {
        throw std::invalid_argument("kGreedyTS: u_upper.size() != g.K");
    }

    std::size_t B = 0;
    for (std::size_t i = 0; i < K; ++i) B += u_upper[i];

    std::vector<mygraph::node_id> V1;
    std::vector<std::uint8_t> inV1;
    build_random_max_independent_set_V1_ts(g, cap, V1, inV1);

    {
        const std::size_t p = matroid::derive_p(g);
        std::size_t sum_cap = 0;
        for (auto c : cap) sum_cap += c;

        std::ostringstream oss;
        oss << "PreselectV1_PartitionMatroid(p=" << p
            << ",sum_cap=" << sum_cap
            << ",|V1|=" << V1.size() << ")"
            << "+Cardinality(B=" << B << ")";
        res.constraint = oss.str();
    }

    res.x.assign(n, 0);
    ksub::Assignment x = res.x;

    res.queries = 0;
    res.matroid_checks = 0;

    double f_x = 0.0;
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    std::size_t chosen = 0;

    auto t0 = Clock::now();

    while (chosen < B) {
        double best_delta = NEG_INF;
        mygraph::node_id best_u = static_cast<mygraph::node_id>(-1);
        ksub::Label best_lbl = 0;

        for (mygraph::node_id u : V1) {
            const std::size_t uu = static_cast<std::size_t>(u);
            if (uu >= n) continue;
            if (x[uu] != 0) continue; // already assigned

            for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl) {
                const double delta = ksub::kfunc_marginal(g, u, lbl, x, f_x);
                ++res.queries;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_u = u;
                    best_lbl = lbl;
                }
            }
        }

        if (best_u == static_cast<mygraph::node_id>(-1)) break;

        const std::size_t bu = static_cast<std::size_t>(best_u);
        if (bu >= n) break;
        if (!inV1[bu]) break;
        if (best_lbl == 0 || static_cast<std::size_t>(best_lbl) > K) break;

        x[bu] = best_lbl;
        f_x += best_delta;
        chosen += 1;
    }

    res.f_value = ksub::kfunc_evaluate(g, x);
    ++res.queries;

    res.x.swap(x);

    res.fair_error = 0.0;

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    ++res.matroid_checks;

    res.total_error = res.fair_error + res.matroid_error;

    auto t1 = Clock::now();
    res.time_sec = std::chrono::duration<double>(t1 - t0).count();

    return res;
}

} 

#endif // K_GREEDY_TS_H
