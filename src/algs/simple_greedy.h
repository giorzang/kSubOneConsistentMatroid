// algs/simple_greedy.h
#ifndef SIMPLE_GREEDY_H
#define SIMPLE_GREEDY_H

#include <vector>
#include <limits>
#include <chrono>
#include <string>
#include <sstream>
#include <stdexcept>

#include "mygraph.h"
#include "kfunctions.h"
#include "matroid.h"
#include "algs/result.h"

namespace algs {

using Clock = std::chrono::high_resolution_clock;


inline Result run_simple_greedy(const mygraph::tinyGraph &g,
                                const matroid::Cap &cap)
{
    Result res;
    res.algo = "SimpleGreedy";

    const std::size_t n = g.n;
    const std::size_t K = g.K;

    const std::size_t p = matroid::derive_p(g);
    if (cap.size() != p) {
        throw std::invalid_argument("SimpleGreedy: cap.size() != derive_p(g)");
    }

    {
        std::size_t sum_cap = 0;
        for (auto c : cap) sum_cap += c;
        std::ostringstream oss;
        oss << "PartitionMatroid(p=" << p << ",sum_cap=" << sum_cap << ")";
        res.constraint = oss.str();
    }

    res.x.assign(n, 0);
    ksub::Assignment x = res.x;

    double f_x = 0.0;
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    res.queries = 0;
    res.matroid_checks = 0;

    std::vector<std::size_t> cnt(p, 0);

    auto t0 = Clock::now();

    while (true) {
        double best_delta = NEG_INF;
        mygraph::node_id best_u = static_cast<mygraph::node_id>(-1);
        ksub::Label best_label = 0;

        for (std::size_t uu = 0; uu < n; ++uu) {
            if (x[uu] != 0) continue;

            ++res.matroid_checks;
            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid >= p) continue;
            if (cnt[pid] >= cap[pid]) continue;

            const mygraph::node_id u = static_cast<mygraph::node_id>(uu);

            for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl) {
                const double delta = ksub::kfunc_marginal(g, u, lbl, x, f_x);
                ++res.queries;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_u = u;
                    best_label = lbl;
                }
            }
        }

        if (best_u == static_cast<mygraph::node_id>(-1)) break;

        const std::size_t bu = static_cast<std::size_t>(best_u);
        const std::size_t pid = static_cast<std::size_t>(g.part_id[bu]);
        if (pid >= p || cnt[pid] >= cap[pid]) break; // defensive

        x[best_u] = best_label;
        f_x += best_delta;
        cnt[pid] += 1;
    }

    res.f_value = ksub::kfunc_evaluate(g, x);
    ++res.queries;

    res.x.swap(x);

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    ++res.matroid_checks;

    res.total_error = res.fair_error + res.matroid_error;

    auto t1 = Clock::now();
    res.time_sec = std::chrono::duration<double>(t1 - t0).count();

    return res;
}

} // namespace algs

#endif // SIMPLE_GREEDY_H
