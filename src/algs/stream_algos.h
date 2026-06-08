// algs/streaming_1consistent_ksubmodular_matroid.h
// ---------------------------------------------------------------
//  Algorithm 1 (StreamMatroid): One-consistent streaming for
//      k-submodular maximization under partition matroid.
//      Swap decision uses exact re-evaluated marginals.
//
//  Algorithm 2 (StreamMatroidW): Same idea but uses cached
//      insertion-time weights W(e) for swap decisions.
// ---------------------------------------------------------------
#ifndef STREAMING_1CONSISTENT_KSUBMODULAR_MATROID_H
#define STREAMING_1CONSISTENT_KSUBMODULAR_MATROID_H

#include <vector>
#include <cstddef>
#include <limits>
#include <chrono>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <cstdint>

#include "mygraph.h"
#include "kfunctions.h"
#include "matroid.h"
#include "algs/result.h"

namespace algs {

// ======================== helpers (local to this TU) ==========================

namespace streaming_detail {

using Clock = std::chrono::high_resolution_clock;

// Build partition-level counts of supp(x) = {u : x[u] != 0}.
inline std::vector<std::size_t> supp_part_counts(
    const mygraph::tinyGraph& g,
    const ksub::Assignment& x,
    std::size_t p)
{
    std::vector<std::size_t> cnt(p, 0);
    for (std::size_t u = 0; u < g.n; ++u) {
        if (x[u] == 0) continue;
        const std::size_t pid = static_cast<std::size_t>(g.part_id[u]);
        if (pid < p) cnt[pid]++;
    }
    return cnt;
}

// Check if supp(x) ∪ {e} is independent in the partition matroid.
// Equivalently: after adding e to partition pid, is cnt[pid]+1 <= cap[pid]?
inline bool can_add(const mygraph::tinyGraph& g,
                    mygraph::node_id e,
                    const std::vector<std::size_t>& part_cnt,
                    const matroid::Cap& cap,
                    std::size_t p)
{
    const std::size_t uu = static_cast<std::size_t>(e);
    const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
    if (pid >= p) return false;
    return part_cnt[pid] + 1 <= cap[pid];
}

// FindCircuit(supp(s) ∪ {e}) \ {e}  for a partition matroid:
// = all elements in supp(s) that share the same partition as e.
inline std::vector<mygraph::node_id> find_circuit_minus_e(
    const mygraph::tinyGraph& g,
    const ksub::Assignment& x,
    mygraph::node_id e)
{
    const std::size_t uu = static_cast<std::size_t>(e);
    const std::size_t pid_e = static_cast<std::size_t>(g.part_id[uu]);

    std::vector<mygraph::node_id> C;
    for (std::size_t v = 0; v < g.n; ++v) {
        if (x[v] == 0) continue;
        if (static_cast<std::size_t>(g.part_id[v]) == pid_e)
            C.push_back(static_cast<mygraph::node_id>(v));
    }
    return C;
}

} // namespace streaming_detail

// =============================================================================
//  Algorithm 1: StreamMatroid  (one-consistent, exact marginal swaps)
// =============================================================================
//
//  NOTE: Matroid check CANNOT be moved before label search because the
//  best label i is needed in BOTH branches (add and swap).
//
//  s ← 0,  α = 2
//  for each e ∈ V (stream):
//      i ← argmax_{i∈[k]}  Δ_{e,i} f(s)
//      if  supp(s) ∪ {e} ∈ F:          // independent → just add
//          s(e) ← i
//      else:
//          C ← FindCircuit(supp(s) ∪ {e}) \ {e}
//          r ← argmin_{r∈C} f(s \ {r})
//          if Δ_{e,i} f(s \ {r})  >  α · Δ_{r,s(r)} f(s \ {r}):
//              s(r) ← 0
//              s(e) ← i
//  return s
// -----------------------------------------------------------------------------

inline Result run_stream_matroid(
    const mygraph::tinyGraph& g,
    const matroid::Cap& cap,
    double alpha = 2.0)
{
    using namespace streaming_detail;

    Result res;
    res.algo = "StreamMatroid";

    const std::size_t n = g.n;
    const std::size_t K = g.K;
    const std::size_t p = matroid::derive_p(g);

    if (cap.size() != p) {
        throw std::invalid_argument("StreamMatroid: cap.size() != derive_p(g)");
    }

    {
        std::ostringstream oss;
        oss << "PartitionMatroid(p=" << p
            << ",alpha=" << alpha << ")";
        res.constraint = oss.str();
    }

    ksub::Assignment x(n, 0);
    double f_x = 0.0;

    res.queries = 0;
    res.matroid_checks = 0;

    // Maintain partition-level support counts incrementally.
    std::vector<std::size_t> part_cnt(p, 0);

    auto t0 = Clock::now();

    // ---- stream over all elements ----
    for (std::size_t uu = 0; uu < n; ++uu) {
        const mygraph::node_id e = static_cast<mygraph::node_id>(uu);
        if (x[uu] != 0) continue; // already assigned (shouldn't happen in stream)

        // Step 1: find best label i for e
        double best_delta = -std::numeric_limits<double>::infinity();
        ksub::Label best_label = 0;

        for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl) {
            const double delta = ksub::kfunc_marginal(g, e, lbl, x, f_x);
            ++res.queries;
            if (delta > best_delta) {
                best_delta = delta;
                best_label = lbl;
            }
        }

        if (best_label == 0) continue; // no valid label (K==0?)

        // Step 2: matroid feasibility check
        ++res.matroid_checks;

        if (can_add(g, e, part_cnt, cap, p)) {
            // supp(s) ∪ {e} ∈ F  →  accept directly
            x[uu] = best_label;
            f_x += best_delta;

            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid < p) part_cnt[pid]++;
        } else {
            // Find circuit C (elements in supp(s) with same partition as e)
            const auto C = find_circuit_minus_e(g, x, e);
            if (C.empty()) continue; // shouldn't happen if cap>0

            // r ← argmin_{r∈C} f(s \ {r})
            //   equivalently: find r whose removal hurts least
            double min_f_without = std::numeric_limits<double>::infinity();
            mygraph::node_id r_best = C[0];

            for (auto r : C) {
                const std::size_t ru = static_cast<std::size_t>(r);
                const ksub::Label old_label = x[ru];

                // Temporarily remove r
                x[ru] = 0;
                const double f_without_r = ksub::kfunc_evaluate(g, x);
                ++res.queries;
                x[ru] = old_label; // restore

                if (f_without_r < min_f_without) {
                    min_f_without = f_without_r;
                    r_best = r;
                }
            }

            const std::size_t ru = static_cast<std::size_t>(r_best);
            const ksub::Label r_label = x[ru];

            // Compute swap condition:
            //   Δ_{e,i} f(s\{r})  >  α · Δ_{r,s(r)} f(s\{r})
            // First: temporarily remove r_best
            x[ru] = 0;
            const double f_minus_r = ksub::kfunc_evaluate(g, x);
            ++res.queries;

            const double delta_e_i = ksub::kfunc_marginal(g, e, best_label, x, f_minus_r);
            ++res.queries;

            const double delta_r_sr = ksub::kfunc_marginal(g, r_best, r_label, x, f_minus_r);
            ++res.queries;

            if (delta_e_i > alpha * delta_r_sr) {
                // Swap: remove r, add e
                // x[ru] is already 0 from temporary removal
                x[uu] = best_label;
                f_x = f_minus_r + delta_e_i;

                // part_cnt: r removed, e added — same partition, count unchanged
                // (since circuit = same partition, and we remove one + add one)
            } else {
                // Reject e, restore r
                x[ru] = r_label;
            }
        }
    }

    res.f_value = ksub::kfunc_evaluate(g, x);
    ++res.queries;

    res.x = std::move(x);

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    ++res.matroid_checks;

    auto t1 = Clock::now();
    res.time_sec = std::chrono::duration<double>(t1 - t0).count();

    return res;
}

// =============================================================================
//  Algorithm 2: StreamMatroidW  (one-consistent, weight-cached swaps)
// =============================================================================
//
//  NOTE: Matroid check CANNOT be moved before label search because the
//  best label i and weight w are needed in BOTH branches (add and swap).
//
//  s ← 0,  W ← 0,  α = 2
//  for each e ∈ V (stream):
//      i ← argmax_{i∈[k]}  Δ_{e,i} f(s)
//      w = Δ_{e,i} f(s)
//      if  supp(s) ∪ {e} ∈ F:
//          s(e) ← i,   W(e) = w
//      else:
//          C ← FindCircuit(supp(s) ∪ {e}) \ {e}
//          r ← argmin_{r∈C} W(r)
//          if w > α · W(r):
//              s(r) ← 0
//              s(e) ← i,   W(e) = w
//  return s
// -----------------------------------------------------------------------------

inline Result run_stream_matroid_w(
    const mygraph::tinyGraph& g,
    const matroid::Cap& cap,
    double alpha = 2.0)
{
    using namespace streaming_detail;

    Result res;
    res.algo = "StreamMatroidW";

    const std::size_t n = g.n;
    const std::size_t K = g.K;
    const std::size_t p = matroid::derive_p(g);

    if (cap.size() != p) {
        throw std::invalid_argument("StreamMatroidW: cap.size() != derive_p(g)");
    }

    {
        std::ostringstream oss;
        oss << "PartitionMatroid(p=" << p
            << ",alpha=" << alpha << ")";
        res.constraint = oss.str();
    }

    ksub::Assignment x(n, 0);
    std::vector<double> W(n, 0.0); // cached insertion-time weights
    double f_x = 0.0;

    res.queries = 0;
    res.matroid_checks = 0;

    // Maintain partition-level support counts and actual members.
    std::vector<std::size_t> part_cnt(p, 0);
    std::vector<std::vector<mygraph::node_id>> part_members(p);

    auto t0 = Clock::now();

    // ---- stream over all elements ----
    for (std::size_t uu = 0; uu < n; ++uu) {
        const mygraph::node_id e = static_cast<mygraph::node_id>(uu);
        if (x[uu] != 0) continue;

        // Step 1: find best label i for e, compute w = Δ_{e,i} f(s)
        double best_delta = -std::numeric_limits<double>::infinity();
        ksub::Label best_label = 0;

        for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl) {
            const double delta = ksub::kfunc_marginal(g, e, lbl, x, f_x);
            ++res.queries;
            if (delta > best_delta) {
                best_delta = delta;
                best_label = lbl;
            }
        }

        if (best_label == 0) continue;

        const double w = best_delta; // Δ_{e,i} f(s)

        // Step 2: matroid feasibility check
        ++res.matroid_checks;
        const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);

        if (can_add(g, e, part_cnt, cap, p)) {
            // supp(s) ∪ {e} ∈ F  →  accept directly
            x[uu] = best_label;
            W[uu] = w;
            f_x += best_delta;

            if (pid < p) {
                part_cnt[pid]++;
                part_members[pid].push_back(e);
            }
        } else {
            // Find circuit C (elements in same partition as e)
            if (pid >= p) continue;
            const auto& C = part_members[pid];
            if (C.empty()) continue;

            // r ← argmin_{r∈C} W(r)
            double min_w = std::numeric_limits<double>::infinity();
            mygraph::node_id r_best = C[0];

            for (auto r : C) {
                const std::size_t ru = static_cast<std::size_t>(r);
                if (W[ru] < min_w) {
                    min_w = W[ru];
                    r_best = r;
                }
            }

            // Swap condition:  w > α · W(r)
            if (w > alpha * min_w) {
                const std::size_t ru = static_cast<std::size_t>(r_best);

                // Remove r_best
                x[ru] = 0;
                W[ru] = 0.0;

                // Add e
                x[uu] = best_label;
                W[uu] = w;

                // Recompute f_x
                f_x = ksub::kfunc_evaluate(g, x);
                ++res.queries;

                // Cập nhật part_members: thay thế r_best bằng e
                auto& members = part_members[pid];
                for (auto& m : members) {
                    if (m == r_best) {
                        m = e;
                        break;
                    }
                }
            }
        }
    }

    res.f_value = ksub::kfunc_evaluate(g, x);
    ++res.queries;

    res.x = std::move(x);

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    ++res.matroid_checks;

    auto t1 = Clock::now();
    res.time_sec = std::chrono::duration<double>(t1 - t0).count();

    return res;
}

// =============================================================================
//  Algorithm 4: StreamRandom
//    (randomized streaming with 50% acceptance probability)
// =============================================================================
//
//  Ý tưởng: duyệt stream, chọn ngẫu nhiên một label i ∈ [1..k].
//  Nếu phần tử thỏa ràng buộc matroid thì thêm vào lời giải với xác suất 50%.
//
//  Pseudocode:
//    s ← 0
//    for each e ∈ V (stream):
//        i ← random label in [1..k]
//        if supp(s) ∪ {e} ∈ F:
//            with probability 0.5:
//                s(e) ← i
//    return s
// -----------------------------------------------------------------------------

inline Result run_stream_random(
    const mygraph::tinyGraph& g,
    const matroid::Cap& cap,
    std::uint64_t seed = 42ULL)
{
    using namespace streaming_detail;

    Result res;
    res.algo = "StreamRandom";

    const std::size_t n = g.n;
    const std::size_t K = g.K;
    const std::size_t p = matroid::derive_p(g);

    if (cap.size() != p) {
        throw std::invalid_argument("StreamRandom: cap.size() != derive_p(g)");
    }

    {
        std::ostringstream oss;
        oss << "PartitionMatroid(p=" << p
            << ",seed=" << seed << ")";
        res.constraint = oss.str();
    }

    ksub::Assignment x(n, 0);
    double f_x = 0.0;

    res.queries = 0;
    res.matroid_checks = 0;

    std::vector<std::size_t> part_cnt(p, 0);

    // Random number generator for 50% coin flip
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    auto t0 = Clock::now();

    for (std::size_t uu = 0; uu < n; ++uu) {
        const mygraph::node_id e = static_cast<mygraph::node_id>(uu);
        if (x[uu] != 0) continue;

        // Check matroid feasibility FIRST
        ++res.matroid_checks;
        if (!can_add(g, e, part_cnt, cap, p)) continue;

        // Coin flip: proceed with probability 50%
        if (coin(rng) < 0.5) {
            // Pick a random label among 1..K
            std::uniform_int_distribution<int> dist_k(1, static_cast<int>(K));
            ksub::Label rand_label = static_cast<ksub::Label>(dist_k(rng));
            
            const double rand_delta = ksub::kfunc_marginal(g, e, rand_label, x, f_x);
            ++res.queries;

            x[uu] = rand_label;
            f_x += rand_delta;

            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid < p) part_cnt[pid]++;
        }
    }

    res.f_value = ksub::kfunc_evaluate(g, x);
    ++res.queries;

    res.x = std::move(x);

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    ++res.matroid_checks;

    auto t1 = Clock::now();
    res.time_sec = std::chrono::duration<double>(t1 - t0).count();

    return res;
}

// =============================================================================
//  Algorithm 5: StreamGreedy
//    (streaming greedy without threshold — accept if matroid-feasible)
// =============================================================================
//
//  Ý tưởng: duyệt stream, nếu phần tử thỏa ràng buộc matroid thì luôn
//  thêm vào tập lời giải. Label được chọn là label có marginal gain
//  lớn nhất.
//
//  Tối ưu: kiểm tra ràng buộc matroid TRƯỚC khi tìm nhãn tốt nhất.
//  Nếu partition đã đầy → skip ngay, tiết kiệm K lần tính marginal gain.
//
//  Pseudocode:
//    s ← 0
//    for each e ∈ V (stream):
//        if supp(s) ∪ {e} ∉ F:  continue     // matroid check first
//        i ← argmax_{i∈[k]}  Δ_{e,i} f(s)
//        s(e) ← i
//    return s
// -----------------------------------------------------------------------------

inline Result run_stream_greedy(
    const mygraph::tinyGraph& g,
    const matroid::Cap& cap)
{
    using namespace streaming_detail;

    Result res;
    res.algo = "StreamGreedy";

    const std::size_t n = g.n;
    const std::size_t K = g.K;
    const std::size_t p = matroid::derive_p(g);

    if (cap.size() != p) {
        throw std::invalid_argument("StreamGreedy: cap.size() != derive_p(g)");
    }

    {
        std::ostringstream oss;
        oss << "PartitionMatroid(p=" << p << ")";
        res.constraint = oss.str();
    }

    ksub::Assignment x(n, 0);
    double f_x = 0.0;

    res.queries = 0;
    res.matroid_checks = 0;

    std::vector<std::size_t> part_cnt(p, 0);

    auto t0 = Clock::now();

    for (std::size_t uu = 0; uu < n; ++uu) {
        const mygraph::node_id e = static_cast<mygraph::node_id>(uu);
        if (x[uu] != 0) continue;

        // Check matroid feasibility FIRST — skip if partition full
        ++res.matroid_checks;
        if (!can_add(g, e, part_cnt, cap, p)) continue;

        // Find best label (only computed when matroid-feasible)
        double best_delta = -std::numeric_limits<double>::infinity();
        ksub::Label best_label = 0;

        for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl) {
            const double delta = ksub::kfunc_marginal(g, e, lbl, x, f_x);
            ++res.queries;
            if (delta > best_delta) {
                best_delta = delta;
                best_label = lbl;
            }
        }

        if (best_label == 0) continue;

        // Accept: matroid-feasible → add directly
        x[uu] = best_label;
        f_x += best_delta;

        const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
        if (pid < p) part_cnt[pid]++;
    }

    res.f_value = ksub::kfunc_evaluate(g, x);
    ++res.queries;

    res.x = std::move(x);

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    ++res.matroid_checks;

    auto t1 = Clock::now();
    res.time_sec = std::chrono::duration<double>(t1 - t0).count();

    return res;
}

} // namespace algs

#endif // STREAMING_1CONSISTENT_KSUBMODULAR_MATROID_H
