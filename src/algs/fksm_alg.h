#ifndef ALGS_FKSM_ALG_H
#define ALGS_FKSM_ALG_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <chrono>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <queue>

#include "mygraph.h"
#include "kfunctions.h"
#include "algs/result.h"
#include "matroid.h"

namespace algs
{

    using Clock = std::chrono::high_resolution_clock;

    inline std::size_t sum_cap(const matroid::Cap &cap)
    {
        std::size_t s = 0;
        for (auto v : cap)
            s += v;
        return s;
    }

    inline std::size_t sum_u_limits(const std::vector<std::size_t> &u_limits, std::size_t K)
    {
        std::size_t s = 0;
        for (std::size_t i = 1; i <= K; ++i)
            s += u_limits[i];
        return s;
    }

    inline std::vector<std::size_t> label_counts(const ksub::Assignment &x, std::size_t K)
    {
        std::vector<std::size_t> cnt(K + 1, 0);
        for (std::size_t u = 0; u < x.size(); ++u)
        {
            const std::size_t lab = static_cast<std::size_t>(x[u]);
            if (lab >= 1 && lab <= K)
                cnt[lab]++;
        }
        return cnt;
    }

    inline std::vector<mygraph::node_id> make_V(const mygraph::tinyGraph &g)
    {
        std::vector<mygraph::node_id> V;
        V.reserve(g.n);
        for (std::size_t uu = 0; uu < g.n; ++uu)
            V.push_back(static_cast<mygraph::node_id>(uu));
        return V;
    }

    inline std::vector<mygraph::node_id> build_max_independent_set_L(
        const mygraph::tinyGraph &g,
        const matroid::Cap &cap)
    {
        const std::size_t n = g.n;
        const std::size_t p = matroid::derive_p(g);

        std::vector<std::size_t> used(p, 0);
        std::vector<mygraph::node_id> L;
        L.reserve(sum_cap(cap));

        for (std::size_t uu = 0; uu < n; ++uu)
        {
            const mygraph::node_id u = static_cast<mygraph::node_id>(uu);
            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid >= p)
                continue;
            if (used[pid] < cap[pid])
            {
                used[pid] += 1;
                L.push_back(u);
            }
        }
        return L;
    }

    inline std::vector<std::size_t> fixed_count_by_part(
        const mygraph::tinyGraph &g,
        const std::vector<mygraph::node_id> &fixed_set)
    {
        const std::size_t p = matroid::derive_p(g);
        std::vector<std::size_t> cnt(p, 0);
        for (auto u : fixed_set)
        {
            const std::size_t uu = static_cast<std::size_t>(u);
            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid < p)
                cnt[pid] += 1;
        }
        return cnt;
    }

    inline std::vector<std::uint8_t> fixed_mask_from_set(
        const mygraph::tinyGraph &g,
        const std::vector<mygraph::node_id> &fixed_set)
    {
        std::vector<std::uint8_t> mask(g.n, 0);
        for (auto u : fixed_set)
        {
            const std::size_t uu = static_cast<std::size_t>(u);
            if (uu < g.n)
                mask[uu] = 1;
        }
        return mask;
    }

    inline bool matroid_can_add_with_fixed(
        const mygraph::tinyGraph &g,
        const ksub::Assignment &x,
        mygraph::node_id e,
        const matroid::Cap &cap,
        const std::vector<std::size_t> &fixed_count,
        const std::vector<std::size_t> &sel_count)
    {
        const std::size_t uu = static_cast<std::size_t>(e);
        if (uu >= g.n)
            return false;
        if (x[uu] != 0)
            return false;

        const std::size_t p = fixed_count.size();
        const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
        if (pid >= p)
            return false;

        const std::size_t have = sel_count[pid] + fixed_count[pid];
        return have + 1 <= cap[pid];
    }

    inline bool matroid_can_add_plain(
        const mygraph::tinyGraph &g,
        const ksub::Assignment &x,
        mygraph::node_id e,
        const matroid::Cap &cap,
        const std::vector<std::size_t> &sel_count)
    {
        const std::size_t uu = static_cast<std::size_t>(e);
        if (uu >= g.n)
            return false;
        if (x[uu] != 0)
            return false;

        const std::size_t p = sel_count.size();
        const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
        if (pid >= p)
            return false;

        return sel_count[pid] + 1 <= cap[pid];
    }

    struct AUFkM_Output
    {
        ksub::Assignment x;
        std::vector<ksub::Label> I;
        double f_value = 0.0;
        std::size_t queries = 0;
        std::size_t matroid_checks = 0;
    };

    inline std::vector<mygraph::node_id> approx_alg3_single_label_partition(
        const mygraph::tinyGraph &g,
        const matroid::Cap &cap,
        const std::vector<std::size_t> &fixed_count,
        const std::vector<std::uint8_t> *fixed_mask,
        const ksub::Assignment &x_base,
        double f_base,
        ksub::Label lbl,
        std::size_t c,
        double /*eps*/,
        std::size_t &queries_io,
        std::size_t &matroid_checks_io)
    {
        std::vector<mygraph::node_id> S;
        if (c == 0)
            return S;

        const std::size_t p = matroid::derive_p(g);
        if (cap.size() != p)
            throw std::invalid_argument("Alg3(lazy-greedy): cap size != p.");
        if (fixed_count.size() != p)
            throw std::invalid_argument("Alg3(lazy-greedy): fixed_count size != p.");
        if (fixed_mask && fixed_mask->size() != g.n)
            throw std::invalid_argument("Alg3(lazy-greedy): fixed_mask size != n.");

        std::vector<std::size_t> sel_count(p, 0);
        for (std::size_t uu = 0; uu < g.n; ++uu)
        {
            if (x_base[uu] == 0)
                continue;
            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid < p)
                sel_count[pid] += 1;
        }

        ksub::Assignment x = x_base;
        double f_x = f_base;
        std::vector<std::size_t> sel2 = sel_count;

        struct Cand
        {
            double delta;
            mygraph::node_id e;
            std::uint32_t ver;
        };
        struct Cmp
        {
            bool operator()(const Cand &a, const Cand &b) const
            {
                return a.delta < b.delta; // max-heap
            }
        };

        std::priority_queue<Cand, std::vector<Cand>, Cmp> pq;
        std::uint32_t ver = 1;

        for (std::size_t uu = 0; uu < g.n; ++uu)
        {
            if (x_base[uu] != 0)
                continue;
            if (fixed_mask && (*fixed_mask)[uu])
                continue;

            const mygraph::node_id e = static_cast<mygraph::node_id>(uu);

            matroid_checks_io++;
            if (!matroid_can_add_with_fixed(g, x_base, e, cap, fixed_count, sel_count))
                continue;

            const double delta = ksub::kfunc_marginal(g, e, lbl, x_base, f_base);
            queries_io++;

            pq.push(Cand{delta, e, ver});
        }

        while (S.size() < c && !pq.empty())
        {
            Cand top = pq.top();
            pq.pop();

            const std::size_t uu = static_cast<std::size_t>(top.e);
            if (uu >= g.n)
                continue;
            if (x[uu] != 0)
                continue;
            if (fixed_mask && (*fixed_mask)[uu])
                continue;

            matroid_checks_io++;
            if (!matroid_can_add_with_fixed(g, x, top.e, cap, fixed_count, sel2))
                continue;

            if (top.ver != ver)
            {
                const double nd = ksub::kfunc_marginal(g, top.e, lbl, x, f_x);
                queries_io++;
                pq.push(Cand{nd, top.e, ver});
                continue;
            }

            if (top.delta <= 0.0)
                break;

            x[uu] = lbl;
            f_x += top.delta;
            S.push_back(top.e);

            const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
            if (pid < p)
                sel2[pid] += 1;

            ver++;
        }

        return S;
    }

    inline std::vector<mygraph::node_id> approx_alg3_single_label_uniform(
        const mygraph::tinyGraph &g,
        const std::vector<std::uint8_t> *fixed_mask,
        const ksub::Assignment &x_base,
        double f_base,
        ksub::Label lbl,
        std::size_t C_total,
        std::size_t c,
        double eps,
        std::size_t &queries_io,
        std::size_t &matroid_checks_io)
    {
        std::vector<mygraph::node_id> S;
        if (c == 0)
            return S;
        if (C_total == 0)
            return S;

        std::size_t supp_sz = 0;
        for (std::size_t uu = 0; uu < g.n; ++uu)
            if (x_base[uu] != 0)
                supp_sz++;

        const std::size_t r = C_total;
        if (supp_sz >= C_total)
            return S;

        double best_val = -std::numeric_limits<double>::infinity();
        mygraph::node_id e_max = static_cast<mygraph::node_id>(-1);

        for (std::size_t uu = 0; uu < g.n; ++uu)
        {
            if (x_base[uu] != 0)
                continue;
            if (fixed_mask && (*fixed_mask)[uu])
                continue;

            matroid_checks_io++;
            if (supp_sz + 1 > C_total)
                continue;

            const mygraph::node_id e = static_cast<mygraph::node_id>(uu);
            const double delta = ksub::kfunc_marginal(g, e, lbl, x_base, f_base);
            queries_io++;

            const double val = f_base + delta;
            if (val > best_val)
            {
                best_val = val;
                e_max = e;
            }
        }
        if (e_max == static_cast<mygraph::node_id>(-1))
            return S;

        const double M = best_val;
        const double alpha = 1.0 / 3.0;
        double theta = M;

        const double thresh_min = (eps > 0.0) ? (alpha * eps * M / static_cast<double>(r)) : 0.0;

        ksub::Assignment x = x_base;
        double f_x = f_base;

        std::size_t used = 0;
        std::size_t supp2 = supp_sz;

        while (theta >= thresh_min)
        {
            for (std::size_t uu = 0; uu < g.n; ++uu)
            {
                if (used >= c)
                    break;
                if (supp2 >= C_total)
                    break;

                if (x[uu] != 0)
                    continue;
                if (fixed_mask && (*fixed_mask)[uu])
                    continue;

                matroid_checks_io++;
                if (supp2 + 1 > C_total)
                    continue;

                const mygraph::node_id e = static_cast<mygraph::node_id>(uu);
                const double delta = ksub::kfunc_marginal(g, e, lbl, x, f_x);
                queries_io++;

                if (delta >= theta)
                {
                    x[uu] = lbl;
                    f_x += delta;
                    S.push_back(e);
                    used++;
                    supp2++;
                }
            }
            theta = (1.0 - eps) * theta;
            if (theta <= 0.0)
                break;
        }

        return S;
    }

    inline AUFkM_Output run_aufkm_l0_partition(
        const mygraph::tinyGraph &g,
        const std::vector<mygraph::node_id> &ground,
        const matroid::Cap &cap,
        const std::vector<std::size_t> &fixed_count,
        const std::vector<std::uint8_t> *fixed_mask,
        const std::vector<std::size_t> &u_limits,
        double eps)
    {
        const std::size_t n = g.n;
        const std::size_t K = g.K;
        const std::size_t p = matroid::derive_p(g);
    
        if (u_limits.size() != K + 1)
            throw std::invalid_argument("AUFkM(partition): u_limits must have size K+1.");
        if (cap.size() != p)
            throw std::invalid_argument("AUFkM(partition): cap size must equal p.");
        if (fixed_count.size() != p)
            throw std::invalid_argument("AUFkM(partition): fixed_count size != p.");
        if (fixed_mask && fixed_mask->size() != n)
            throw std::invalid_argument("AUFkM(partition): fixed_mask size != n.");
    
        AUFkM_Output out;
        out.x.assign(n, 0);
    
        out.I.reserve(K);
        for (std::size_t i = 1; i <= K; ++i)
            out.I.push_back(static_cast<ksub::Label>(i));
    
        // ----- As confirmed: f_base = 0.0 always -----
        double f_x = 0.0;
    
        std::vector<std::size_t> cnt_label(K + 1, 0);
        std::vector<std::size_t> sel_part(p, 0);
    
        const std::size_t r = sum_cap(cap);
    
        // Compute M = max_{e,i} f(<e,i>) = max_{e,i} Delta_{e,i} f(0)
        double M = -std::numeric_limits<double>::infinity();
        std::vector<std::size_t> sel0(p, 0);
    
        for (auto e : ground)
        {
            const std::size_t uu = static_cast<std::size_t>(e);
            if (uu >= n)
                continue;
            if (fixed_mask && (*fixed_mask)[uu])
                continue;
    
            out.matroid_checks++;
            if (!matroid_can_add_with_fixed(g, out.x, e, cap, fixed_count, sel0))
                continue;
    
            for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl)
            {
                const double delta = ksub::kfunc_marginal(g, e, lbl, out.x, 0.0);
                out.queries++;
                if (delta > M)
                    M = delta;
            }
        }
        if (!std::isfinite(M))
            M = 0.0;
    
        double theta = M;
    
        if (r == 0 || M <= 0.0)
        {
            out.f_value = ksub::kfunc_evaluate(g, out.x);
            out.queries++;
            return out;
        }
    
        const double theta_min = (2.0 * eps * M) / (3.0 * static_cast<double>(r));
    
        bool stop_for_last_label = false;
    
        // ---------------- Thresholding loop ----------------
        while (theta > 0.0 && theta >= theta_min)
        {
            for (auto e : ground)
            {
                const std::size_t uu = static_cast<std::size_t>(e);
                if (uu >= n)
                    continue;
                if (out.x[uu] != 0)
                    continue;
                if (fixed_mask && (*fixed_mask)[uu])
                    continue;
    
                out.matroid_checks++;
                if (!matroid_can_add_with_fixed(g, out.x, e, cap, fixed_count, sel_part))
                    continue;
    
                double best_delta = -std::numeric_limits<double>::infinity();
                ksub::Label best_label = 0;
    
                for (auto lbl : out.I)
                {
                    const double delta = ksub::kfunc_marginal(g, e, lbl, out.x, f_x);
                    out.queries++;
                    if (delta > best_delta)
                    {
                        best_delta = delta;
                        best_label = lbl;
                    }
                }
    
                if (best_label != 0 && best_delta >= theta)
                {
                    out.x[uu] = best_label;
                    f_x += best_delta;
    
                    const std::size_t bi = static_cast<std::size_t>(best_label);
                    cnt_label[bi] += 1;
    
                    const std::size_t pid = static_cast<std::size_t>(g.part_id[uu]);
                    if (pid < p)
                        sel_part[pid] += 1;
    
                    if (cnt_label[bi] >= u_limits[bi])
                    {
                        out.I.erase(std::remove(out.I.begin(), out.I.end(), best_label), out.I.end());
                    }
    
                    if (out.I.size() == 1)
                    {
                        stop_for_last_label = true;
                        break;
                    }
                    if (out.I.empty())
                    {
                        out.f_value = f_x;
                        return out;
                    }
                }
            }
    
            if (stop_for_last_label)
                break;
    
            theta = (1.0 - eps) * theta;
        }
    
        // ---------------- Branch |I| = 1 (UPDATED to match pseudocode) ----------------
        if (out.I.size() == 1)
        {
            const ksub::Label last = out.I[0];
            const std::size_t last_i = static_cast<std::size_t>(last);
            const std::size_t u_last = u_limits[last_i];
    
            // Candidate s  = current out.x
            const double f_s = ksub::kfunc_evaluate(g, out.x);
            out.queries++;
    
            // Candidate s' built independently by Alg_sm on g(S)=f(sum_{e in S} <e,last>)
            ksub::Assignment x_prime(n, 0);
    
            if (u_last > 0)
            {
                const auto S = approx_alg3_single_label_partition(
                    g, cap, fixed_count, fixed_mask,
                    x_prime, 0.0, last, u_last, eps,
                    out.queries, out.matroid_checks);
    
                for (auto u : S)
                {
                    const std::size_t uu2 = static_cast<std::size_t>(u);
                    if (uu2 < n)
                        x_prime[uu2] = last;
                }
            }
    
            const double f_sp = ksub::kfunc_evaluate(g, x_prime);
            out.queries++;
    
            if (f_sp > f_s)
            {
                out.x = std::move(x_prime);
                out.f_value = f_sp;
            }
            else
            {
                out.f_value = f_s;
            }
            return out;
        }
    
        // Default: return s
        out.f_value = ksub::kfunc_evaluate(g, out.x);
        out.queries++;
        return out;
    }
    

    inline AUFkM_Output run_aufkm_l0_uniform(
        const mygraph::tinyGraph &g,
        const std::vector<mygraph::node_id> &ground,
        std::size_t C_total,
        const std::vector<std::size_t> &u_limits,
        double eps,
        std::uint64_t seed = 42ULL)
    {
        const std::size_t n = g.n;
        const std::size_t K = g.K;
    
        if (u_limits.size() != K + 1)
            throw std::invalid_argument("AUFkM(uniform): u_limits must have size K+1.");
    
        AUFkM_Output out;
        out.x.assign(n, 0);
    
        out.I.reserve(K);
        for (std::size_t i = 1; i <= K; ++i)
            out.I.push_back(static_cast<ksub::Label>(i));
    
        // ----- As confirmed: f_base = 0.0 always -----
        double f_x = 0.0;
    
        std::vector<std::size_t> cnt_label(K + 1, 0);
        std::size_t supp_sz = 0;
    
        const std::size_t r = C_total;
    
        if (C_total == 0)
        {
            out.f_value = 0.0;
            return out;
        }
    
        // Compute M = max_{e,i} f(<e,i>) = max_{e,i} Delta_{e,i} f(0)
        double M = -std::numeric_limits<double>::infinity();
        for (auto e : ground)
        {
            const std::size_t uu = static_cast<std::size_t>(e);
            if (uu >= n)
                continue;
            if (supp_sz + 1 > C_total)
                continue;
    
            for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(K); ++lbl)
            {
                const double delta = ksub::kfunc_marginal(g, e, lbl, out.x, 0.0);
                out.queries++;
                if (delta > M)
                    M = delta;
            }
        }
        if (!std::isfinite(M))
            M = 0.0;
    
        double theta = M;
    
        if (r == 0 || M <= 0.0)
        {
            out.f_value = ksub::kfunc_evaluate(g, out.x);
            out.queries++;
            return out;
        }
    
        const double theta_min = (2.0 * eps * M) / (3.0 * static_cast<double>(r));
    
        bool stop_for_last_label = false;
    
        // ---------------- Thresholding loop ----------------
        while (theta > 0.0 && theta >= theta_min)
        {
            for (auto e : ground)
            {
                const std::size_t uu = static_cast<std::size_t>(e);
                if (uu >= n)
                    continue;
                if (out.x[uu] != 0)
                    continue;
                if (supp_sz >= C_total)
                    break;
    
                out.matroid_checks++;
                if (supp_sz + 1 > C_total)
                    continue;
    
                double best_delta = -std::numeric_limits<double>::infinity();
                ksub::Label best_label = 0;
    
                for (auto lbl : out.I)
                {
                    const double delta = ksub::kfunc_marginal(g, e, lbl, out.x, f_x);
                    out.queries++;
                    if (delta > best_delta)
                    {
                        best_delta = delta;
                        best_label = lbl;
                    }
                }
    
                if (best_label != 0 && best_delta >= theta)
                {
                    out.x[uu] = best_label;
                    f_x += best_delta;
                    supp_sz++;
    
                    const std::size_t bi = static_cast<std::size_t>(best_label);
                    cnt_label[bi] += 1;
    
                    if (cnt_label[bi] >= u_limits[bi])
                    {
                        out.I.erase(std::remove(out.I.begin(), out.I.end(), best_label), out.I.end());
                    }
    
                    if (out.I.size() == 1)
                    {
                        stop_for_last_label = true;
                        break;
                    }
                    if (out.I.empty())
                    {
                        out.f_value = f_x;
                        return out;
                    }
                }
            }
    
            if (stop_for_last_label)
                break;
    
            theta = (1.0 - eps) * theta;
        }
    
        // ---------------- Branch |I| = 1 (UPDATED to match pseudocode) ----------------
        if (out.I.size() == 1)
        {
            const ksub::Label last = out.I[0];
            const std::size_t last_i = static_cast<std::size_t>(last);
            const std::size_t u_last = u_limits[last_i];
    
            // Candidate s  = current out.x
            const double f_s = ksub::kfunc_evaluate(g, out.x);
            out.queries++;
    
            // Candidate s' built independently by Alg_sm on g(S)=f(sum_{e in S} <e,last>)
            ksub::Assignment x_prime(n, 0);
    
            const std::size_t C_prime = std::min(C_total, u_last);
    
            if (C_prime > 0)
            {
                const auto S = approx_alg3_single_label_uniform(
                    g, nullptr,
                    x_prime, 0.0, last,
                    C_prime, C_prime, eps,
                    out.queries, out.matroid_checks);
    
                for (auto u : S)
                {
                    const std::size_t uu2 = static_cast<std::size_t>(u);
                    if (uu2 < n)
                        x_prime[uu2] = last;
                }
            }
    
            const double f_sp = ksub::kfunc_evaluate(g, x_prime);
            out.queries++;
    
            if (f_sp > f_s)
            {
                out.x = std::move(x_prime);
                out.f_value = f_sp;
            }
            else
            {
                out.f_value = f_s;
            }
            return out;
        }
    
        // Default: return s
        out.f_value = ksub::kfunc_evaluate(g, out.x);
        out.queries++;
        return out;
    }
    

    inline std::vector<mygraph::node_id> nodes_of_label(
        const ksub::Assignment &x,
        ksub::Label lbl)
    {
        std::vector<mygraph::node_id> v;
        for (std::size_t uu = 0; uu < x.size(); ++uu)
        {
            if (x[uu] == lbl)
                v.push_back(static_cast<mygraph::node_id>(uu));
        }
        return v;
    }

    inline Result run_fksm_alg(
        const mygraph::tinyGraph &g,
        const matroid::Cap &cap,
        const std::vector<std::size_t> &u_limits,
        const std::vector<std::size_t> &l_limits,
        double eps,
        std::uint64_t seed = 42ULL)
    {
        Result res;
        res.algo = "FkSM-Alg";
    
        {
            std::ostringstream oss;
            oss << "partition_matroid(rank=" << sum_cap(cap) << "), eps=" << eps;
            res.constraint = oss.str();
        }
    
        const std::size_t n = g.n;
        const std::size_t K = g.K;
        const std::size_t p = matroid::derive_p(g);
    
        if (cap.size() != p)
            throw std::invalid_argument("AFkM: cap size must equal p.");
        if (u_limits.size() != K + 1)
            throw std::invalid_argument("AFkM: u_limits must have size K+1.");
        if (l_limits.size() != K + 1)
            throw std::invalid_argument("AFkM: l_limits must have size K+1.");
        if (eps <= 0.0 || eps >= 1.0)
            throw std::invalid_argument("AFkM: eps must be in (0,1).");
    
        auto t0 = Clock::now();
    
        std::mt19937_64 rng(seed);
    
        const std::vector<mygraph::node_id> V = make_V(g);
    
        // ------------------------------------------------------------
        // Phase 1: L = argmax_{S in M} |S|  (max independent set for partition matroid)
        // ------------------------------------------------------------
        std::vector<mygraph::node_id> L = build_max_independent_set_L(g, cap);
    
        // ------------------------------------------------------------
        // u <- Adapt AUFkM with upper bounds l_i and M is size constraint sum l_i
        // (implemented as AUFkM-uniform with C_total = sum l_i and u_limits = l_limits)
        // ------------------------------------------------------------
        const std::size_t C_total = sum_u_limits(l_limits, K);
    
        AUFkM_Output u_sol = run_aufkm_l0_uniform(g, V, C_total, l_limits, eps, seed);
        res.queries += u_sol.queries;
        res.matroid_checks += u_sol.matroid_checks;
    
        // ------------------------------------------------------------
        // If |(u)_i| < l_i then fill from L \ supp(u) (and remove from L)
        // ------------------------------------------------------------
        // supp(u) mask
        std::vector<std::uint8_t> in_supp_u(n, 0);
        for (std::size_t uu = 0; uu < n; ++uu)
            if (u_sol.x[uu] != 0)
                in_supp_u[uu] = 1;
    
        // pool = L \ supp(u)
        std::vector<mygraph::node_id> pool;
        pool.reserve(L.size());
        for (auto e : L)
        {
            const std::size_t uu = static_cast<std::size_t>(e);
            if (uu < n && !in_supp_u[uu])
                pool.push_back(e);
        }
        std::shuffle(pool.begin(), pool.end(), rng);
    
        auto cnt_u = label_counts(u_sol.x, K);
    
        for (std::size_t i = 1; i <= K; ++i)
        {
            if (cnt_u[i] >= l_limits[i])
                continue;
    
            const std::size_t need = l_limits[i] - cnt_u[i];
            if (pool.size() < need)
                throw std::runtime_error("AFkM: not enough elements in L \\ supp(u) to fill lower bounds l_i.");
    
            const ksub::Label lbl = static_cast<ksub::Label>(i);
    
            for (std::size_t t = 0; t < need; ++t)
            {
                const mygraph::node_id e = pool.back();
                pool.pop_back();
    
                const std::size_t uu = static_cast<std::size_t>(e);
                if (uu >= n)
                    continue;
                if (u_sol.x[uu] != 0)
                    continue; // safety
    
                u_sol.x[uu] = lbl;
                cnt_u[i] += 1;
                in_supp_u[uu] = 1;
            }
        }
    
        // ------------------------------------------------------------
        // Uniformly sample 2*floor(l_i/2) from (u)_i and split into u^(1), u^(2)
        // ------------------------------------------------------------
        ksub::Assignment u1(n, 0), u2(n, 0);
        std::vector<mygraph::node_id> fixed_set1, fixed_set2;
        fixed_set1.reserve(n);
        fixed_set2.reserve(n);
    
        for (std::size_t i = 1; i <= K; ++i)
        {
            const std::size_t take_total = 2 * (l_limits[i] / 2);
            if (take_total == 0)
                continue;
    
            const ksub::Label lbl = static_cast<ksub::Label>(i);
            auto Ui = nodes_of_label(u_sol.x, lbl);
    
            if (Ui.size() < take_total)
                throw std::runtime_error("AFkM: not enough elements in (u)_i after filling lower bounds.");
    
            std::shuffle(Ui.begin(), Ui.end(), rng);
    
            const std::size_t half = take_total / 2;
            for (std::size_t t = 0; t < take_total; ++t)
            {
                const mygraph::node_id e = Ui[t];
                const std::size_t uu = static_cast<std::size_t>(e);
                if (uu >= n)
                    continue;
    
                if (t < half)
                {
                    u1[uu] = lbl;
                    fixed_set1.push_back(e);
                }
                else
                {
                    u2[uu] = lbl;
                    fixed_set2.push_back(e);
                }
            }
        }
    
        // ------------------------------------------------------------
        // Define matroids M_j via fixed sets: M_j = {T : T U supp(u^(j)) in M}
        // ------------------------------------------------------------
        const std::vector<std::size_t> fixed_count1 = fixed_count_by_part(g, fixed_set1);
        const std::vector<std::size_t> fixed_count2 = fixed_count_by_part(g, fixed_set2);
    
        const std::vector<std::uint8_t> fixed_mask1 = fixed_mask_from_set(g, fixed_set1);
        const std::vector<std::uint8_t> fixed_mask2 = fixed_mask_from_set(g, fixed_set2);
    
        // x^(j) <- AUFkM under M_j with upper bounds u_i
        AUFkM_Output x1 = run_aufkm_l0_partition(g, V, cap, fixed_count1, &fixed_mask1, u_limits, eps);
        AUFkM_Output x2 = run_aufkm_l0_partition(g, V, cap, fixed_count2, &fixed_mask2, u_limits, eps);
    
        res.queries += (x1.queries + x2.queries);
        res.matroid_checks += (x1.matroid_checks + x2.matroid_checks);
    
        // ------------------------------------------------------------
        // Phase 2: y^(j) = x^(j); merge u^(j) into y^(j) if |y_i| < u_i
        // ------------------------------------------------------------
        auto refine_merge = [&](ksub::Assignment y, const ksub::Assignment &uj) -> ksub::Assignment
        {
            auto cnt = label_counts(y, K);
            for (std::size_t uu = 0; uu < n; ++uu)
            {
                const ksub::Label lbl = uj[uu];
                if (lbl == 0)
                    continue;
                if (y[uu] != 0)
                    continue;
    
                const std::size_t i = static_cast<std::size_t>(lbl);
                if (i < 1 || i > K)
                    continue;
                if (cnt[i] >= u_limits[i])
                    continue;
    
                y[uu] = lbl;
                cnt[i] += 1;
            }
            return y;
        };
    
        ksub::Assignment y1 = refine_merge(x1.x, u1);
        ksub::Assignment y2 = refine_merge(x2.x, u2);
    
        // ------------------------------------------------------------
        // Select s = argmax_{x in {y^(1), y^(2), u}} f(x)
        // ------------------------------------------------------------
        const double f_y1 = ksub::kfunc_evaluate(g, y1);
        res.queries++;
        const double f_y2 = ksub::kfunc_evaluate(g, y2);
        res.queries++;
        const double f_u  = ksub::kfunc_evaluate(g, u_sol.x);
        res.queries++;
    
        res.x = std::move(y1);
        res.f_value = f_y1;
    
        if (f_y2 > res.f_value)
        {
            res.x = std::move(y2);
            res.f_value = f_y2;
        }
        if (f_u > res.f_value)
        {
            res.x = std::move(u_sol.x);
            res.f_value = f_u;
        }
    
        // ------------------------------------------------------------
        // Metrics: fair_error, matroid_error, total_error, time
        // ------------------------------------------------------------
        {
            auto cnt = label_counts(res.x, K);
            double fair_err = 0.0;
            for (std::size_t i = 1; i <= K; ++i)
            {
                const double over = (cnt[i] > u_limits[i]) ? double(cnt[i] - u_limits[i]) : 0.0;
                const double under = (cnt[i] < l_limits[i]) ? double(l_limits[i] - cnt[i]) : 0.0;
                fair_err += std::max(over, under);
            }
            res.fair_error = fair_err;
        }
    
        res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
        res.matroid_checks += 1;
    
        res.total_error = res.fair_error + res.matroid_error;
    
        auto t1 = Clock::now();
        res.time_sec = std::chrono::duration<double>(t1 - t0).count();
        return res;
    }

} // namespace algs

#endif // ALGS_FKSM_ALG_H
