// src/matroid/partition_matroid.h
#ifndef MATROID_PARTITION_MATROID_H
#define MATROID_PARTITION_MATROID_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#include "mygraph.h"
#include "kfunctions.h"

namespace matroid {

inline std::size_t derive_p(const mygraph::tinyGraph& g) {
    if (g.part_id.size() != g.n) {
        throw std::invalid_argument("derive_p: g.part_id.size() != g.n");
    }
    if (g.n == 0) return 0;

    std::uint32_t mx = 0;
    for (std::size_t u = 0; u < g.n; ++u) {
        mx = std::max(mx, g.part_id[u]);
    }
    return static_cast<std::size_t>(mx) + 1;
}

inline std::vector<std::size_t> count_parts_(const mygraph::tinyGraph& g,
                                             const ksub::Assignment& x,
                                             std::size_t p)
{
    if (x.size() != g.n) {
        throw std::invalid_argument("count_parts_: x.size() != g.n");
    }
    if (g.part_id.size() != g.n) {
        throw std::invalid_argument("count_parts_: g.part_id.size() != g.n");
    }

    std::vector<std::size_t> cnt(p, 0);
    for (std::size_t u = 0; u < g.n; ++u) {
        if (x[u] == 0) continue;
        const std::size_t pid = static_cast<std::size_t>(g.part_id[u]);
        if (pid >= p) {
            throw std::invalid_argument("count_parts_: part_id out of range");
        }
        cnt[pid]++;
    }
    return cnt;
}

inline std::size_t matroid_violation(const mygraph::tinyGraph& g,
                                     const ksub::Assignment& x,
                                     const std::vector<std::size_t>& cap)
{
    const std::size_t p = derive_p(g);
    if (p == 0) return 0;

    if (cap.size() != p) {
        throw std::invalid_argument("matroid_violation: cap.size() != p");
    }

    auto cnt = count_parts_(g, x, p);

    std::size_t viol = 0;
    for (std::size_t j = 0; j < p; ++j) {
        if (cnt[j] > cap[j]) viol += (cnt[j] - cap[j]);
    }
    return viol;
}

inline bool matroid_independent(const mygraph::tinyGraph& g,
                                const ksub::Assignment& x,
                                const std::vector<std::size_t>& cap)
{
    return matroid_violation(g, x, cap) == 0;
}

inline std::size_t matroid_violation_add_one(const mygraph::tinyGraph& g,
                                             const ksub::Assignment& x,
                                             mygraph::node_id u,
                                             const std::vector<std::size_t>& cap)
{
    if (x.size() != g.n) {
        throw std::invalid_argument("matroid_violation_add_one: x.size() != g.n");
    }
    if (g.part_id.size() != g.n) {
        throw std::invalid_argument("matroid_violation_add_one: g.part_id.size() != g.n");
    }

    const std::size_t uu = static_cast<std::size_t>(u);
    if (uu >= g.n) {
        throw std::out_of_range("matroid_violation_add_one: u out of range");
    }

    if (x[uu] != 0) return 0;

    const std::size_t p = derive_p(g);
    if (p == 0) return 0;

    if (cap.size() != p) {
        throw std::invalid_argument("matroid_violation_add_one: cap.size() != p");
    }

    const std::size_t pid_u = static_cast<std::size_t>(g.part_id[uu]);
    if (pid_u >= p) {
        throw std::invalid_argument("matroid_violation_add_one: part_id[u] out of range");
    }

    std::size_t cnt_u = 0;
    for (std::size_t v = 0; v < g.n; ++v) {
        if (x[v] == 0) continue;
        if (static_cast<std::size_t>(g.part_id[v]) == pid_u) cnt_u++;
    }

    return (cnt_u >= cap[pid_u]) ? 1 : 0;
}

} // namespace matroid

#endif // MATROID_PARTITION_MATROID_H
