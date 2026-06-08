#ifndef KSUB_OBJECTVALUE_SENSOR_ENTROPY_H
#define KSUB_OBJECTVALUE_SENSOR_ENTROPY_H

#include "objectvalue_common.h"
#include <stdexcept>
#include <cstddef>

namespace ksub {

using mygraph::node_id;


inline double kfunc_evaluate(const mygraph::tinyGraph &g,
                             const Assignment &x)
{
    const std::size_t n = g.n;
    const std::size_t K = g.K;
    if (n == 0 || K == 0) return 0.0;

    double value = 0.0;
    for (node_id u = 0; u < static_cast<node_id>(n); ++u) {
        const Label lu = (u < x.size() ? x[u] : static_cast<Label>(0));
        if (lu == 0) continue;
        if (lu > static_cast<Label>(K)) {
            throw std::runtime_error("KFUNC_SENSOR_ENTROPY_DISCRETE: label out of range.");
        }
        value += g.nodes[u].weights[static_cast<std::size_t>(lu - 1)];
    }
    return value;
}

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x)
{
    const std::size_t K = g.K;

    const Label old_label = (u < x.size() ? x[u] : static_cast<Label>(0));
    if (old_label == new_label) return 0.0;

    if (new_label > static_cast<Label>(K) || old_label > static_cast<Label>(K)) {
        throw std::runtime_error("KFUNC_SENSOR_ENTROPY_DISCRETE: label out of range.");
    }

    const double old_w = (old_label == 0)
        ? 0.0
        : g.nodes[u].weights[static_cast<std::size_t>(old_label - 1)];

    const double new_w = (new_label == 0)
        ? 0.0
        : g.nodes[u].weights[static_cast<std::size_t>(new_label - 1)];

    return new_w - old_w;
}

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x,
                             double /*f_x*/)
{
    return kfunc_marginal(g, u, new_label, x);
}

}

#endif
