// src/objectvalue/maxkcut.h
#ifndef KSUB_OBJECTVALUE_MAXKCUT_H
#define KSUB_OBJECTVALUE_MAXKCUT_H

#include "objectvalue_common.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace ksub {

using mygraph::node_id;
using mygraph::edge_id;


inline double kfunc_evaluate(const mygraph::tinyGraph &g,
                             const Assignment &x)
{
    const std::size_t m = g.m;
    double value = 0.0;

    const bool x_full = (x.size() >= g.n);
    const Label *xptr = x.data();
    const std::size_t xsz = x.size();

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) reduction(+:value)
#endif
    for (edge_id eid = 0; eid < static_cast<edge_id>(m); ++eid) {
        const auto &E = g.edges[eid];
        const node_id u = E.u;
        const node_id v = E.v;

        Label lu, lv;
        if (x_full) {
            lu = xptr[u];
            lv = xptr[v];
        } else {
            lu = (u < xsz ? xptr[u] : 0);
            lv = (v < xsz ? xptr[v] : 0);
        }

        if (lu == 0 || lv == 0) continue;
        if (lu == lv) continue;

        value += E.weights[0];
    }
    return value;
}

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x)
{
    const Label old_label = (u < x.size() ? x[u] : 0);
    if (old_label == new_label) return 0.0;

    double delta = 0.0;

    const Label *xptr = x.data();
    const std::size_t xsz = x.size();

    if (u >= g.incident.size()) {
        const std::size_t m = g.m;

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) reduction(+:delta)
#endif
        for (edge_id eid = 0; eid < static_cast<edge_id>(m); ++eid) {
            const auto &E = g.edges[eid];
            const node_id a = E.u;
            const node_id b = E.v;

            if (a != u && b != u) continue;
            const node_id v = (a == u ? b : a);

            const Label lv = (v < xsz ? xptr[v] : 0);
            const double w  = E.weights[0];

            const double before =
                (old_label != 0 && lv != 0 && old_label != lv) ? w : 0.0;

            const double after  =
                (new_label != 0 && lv != 0 && new_label != lv) ? w : 0.0;

            delta += (after - before);
        }
        return delta;
    }

    const auto &inc = g.incident[u];
    const std::size_t deg = inc.size();

    constexpr std::size_t PAR_DEG_THRESHOLD = 100000;

#if defined(_OPENMP)
    if (deg >= PAR_DEG_THRESHOLD) {
        #pragma omp parallel for schedule(static) reduction(+:delta)
        for (std::size_t i = 0; i < deg; ++i) {
            const edge_id eid = inc[i];
            const auto &E = g.edges[eid];

            node_id v;
            v = (E.u == u ? E.v : E.u);

            const Label lv = (v < xsz ? xptr[v] : 0);
            const double w = E.weights[0];

            const double before =
                (old_label != 0 && lv != 0 && old_label != lv) ? w : 0.0;

            const double after  =
                (new_label != 0 && lv != 0 && new_label != lv) ? w : 0.0;

            delta += (after - before);
        }
        return delta;
    }
#endif

    for (edge_id eid : inc) {
        const auto &E = g.edges[eid];

        node_id v = (E.u == u ? E.v : E.u);
        const Label lv = (v < xsz ? xptr[v] : 0);
        const double w  = E.weights[0];

        const double before =
            (old_label != 0 && lv != 0 && old_label != lv) ? w : 0.0;

        const double after  =
            (new_label != 0 && lv != 0 && new_label != lv) ? w : 0.0;

        delta += (after - before);
    }

    return delta;
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
