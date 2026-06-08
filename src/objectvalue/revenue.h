#ifndef KSUB_OBJECTVALUE_REVENUE_H
#define KSUB_OBJECTVALUE_REVENUE_H

#include "objectvalue_common.h"

namespace ksub {

using mygraph::node_id;
using mygraph::edge_id;


double kfunc_evaluate(const mygraph::tinyGraph &g,
                      const Assignment &x)
{
    const std::size_t n = g.n;
    std::vector<double> inbound(n, 0.0);

    for (edge_id eid = 0; eid < g.m; ++eid) {
        const auto &E = g.edges[eid];
        node_id v = E.u; // nguồn
        node_id u = E.v; // đích

        Label lv = (v < x.size() ? x[v] : 0);
        if (lv == 0) continue; // v ∉ S

        inbound[u] += E.weights[0];
    }

    double value = 0.0;
    for (node_id u = 0; u < n; ++u) {
        double s = inbound[u];
        if (s <= 0.0) continue;

        double alpha_u = g.nodes[u].alpha[0];
        value += std::pow(s, alpha_u);
    }
    return value;
}

double kfunc_marginal(const mygraph::tinyGraph &g,
                      node_id xnode,
                      Label new_label,
                      const Assignment &x)
{
    const std::size_t n = g.n;

    Label old_label = (xnode < x.size() ? x[xnode] : 0);
    bool old_in = (old_label != 0);
    bool new_in = (new_label != 0);
    if (old_in == new_in) return 0.0;

    std::vector<double> inbound_before(n, 0.0);
    for (edge_id eid = 0; eid < g.m; ++eid) {
        const auto &E = g.edges[eid];
        node_id v = E.u;
        node_id u = E.v;

        Label lv = (v < x.size() ? x[v] : 0);
        if (lv == 0) continue;

        inbound_before[u] += E.weights[0];
    }

    std::vector<double> inbound_after = inbound_before;

    if (old_in || new_in) {
        if (xnode < g.incident.size()) {
            for (edge_id eid : g.incident[xnode]) {
                const auto &E = g.edges[eid];

                node_id u = (g.undirected ? (E.u == xnode ? E.v : E.u) : E.v);
                double w = E.weights[0];

                if (old_in) inbound_after[u] -= w;
                if (new_in) inbound_after[u] += w;
            }
        } else {
            for (edge_id eid = 0; eid < g.m; ++eid) {
                const auto &E = g.edges[eid];
                node_id v = E.u;
                node_id u = E.v;
                double w = E.weights[0];

                if (v != xnode) continue;

                if (old_in) inbound_after[u] -= w;
                if (new_in) inbound_after[u] += w;
            }
        }
    }

    double before_val = 0.0, after_val = 0.0;
    for (node_id u = 0; u < n; ++u) {
        double a = g.nodes[u].alpha[0];

        if (inbound_before[u] > 0.0)
            before_val += std::pow(inbound_before[u], a);

        if (inbound_after[u] > 0.0)
            after_val += std::pow(inbound_after[u], a);
    }

    return after_val - before_val;
}

double kfunc_marginal(const mygraph::tinyGraph &g,
                      node_id xnode,
                      Label new_label,
                      const Assignment &x,
                      double f_x)
{
    const std::size_t n = g.n;

    Label old_label = (xnode < x.size() ? x[xnode] : 0);
    bool old_in = (old_label != 0);
    bool new_in = (new_label != 0);
    if (old_in == new_in) return 0.0;

    std::vector<double> inbound_before(n, 0.0);
    for (edge_id eid = 0; eid < g.m; ++eid) {
        const auto &E = g.edges[eid];
        node_id v = E.u;
        node_id u = E.v;

        Label lv = (v < x.size() ? x[v] : 0);
        if (lv == 0) continue;

        inbound_before[u] += E.weights[0];
    }

    std::vector<double> inbound_after = inbound_before;

    if (old_in || new_in) {
        if (xnode < g.incident.size()) {
            for (edge_id eid : g.incident[xnode]) {
                const auto &E = g.edges[eid];
                node_id u = (g.undirected ? (E.u == xnode ? E.v : E.u) : E.v);
                double w = E.weights[0];

                if (old_in) inbound_after[u] -= w;
                if (new_in) inbound_after[u] += w;
            }
        } else {
            for (edge_id eid = 0; eid < g.m; ++eid) {
                const auto &E = g.edges[eid];
                node_id v = E.u;
                node_id u = E.v;
                double w = E.weights[0];

                if (v != xnode) continue;

                if (old_in) inbound_after[u] -= w;
                if (new_in) inbound_after[u] += w;
            }
        }
    }

    double after_val = 0.0;
    for (node_id u = 0; u < n; ++u) {
        double s = inbound_after[u];
        if (s <= 0.0) continue;
        double a = g.nodes[u].alpha[0];
        after_val += std::pow(s, a);
    }

    return after_val - f_x;
}

} 

#endif
