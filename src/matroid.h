// src/matroid.h
#ifndef MATROID_H
#define MATROID_H

#include <vector>
#include <cstddef>
#include "mygraph.h"
#include "kfunctions.h"

namespace matroid {

using Cap = std::vector<std::size_t>;

std::size_t derive_p(const mygraph::tinyGraph& g);

std::size_t matroid_violation(const mygraph::tinyGraph& g,
                              const ksub::Assignment& x,
                              const Cap& cap);

bool matroid_independent(const mygraph::tinyGraph& g,
                         const ksub::Assignment& x,
                         const Cap& cap);

std::size_t matroid_violation_add_one(const mygraph::tinyGraph& g,
                                      const ksub::Assignment& x,
                                      mygraph::node_id u,
                                      const Cap& cap);

} // namespace matroid

#include "matroid_impl.h"

#endif // MATROID_H
