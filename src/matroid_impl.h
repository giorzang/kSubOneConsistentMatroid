// src/matroid_impl.h
#ifndef MATROID_IMPL_H
#define MATROID_IMPL_H

#if !defined(MATROID_PARTITION)
#define MATROID_PARTITION
#endif

#ifdef MATROID_PARTITION
#include "matroid/partition_matroid.h"
#else
#error "No matroid selected. Define MATROID_PARTITION or other matroid macro."
#endif

#endif // MATROID_IMPL_H
