// src/kfunctions_impl.h
#ifndef KFUNCTIONS_IMPL_H
#define KFUNCTIONS_IMPL_H


#if (defined(KFUNC_MAXKCUT) + defined(KFUNC_REVENUE) + defined(KFUNC_KIC) + defined(KFUNC_SENSOR_ENTROPY)) != 1
#error "You must define exactly one of: KFUNC_MAXKCUT, KFUNC_REVENUE, KFUNC_KIC, KFUNC_SENSOR_ENTROPY"
#endif

#if defined(KFUNC_MAXKCUT)
#include "objectvalue/maxkcut.h"
#elif defined(KFUNC_REVENUE)
#include "objectvalue/revenue.h"
#elif defined(KFUNC_KIC)
#include "objectvalue/kic.h"
#elif defined(KFUNC_SENSOR_ENTROPY)
#include "objectvalue/sensor_entropy.h"
#endif

#endif // KFUNCTIONS_IMPL_H
