#pragma once

#include "tscout/marker.h"

/*
 * Wrapper to add TScout markers to an executor. In the executor file, rename
 * the current Exec<blah> function to WrappedExec<blah> and then add
 * TS_EXECUTOR_WRAPPER<blah> beneath it. See src/backend/executors for examples.
 *
 * There is a small list of executors that cannot use this macro due to function
 * signature differences. If the macro below changes, be sure to update those
 * executors as well. The current list is:
 *
 * src/backend/executors/nodeSubplan.c
 * src/backend/executors/nodeHashJoin.c
 */
#define TS_EXECUTOR_WRAPPER(node_type)                                         \
  static TupleTableSlot *Exec##node_type(PlanState *pstate) {                  \
    TupleTableSlot *result;                                                    \
    TS_MARKER(Exec##node_type##_begin, pstate->plan->plan_node_id);            \
                                                                               \
    result = WrappedExec##node_type(pstate);                                   \
                                                                               \
    TS_MARKER(Exec##node_type##_end, pstate->plan->plan_node_id);              \
    return result;                                                             \
  }