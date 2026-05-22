# graphld.multiprocessing_template

Parallel processing utilities for LDGM-based algorithms.

`ParallelProcessor` is a base class for implementing parallel algorithms with LDGMs, wrapping Python's `multiprocessing` module. It splits work among processes, each of which loads a subset of LD blocks.

For higher-level context, see the [Parallel Processing guide](../python_api/parallel_processing.md).

::: graphld.multiprocessing_template
    options:
      show_root_heading: true
      members_order: source
