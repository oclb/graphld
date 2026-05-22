# Parallel Processing

`ParallelProcessor` is the base class for LDGM-backed multiprocessing workflows in GraphLD. It splits work across processes so each worker can load only the LD blocks it needs.

The main practical benefit is that LDGM loading happens inside worker processes rather than being serialized in the parent process.

For a concrete example, see `tests/test_multiprocessing.py`.

See also:

- [graphld.multiprocessing_template API Reference](../api/multiprocessing.md)
- [graphld package overview](../api/graphld.md)
