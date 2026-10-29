#!/usr/bin/env python3
"""Explicit command entry point for full-area distributions; import does nothing."""
if __name__ == '__main__':
    import sys
    from module.tabpfn_runtime import bootstrap
    bootstrap(sys.argv[1:], 'distribution')
    from module.full_distribution import main
    raise SystemExit(main(sys.argv[1:] if len(sys.argv) > 1 else ['--help']))
