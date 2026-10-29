#!/usr/bin/env python3
"""Run collection, preparation, and c/phi/gamma distribution generation."""
def main(argv=None):
    import sys
    from module.tabpfn_runtime import bootstrap
    argv = sys.argv[1:] if argv is None else argv
    bootstrap(argv, 'analysis')
    from module.pipeline import main as pipeline_main
    return pipeline_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
