# -*- coding: utf-8 -*-
"""Command-line entry point for the package.

    python -m beta_stability --config path/to/species.yaml

This is the same command as the ``beta_stability-cluster`` console script
declared in ``pyproject.toml``; both run :func:`beta_stability.run.main`.
"""
from beta_stability.run import main


if __name__ == '__main__':
    main()
