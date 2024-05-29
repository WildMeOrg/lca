# -*- coding: utf-8 -*-
from init_logger import init_logger

init_logger()

try:
    from lca._version import __version__
except ImportError:
    __version__ = '0.0.0'

# from lca.version import version as __version__  # NOQA
import __main__  # NOQA

# try:
#     import _plugin  # NOQA
# except ModuleNotFoundError:
#     logger.warn('Wildbook-IA (wbia) needs to be installed')

import baseline  # NOQA
import cid_to_lca  # NOQA
import cluster_tools  # NOQA
import compare_clusterings  # NOQA
import db_interface  # NOQA
import db_interface_sim  # NOQA
import draw_lca  # NOQA
import edge_generator  # NOQA
import edge_generator_sim  # NOQA
import exp_scores  # NOQA
import ga_driver  # NOQA
import graph_algorithm  # NOQA
import lca  # NOQA
import lca_alg1  # NOQA
import lca_alg2  # NOQA
import lca_heap  # NOQA
import lca_queues  # NOQA
import example_driver  # NOQA
import run_from_simulator  # NOQA
import simulator  # NOQA
import test_cluster_tools  # NOQA
import test_graph_algorithm  # NOQA
import weight_manager  # NOQA
import weighter  # NOQA
