=======================================
LCA - Local Clustering and Alternatives
=======================================

Reference implementation of the LCA algorithm - still a work in progress, but accelerating :D


Requirements
------------

* Python 3.7+
* Python dependencies listed in requirements.txt


Citation
--------

If you use this code or its models in your research, please cite:

.. code:: text

    TBD

Documentation
-------------

At the moment (11-21) the documentation is scattered throughout the code, and, of course, some of it is out of date.
But, a good starting point for how to use it is in curate_using_lca.py.  More to come soon.


How to run
-------------

1. Install all requirements:

.. code-block:: python

  install -r requirements.txt

2. Prepare the configuration file according to the example in `configs/config_default.yaml`.
Important details about the configuration file:

* You need a JSON file with annotations. The annotations should contain ground truth information. 'name_keys' in configuration file should be setup to specify how the ground truth are constructed.  
* You need an `embeddings.pickle` file containing embeddings from the re-identification method output for each annotation.

Detailed documentation can be found in `configs/config_default.yaml`.


3. Run the `run.sh` script (make sure it has executable permission). This script executes the `run.py` script with the provided configuration file. You can also directly run `run.py` with the config file by executing:

.. code-block:: bash
  
  python3 run.py --config ./configs/config_default.yaml


4. While LCA is running, you can follow its progress in the `log_file` specified in the config file. Search for `Incremental_stats` to see the accuracy metrics.


5. When LCA is finished, you can find 3 files saved under `db_path`/`exp_name`:

- `quads.csv`: Database file where each quad is of the form (n0, n1, w, aug_name). Here, n0 and n1 are the nodes, w is the signed weight, and aug_name is the augmentation method (a verification algorithm or a human annotator/reviewer) that produced the edge.

- `verifiers_probs.json`: Dictionary containing verification scores/probabilities and associated human decisions. The keys of the dictionary are the verification (augmentation) algorithm names, and the values are dictionaries of probabilities for pairs marked positive (same animal) and negative. Note that the relative proportion of positive and negative scores can matter. This is entirely used for the weighting calibration.
  
.. code-block:: python
  
  ALGO_AUG_NAME: {
      'gt_positive_probs': new_pos_probs,
      'gt_negative_probs': new_neg_probs,
  }
  

- `clustering.json`: The changes to the clustering the method started with, represented as a mapping from cluster ID to a list (or set) of annotation/node IDs.


Task list
---------

https://docs.google.com/document/d/1Ph9CggXPqkzHC-pBEABDTftJxBdBgJIUo8EEJoMzkbo/edit
