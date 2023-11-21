# LCA Algorithm Repository

## Summary status as of 2020-06-18

## Summary status as of 2019-09-14:

Tested on Python 3.7

Can run simulations using either ground truth clusters specified in a
json file (see gt_ex.json) or using ground truth clusters generated
from a gamma distribution.  Parameters of the simulation can be set
inside the main area of run_from_simulation.py

Example of running on a pre-specified ground-truth

python run_from_simulator.py gt_ex.json tmp

If you leave out the gt_ex.json it runs the
gamma-distribution-generated ground-truth.  This is a bit ugly and I
will fix it soon.

Other than significant clean-up, here are major to dos:

1. Add nodes and edges
2. Delete nodes
3. Use initial clustering with a name anchor so that the name stays
with the node with the lowest id.
4. Provide intermediate clustering results on demand.
