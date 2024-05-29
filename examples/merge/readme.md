# Example - Merge

Here is an example of running the graph algorithm with the goal
of merging two previously unrelated databasess. This requires a basic
combination of the databases, followed by running the graph algorithm
to combine ids. This can lead to merges when nodes from the two
different databases are shown to be from the same animal, and to
splits when the combined information shows that the combined
information from the two sets of nodes shows that a cluster of nodes
supposedly from the same animal is in fact of different animals. The
example here shows a combined merge/split.

As per usual, the information provided here starts after the ranking
algorithm has been run, and then the sufficiently highly ranking
matches have been formed into edges and run through a verification
algorithm.



## request_example.json

This starts with a database of three clusters. Think of this as the
merge between two databases. The first contains a, b, c in an initial
cluster and the second contains d, e, f, g and h, i as two separate
clusters. The ground truth (provided to the generator) on the other
hand connects a, b, d, e, f in one cluster, c, g in a second and h, i
in a third. This rearrangement requires a merge/split operation.

The generator only uses the ground truth and the distribution of edges
to form the weighters. There are no hand-specified verification or
human edges.

The actual query contains one or two matches for each of a, b and c
--- in other words the query is assumed to have gone from the smaller
database to the large.  Some matches are positive. Together, these
make it look like there should be one big merge, which actually occurs
as an intermediate stage.  It is not stable enough to converge before
augmentation starts.  When this does start, the graph algorithm
discovers negative connections that cause the final split.

## verifier_probs.json

I've added a few here over the earlier example, but nothing
substantial. See the first example for details.

## default_confi.ini

There are no significant changes here.

##  Command-line

python wbia_lca/example_driver.py \
    --ga_config examples/merge/config.ini \
    --verifier_gt examples/merge/verifier_probs.json \
    --request examples/merge/request_example.json \
    --db_result examples/merge/result.json
