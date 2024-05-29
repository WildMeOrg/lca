# Example

An example of running the graph algorithm with the query being about
new nodes, being about a new human edge, and about a cluster than
needs to investigated for a potential change. This is more than
typical.  What's typical would be a query with only verifier edge prob
quads, often involving new nodes. Another typical use case would
involve one or a few clusters that need to be examined, together with
a few human decistions that could impact the clusters.

Now for an explanation of details...


## request_example.json

This starts with a database of four preliminary clusters in the
clustering, and edge quads joining them that are based on both vamp
and human augmentation results.  Note that the edge quads have
positive and negative integer weights. The clusters in the clustering
aren't optimal at the start:  the cluster containing k and l should
also have m. That's why it is is part of the query. Typically, there
might be a human edge adding for information for a cluster, but here I
didn't write it this way.

The generator has some edges specified by hand --- edges that would
come from a request to the verification algorithm or to a
human. Additional edges can come from simulating the generation of
edges using the ground truth (gt) clusters. The probabilities needed
to control this simulation come from verifier_probs.json

The delay_steps just introduce a waiting period between when a
verification quad is requested and when it is returned.

## verifier_probs.json

For each verification algorithm used here there is a list of
probablities produced running the verification algorithm on node
(annotation or encounter) pairs for which there is known
human-generated ("ground truth") decisions. It is imporant that these
be indicative of the expected distribution of examples, but in terms
of the scores (probs) and the relative frquency of positive and
negative examples.

## confi.ini

There are no significant changes here.

##  Command-line

python wbia_lca/example_driver.py \
    --ga_config examples/default/config.ini \
    --verifier_gt examples/default/verifier_probs.json \
    --request examples/default/request_example.json \
    --db_result examples/default/result.json
