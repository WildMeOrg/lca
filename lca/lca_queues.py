# -*- coding: utf-8 -*-
"""
lca_queues.py

Manage the position of the candidate LCAs on three different queues
and one set:

. the main priority Q

. the scoring queue, S, which is the set of LCAs waiting to be
(re)evaluated for alternative local clustering

. the queue, W, of LCAs waiting for an augmentation edge result

. the futile set of LCAs no longer under consideration -- where no
addition of edges could change the clustering decision for the
subgraph covered by the LCA.

All current LCAs must be in exactly one of the queues / sets at the
start of each outer loop of the LCA algorithm.
"""

import logging

import lca_heap as lh


logger = logging.getLogger('lca')


class lca_queues(object):  # NOQA
    def __init__(self, lcas=None):
        """
        The initializer, putting all LCAs on the scoring queue and
        making S, W and futile all empty.
        """
        self.Q = lh.lca_heap()
        if lcas is None:
            self.S = set()
        else:
            self.S = set(lcas)
        self.W = set()
        self.futile = set()

    def num_lcas(self):
        return len(self.Q) + len(self.S) + len(self.W) + len(self.futile)

    def top_Q(self):
        """
        Access the top LCA on the main priority queue, Q, without
        removing it. If it is empty, return None
        """
        if len(self.Q) > 0:
            return self.Q.top_Q()
        else:
            return None

    def pop_Q(self):
        """
        Remove the top LCA from the main priority queue, Q.
        """
        self.Q.pop_Q()

    def add_to_Q(self, lcas):
        """
        Add the given LCAs to the main priority queue, Q.
        """
        if type(lcas) != list and type(lcas) != set:
            lcas = [lcas]
        for a in lcas:
            self.Q.insert(a)

    def get_S(self):
        """
        Access the scoring queue set S
        """
        return self.S

    def clear_S(self):
        """
        Clear the scoring queue set S.
        """
        self.S.clear()

    def add_to_S(self, a):
        """
        Add LCA a to the scoring set. This assumes without checking
        that a is not already on any of the queues.
        """
        self.S.add(a)

    def add_to_W(self, a):
        """
        Add LCA a to the weight  set. This assumes without checking
        that a is not already on any of the queues.
        """
        self.W.add(a)

    def num_on_W(self):
        """
        Return the number of LCAs on the waiting (for edge
        augmentation) queue.
        """
        return len(self.W)

    def add_to_futile(self, a):
        """
        Mark an LCA as futile --- all possibilities reasonably explored
        """
        self.futile.add(a)

    def remove(self, lcas):
        """
        Remove the given LCAs from the queues.
        """
        if type(lcas) != list and type(lcas) != set:
            lcas = [lcas]
        rW = 0
        rS = 0
        rQ = 0
        rf = 0
        for a in lcas:
            if a in self.W:
                rW += 1
                self.W.remove(a)
            elif a in self.S:
                rS += 1
                self.S.remove(a)
            elif a in self.futile:
                rf += 1
                self.futile.remove(a)
            else:
                rQ += 1
                self.Q.remove(a)
        # logger.info('Removed Q: %d S: %d W: %d futile: %d' % (rQ, rS, rW, rf))

    def reset_waiting(self):
        self.S |= self.W
        self.W.clear()

    def switch_to_splitting(self):
        """
        Switch to the splitting phase of the computation.  This
        requires clearing all queues prior to the creation of
        singleton LCAs for splitting.
        """
        self.Q.clear()
        self.S.clear()
        self.W.clear()

    def switch_to_stability(self):
        """
        Nothing needs to be be done because of the splitting phase.
        """
        pass

    def score_change(self, a, from_delta, to_delta):
        """
        Move the position of LCA a based on the change to the score
        for its current local clustering, as recorded in from_delta,
        and its best alternative, as recorded in to_delta.  The key is
        the to_delta value because only when this is negative does the
        alternative clustering need to be recomputed.
        """
        if a in self.S:
            pass  # leave it for an update
        elif to_delta < 0:
            self.remove(a)
            self.add_to_S(a)
        else:
            self.remove(a)
            self.Q.insert(a)

    def which_queue(self, a):
        """
        Return a single character string indicating the queue that LCA
        a is on.  Choices are 'S' (scoring), 'W' (waiting - for
        augmentation), 'Q' (main queue) or 'F' (futile).  If the LCA is
        not on any of the queues, None is returned.
        """
        if a in self.S:
            return 'S'
        elif a in self.W:
            return 'W'
        elif a in self.Q.heap:
            return 'Q'
        elif a in self.futile:
            return 'F'
        else:
            return None

    def is_consistent(self):
        """
        Check the consistencu of the queues: the main Q is itself
        consistent, and all queues are pairwise disjoint from each
        other.
        """
        all_ok = self.Q.is_consistent()
        q_set = set(self.Q.lca2index.keys())

        qs = q_set & self.S
        if len(qs) > 0:
            logger.info('LCA queues, Q and S intersect')
            all_ok = False

        qw = q_set & self.W
        if len(qw) > 0:
            logger.info('LCA queues, Q and W intersect')
            all_ok = False

        qd = q_set & self.futile
        if len(qd) > 0:
            logger.info('LCA queues, Q and futile queue intersect')
            all_ok = False

        sw = self.S & self.W
        if len(sw) > 0:
            logger.info('LCA queues, S and W intersect')
            all_ok = False

        sd = self.S & self.futile
        if len(sd) > 0:
            logger.info('LCA queues, S and futile intersect')
            all_ok = False

        wd = self.W & self.futilene
        if len(wd) > 0:
            logger.info('LCA queues, W and futile queue intersect')
            all_ok = False

        return all_ok

    def log(self):
        logger.info(
            'Number of LCAs on Q %d, W %d, S %d, futile %d'
            % (len(self.Q), len(self.W), len(self.S), len(self.futile))
        )

    def info_long(self, max_entries=-1):
        logger.info('LCA Queues:')

        # Log Q
        spaces = ' ' * 4
        if len(self.Q) == 0:
            logger.info('Q: <empty>')
        else:
            if max_entries <= 0 or max_entries > len(self.Q):
                max_entries = len(self.Q)
            logger.info('Q: %d entries; printing %d' % (len(self.Q), max_entries))
            for i in range(max_entries):
                initial_str = '%4d: ' % i
                self.Q.heap[i].pprint_short(
                    initial_str=initial_str, stop_after_from=False
                )

        # Log S
        if len(self.S) == 0:
            logger.info('S: <empty>')
        else:
            logger.info('S: %d entries:' % len(self.S))
            for lca in self.S:
                lca.pprint_short(initial_str=spaces, stop_after_from=True)

        # Log W
        if len(self.W) == 0:
            logger.info('W: <empty>')
        else:
            logger.info('W: %d entries:' % len(self.W))
            for lca in self.W:
                lca.pprint_short(initial_str=spaces, stop_after_from=True)


def test_lca_queues():
    v = [
        lh.lca_lite(123, 1.0),
        lh.lca_lite(456, 5.3),
        lh.lca_lite(827, 7.8),
        lh.lca_lite(389, 8.9),
        lh.lca_lite(648, 8.6),
        lh.lca_lite(459, 9.4),
        lh.lca_lite(628, 8.2),
        lh.lca_lite(747, 4.7),
    ]
    queues = lca_queues(v)

    logger.info('')
    logger.info(
        'After initialization: lengths should be (0, %d, 0, 0)'
        ' and are (%d, %d, %d, %d)'
        % (len(v), len(queues.Q), len(queues.S), len(queues.W), len(queues.futile))
    )
    queues.get_S()
    queues.clear_S()
    queues.add_to_S(v[0])
    queues.add_to_S(v[1])
    queues.add_to_Q(v[2:-2])
    queues.add_to_W(v[-2])
    queues.add_to_W(v[-1])
    lcas_on_S = v[:2]
    lcas_on_Q = v[2:-2]
    lcas_on_W = v[-2:]
    logger.info(
        'After moving around: lengths should be (%d, 2, 2)'
        ' and are (%d, %d, %d)'
        % (len(v) - 4, len(queues.Q), len(queues.S), len(queues.W))
    )
    logger.info('num_on_W should be %d, and is %d' % (len(queues.W), queues.num_on_W()))
    logger.info(
        'Which queue: should be S and is %s' % (queues.which_queue(lcas_on_S[0]),)
    )
    logger.info(
        'Which queue: should be Q and is %s' % (queues.which_queue(lcas_on_Q[0]),)
    )
    logger.info(
        'Which queue: should be W and is %s' % (queues.which_queue(lcas_on_W[0]),)
    )

    logger.info('Here is Q:')
    queues.Q.print_structure()
    a = queues.top_Q()
    logger.info('top of queue should have values (459, 9.4) and has %s' % (str(a),))
    queues.pop_Q()
    a1 = queues.top_Q()
    logger.info(
        'popped off queue; new top should have values (389, 8.9) and has %s' % (str(a1),)
    )
    queues.add_to_Q(a)  # put it back on....
    logger.info('put top back on')

    logger.info('---------------')
    logger.info('Testing score_change method:')
    queues.score_change(lcas_on_S[0], 4, -3)
    logger.info('Changed on S should stay on S: %s' % (queues.which_queue(lcas_on_S[0]),))
    queues.score_change(lcas_on_Q[0], 4, -3)
    logger.info(
        "Negative 'to' score change from Q should be on S: %s"
        % (queues.which_queue(lcas_on_Q[0]),)
    )
    queues.score_change(lcas_on_Q[1], -4, 3)
    logger.info(
        "Negative 'from' score change (positive to) from Q should be on Q: %s"
        % (queues.which_queue(lcas_on_Q[1]),)
    )
    queues.score_change(lcas_on_W[0], 4, -3)
    logger.info(
        "Negative 'to' score change from W should be on S: %s"
        % (queues.which_queue(lcas_on_W[0]),)
    )
    queues.score_change(lcas_on_W[1], -4, 3)
    logger.info(
        "Negative 'from' score change (positive to) from W should be on Q: %s"
        % (queues.which_queue(lcas_on_W[1]),)
    )

    v = []
    queues = lca_queues(v)
    a = queues.top_Q()
    logger.info('Empty main queue should have a top of None: %s' % (a,))

    v = [
        lh.lca_lite(123, 1.0),
        lh.lca_lite(459, 9.4),
        lh.lca_lite(628, 8.2),
        lh.lca_lite(747, 4.7),
    ]
    queues = lca_queues(v)

    logger.info('')
    logger.info(
        'After initialization (again): lengths should be (0, %d, 0, 0)'
        ' and are (%d, %d, %d, %d)'
        % (len(v), len(queues.Q), len(queues.S), len(queues.W), len(queues.futile))
    )
    queues.clear_S()
    queues.add_to_Q(v[0])
    queues.add_to_futile(v[1])
    queues.add_to_futile(v[2])
    queues.add_to_futile(v[3])
    logger.info(
        'After moving around: lengths should be (1, 0, 0, 3)'
        ' and are (%d, %d, %d, %d)'
        % (len(queues.Q), len(queues.S), len(queues.W), len(queues.futile))
    )
    queues.remove(v[3])
    queues.score_change(v[2], from_delta=0, to_delta=-4)
    logger.info('Which queue: should be Q and is %s' % (queues.which_queue(v[0]),))
    logger.info('Which queue: should be F and is %s' % (queues.which_queue(v[1]),))
    logger.info('Which queue: should be S and is %s' % (queues.which_queue(v[2]),))
    logger.info(
        'At end: lengths should be (1, 1, 0, 1)'
        ' and are (%d, %d, %d, %d)'
        % (len(queues.Q), len(queues.S), len(queues.W), len(queues.futile))
    )


if __name__ == '__main__':
    test_lca_queues()
