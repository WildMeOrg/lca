# -*- coding: utf-8 -*-
"""
Implements a max heap of LCA's.  Priority is based on the delta_score value.
LCA's may be removed from the middle. In fact this is intended to be a common
operation since most LCA's are relatively short-lived and never reach the
top of the heap.  The LCA object is assumed to have a hash value.

This is implemented using an auxiliary dictionary that associates a heap index
with each LCA. These indices are updated as the LCA is moved through the heap
during percolate up and percolate down.
"""
import logging


logger = logging.getLogger('wbia_lca')


class lca_heap(object):  # NOQA
    def __init__(self):
        self.heap = []
        self.lca2index = dict()

    def top_Q(self):
        """
        Access the top LCA (max delta score) of the priority queue without
        deleting it.
        """
        return self.heap[0]

    def __len__(self):
        """
        Return the number of LCAs on the priority queue.
        """
        return len(self.heap)

    def pop_Q(self):
        """
        Remove the top LCA from the priority queue.  Does not return it.
        """
        if len(self.heap) == 1:
            self.clear()
        else:
            a_top = self.heap[0]
            del self.lca2index[a_top]
            a_last_leaf = self.heap.pop()
            self.heap[0] = a_last_leaf
            self.lca2index[a_last_leaf] = 0
            self.percolate_down(0)

    def remove(self, a):
        """
        Remove a particular LCA from the priority queue, no matter where it is
        Assert fails if the LCA is not there.
        """
        """
        a.pprint()
        logger.info("len of index", len(self.lca2index))
        logger.info("len of heap", len(self.heap))
        """
        assert a in self.lca2index

        # Using the dictionary, find the location of LCA a in the
        # heap, and then remove a from the dictionary
        loc = self.lca2index[a]
        del self.lca2index[a]

        # Special case of removing the end of list.
        # This also works for removing the last item.
        if loc == len(self.heap) - 1:
            self.heap.pop()
            return

        a_last_leaf = self.heap.pop()
        self.heap[loc] = a_last_leaf
        self.lca2index[a_last_leaf] = loc
        p_loc = (loc - 1) // 2
        if p_loc >= 0 and self.heap[p_loc].delta_score() < self.heap[loc].delta_score():
            self.percolate_up(loc)
        else:
            self.percolate_down(loc)

    def insert(self, a):
        """
        Insert an LCA into the priority queue. Assumes without checking that
        the LCA is not already in the queue.
        """
        self.heap.append(a)
        last_index = len(self.heap) - 1
        self.lca2index[a] = last_index
        self.percolate_up(last_index)

    def clear(self):
        """
        Completely clear the priority queue of all LCAs
        """
        self.heap.clear()
        self.lca2index.clear()

    def get_all(self):
        """
        Return a shallow copy of the LCA heap.
        """
        return self.heap

    def percolate_down(self, i):
        """
        Run the standard percolate down operation from the particular
        index i, update the mapping from LCAs to indices at each step.
        """
        i_lca = self.heap[i]
        i_delta = i_lca.delta_score()
        last_interior = len(self.heap) // 2 - 1

        while i <= last_interior:
            c = 2 * i + 1
            c_lca = self.heap[c]
            c_delta = c_lca.delta_score()
            rc = c + 1
            if rc < len(self.heap) and self.heap[rc].delta_score() > c_delta:
                c = rc
                c_lca = self.heap[c]
                c_delta = c_lca.delta_score()

            if i_delta >= c_delta:
                break
            else:
                self.heap[i] = c_lca
                self.lca2index[c_lca] = i
                i = c

        self.lca2index[i_lca] = i
        self.heap[i] = i_lca

    def percolate_up(self, i):
        """
        Run the standard percolate up operation from the particular
        index i, update the mapping from LCAs to indices at each step.
        """
        i_lca = self.heap[i]
        i_delta = i_lca.delta_score()
        while i > 0:
            p = (i - 1) // 2
            p_lca = self.heap[p]
            p_delta = p_lca.delta_score()
            if p_delta >= i_delta:
                break
            else:
                self.heap[i] = p_lca
                self.lca2index[p_lca] = i
                i = p
        self.heap[i] = i_lca
        self.lca2index[i_lca] = i

    def print_structure(self):
        """
        Print the LCA heap vector and dictionary.
        """
        logger.info('Here is the heap vector')
        for i, lca in enumerate(self.heap):
            logger.info('    %d: %s' % (i, str(lca)))
        logger.info('Here is the dictionary')
        for k, v in self.lca2index.items():
            logger.info('    %s: %d' % (str(k), v))

    def is_consistent(self):
        """
        Check the consistency of the LCA heap
        """
        is_ok = True
        if len(self.heap) != len(self.lca2index):
            is_ok = False
            logger.info(
                'is_consistent: heap is len',
                len(self.heap),
                ' while lca2index is len',
                len(self.lca2index),
            )

        # Make sure each index is in the heap
        for i, lca in enumerate(self.heap):
            if lca not in self.lca2index:
                logger.info(
                    'lca at location %d with heap value %d not in lca2index'
                    % (i, lca.__heap)
                )
                is_ok = False

        # Make sure each lca2index entry is unique
        if len(self.lca2index.values()) != len(set(self.lca2index.values())):
            logger.info('Duplicated indices in lca2index values')
            is_ok = False

        # Make sure all indices are in range
        if not all([0 <= i < len(self.heap) for i in self.lca2index.values()]):
            logger.info('At least one index out of range in self.lca2index.values')
            is_ok = False

        # Finally test to see if the ordering property is maintained
        last_internal = len(self.heap) // 2 - 1
        for i in range(last_internal + 1):
            lchild = 2 * i + 1
            rchild = lchild + 1
            if self.heap[i].delta_score() < self.heap[lchild].delta_score():
                logger.info(
                    'Heap index %d has score %1.1f less than left child %d with score %1.1f'
                    % (
                        i,
                        self.heap[i].delta_score(),
                        lchild,
                        self.heap[lchild].delta_score(),
                    )
                )
                is_ok = False

            if (
                rchild < len(self.heap)
                and self.heap[i].delta_score() < self.heap[lchild].delta_score()
            ):
                logger.info(
                    'Heap index %d has score %1.1f, less than right child %d with score %1.1f'
                    % (
                        i,
                        self.heap[i].delta_score(),
                        rchild,
                        self.heap[rchild].delta_score(),
                    )
                )
                is_ok = False

        if not is_ok:
            logger.info('Output of inconsistent data structure')
            self.print_structure()

        return is_ok


class lca_lite(object):  # NOQA
    """
    A version of the LCA object used for testing. It only has a hash value and a
    delta score.
    """

    def __init__(self, hash_value, delta_s):
        self.__hash = hash_value
        self.m_delta_score = delta_s

    def __hash__(self):
        return self.__hash

    def delta_score(self):
        return self.m_delta_score

    def __str__(self):
        return 'hash = %d, delta_score = %1.1f' % (self.__hash, self.m_delta_score)

    def pprint(self):
        logger.info(str(self))


def test_lca_heap():
    h = lca_heap()

    v = [
        lca_lite(123, 1.0),
        lca_lite(456, 5.3),
        lca_lite(827, 7.8),
        lca_lite(389, 8.9),
        lca_lite(648, 8.6),
        lca_lite(459, 9.4),
        lca_lite(628, 8.2),
        lca_lite(747, 4.7),
    ]
    remove0 = v[1]

    found_error = False
    for a in v:
        h.insert(a)
        if not h.is_consistent():
            found_error = True
            logger.info('Breaking on inconsistency')
            break

    if not found_error:
        logger.info('After %d successful inserts the heap looks like' % len(h))
        h.print_structure()

    logger.info('Top of queue should be (459, 9.4) and is %s' % str(h.top_Q()))
    remove1 = lca_lite(585, 8.5)
    h.insert(remove1)

    h.pop_Q()
    logger.info('After pop_Q')
    if not h.is_consistent():
        logger.info('Inconsistent')
    else:
        logger.info('Consistent. Here is queue.')
        h.print_structure()
        logger.info(
            'top value should have delta_score 8.9.  It has %s' % str(h.get_all()[0])
        )

    h.insert(lca_lite(183, 8.3))
    logger.info(
        'Trying two remove operations; one should trigger percolate up and the other percolate down'
    )
    h.remove(remove0)
    h.remove(remove1)
    if not h.is_consistent():
        logger.info('Inconsistent')
    else:
        logger.info('Consistent. Here is queue.')
        h.print_structure()

    while not len(h) == 0:
        h.pop_Q()
        if not h.is_consistent():
            logger.info('Inconsistent')
            break

    logger.info('Emptied the queue')

    logger.info('Running special inserts to trigger more percolate up calls in remove.')
    v = [
        lca_lite(123, 19),
        lca_lite(459, 16),
        lca_lite(628, 6),
        lca_lite(747, 13),
        lca_lite(827, 11),
        lca_lite(389, 4),
        lca_lite(456, 2),
        lca_lite(277, 12),
        lca_lite(648, 8),
    ]
    remove0 = v[5]
    remove1 = v[6]
    for a in v:
        h.insert(a)
        if not h.is_consistent():
            logger.info('Inconsistent during insert')
            break
    h.remove(remove0)
    h.remove(remove1)
    if not h.is_consistent():
        logger.info('Inconsistent during remove')
    else:
        logger.info('All consistent during the remove examples')

    h.clear()
    logger.info('Running a bunch more inserts and removes.')
    v = [
        lca_lite(123, 1.0),
        lca_lite(459, 9.4),
        lca_lite(628, 8.2),
        lca_lite(747, 8.7),
        lca_lite(827, 7.8),
        lca_lite(389, 8.9),
        lca_lite(456, 5.3),
        lca_lite(277, 6.7),
        lca_lite(648, 8.6),
        lca_lite(723, 9.9),
        lca_lite(823, 2.3),
        lca_lite(234, 6.5),
    ]

    error_found = False
    for a in v:
        h.insert(a)
        if not h.is_consistent():
            logger.info('Inconsistent during insert')
            error_found = True
            break
    if not error_found:
        logger.info('No errors')

    error_found = False
    for a in v:
        h.remove(a)
        if not h.is_consistent():
            logger.info('Inconsistent during remove')
            error_found = True
            break
    if not error_found:
        logger.info('All consistent.  No errors')


if __name__ == '__main__':
    test_lca_heap()
