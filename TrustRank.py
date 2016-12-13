#-*- coding: utf-8 -*-

# A TrustDistrust algorithm implementation.

import pickle
import time

import numpy as np
import scipy.sparse as sp


# This implementation based on article "Propagating Both Trust and Distrust
# with Target Differentiation for Combating Web Spam" from Proceedings of the
# Twenty-Fifth AAAI Conference on Artificial Intelligence (pp. 1293-1297).
#
# Notation (according to article):
#     t(p) -- T-rank of the page.
#     d(p) -- D-rank of the page.
#     good_set(p) -- set of good pages (s(p) in article) 0.0 or 1.0
#     bad_set(p) -- set of good pages (s(p) in article) 0.0 or 1.0
#

PARAMETERS_DEFAULT = {"alpha_d": 0.85,
                      "alpha_t": 0.85,
                      "beta": 0.5,
                      "precision": 10**-4,
                      "max_iterations": 50,
                      }

#-----------------------------------------------------------------------------


def trust_distrust_rank(graph_matrix,
                        labeled_data,
                        parameters=PARAMETERS_DEFAULT,
                        print_out=False):

    start_time = time.time()

    n = graph_matrix.shape[0]
    d = np.ones(n)
    t = np.ones(n)
    n_ones = np.ones(n)

    good_set = 1.0 * (labeled_data > 0) / (sum((labeled_data > 0)))
    bad_set = 1.0 * (labeled_data < 0) / (sum(labeled_data < 0))

    # Loading parameters.
    if "alpha_d" in parameters:
        alpha_d = parameters["alpha_d"]
    else:
        alpha_d = PARAMETERS_DEFAULT["alpha_d"]

    if "alpha_t" in parameters:
        alpha_t = parameters["alpha_t"]
    else:
        alpha_t = PARAMETERS_DEFAULT["alpha_t"]

    if "beta" in parameters:
        beta = parameters["beta"]
    else:
        beta = PARAMETERS_DEFAULT["beta"]

    if "precision" in parameters:
        precision = parameters["precision"]
    else:
        precision = PARAMETERS_DEFAULT["precision"]

    if "max_iterations" in parameters:
        max_iterations = parameters["max_iterations"]
    else:
        max_iterations = PARAMETERS_DEFAULT["max_iterations"]

    #d[0] = 1

    indegree = graph_matrix * n_ones
    outdegree = n_ones.T * graph_matrix

    # Replace zeros with ones (it's ok, because we won't use it).
    indegree[indegree == 0] = np.ones(sum(indegree == 0))
    outdegree[outdegree == 0] = np.ones(sum(outdegree == 0))

    for i in xrange(max_iterations):
        print "-" * 80
        print "Iteration", i, "started"

        d_old = d
        t_old = t

        denominator = (1 - beta) * d + beta * t
        zero_den_indexes = denominator == 0

        d_mult = 2 * np.ones(n)
        t_mult = 2 * np.ones(n)

        # Compute d_mult for non-zero indexes of denominator
        d_mult[~zero_den_indexes] = \
            np.divide(alpha_d * (1 - beta) * d[~zero_den_indexes],
                      denominator[~zero_den_indexes])

        d_mult[zero_den_indexes] = 0.5 * np.ones(sum(zero_den_indexes))

        # Compute t_mult for non-zero indexes of denominator
        t_mult[~zero_den_indexes] = \
            np.divide(alpha_t * beta * t[~zero_den_indexes],
                      denominator[~zero_den_indexes])

        t_mult[zero_den_indexes] = 0.5 * np.ones(sum(zero_den_indexes))

        d = d_mult * (graph_matrix * (d / indegree)) \
            + (1 - alpha_d) * good_set
        t = t_mult * ((t / outdegree) * graph_matrix) \
            + (1 - alpha_t) * bad_set

        d_dist = np.linalg.norm(d_old - d, 1)
        t_dist = np.linalg.norm(t_old - t, 1)

        if print_out:
            print "D-Rank distance: %.3f" % d_dist
            print "T-Rank distance: %.3f" % t_dist

            print "Big D-rank on spamers: %.2f%%" % \
                (100.0 * sum(d[labeled_data < 0] > 0.1)
                    / sum(labeled_data < 0))
            print "Big T-rank on spamers: %.2f%%" % \
                (100.0 * sum(t[labeled_data < 0] > 0.1)
                    / sum(labeled_data < 0))

            print "Low D-rank on normal: %.2f%%" % \
                (100.0 * sum(d[labeled_data > 0] < 0.1)
                    / sum(labeled_data > 0))
            print "Low T-rank on normal: %.2f%%" % \
                (100.0 * sum(t[labeled_data > 0] < 0.1)
                    / sum(labeled_data > 0))

        print "-" * 80 + "\n"

        if (d_dist + t_dist) < n * precision:
            print "t and d were obtained with average precision", precision
            print "-" * 80 + "\n"
            break

    print i + 1, "iterations were needed to get solution with time",
    print "%.2f sec" % (time.time() - start_time)

    return d, t

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    # Notation (according to rdkl):
    #     n -- number of pages in collection.
    #     graph_matrix -- scipy sparse graph matrix (oriented).
    #
    # Load data and initialization.
    try:
        f = open("A.pkl")
        graph_matrix = pickle.load(f)
        f.close()
    except:
        print "Encountered problems with file A.pkl (can't load A from it)."

    n = graph_matrix.shape[0]

    # Names of files that contains manually labeled data.
    names = ["v900rand.pkl",
             "rtwitter.1.csv.pkl",
             "rtwitter.2.csv.pkl",
             "rtwitter.3.csv.pkl",
             "rtwitter.4.csv.pkl",
             ]

    labeled_data = np.zeros(n)

    for name in names:
        temp = np.zeros(n)
        try:
            temp[:] = pickle.load(open(name)).todense().flatten()
        except:
            print "Can not load file", name

        labeled_data += temp

    trust_distrust_rank(graph_matrix, labeled_data)
