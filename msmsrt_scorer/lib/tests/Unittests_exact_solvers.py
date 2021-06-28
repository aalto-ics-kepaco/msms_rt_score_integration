####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import unittest
import numpy as np
import itertools as it

from collections import OrderedDict

from msmsrt_scorer.lib.exact_solvers import ChainFactorGraph, RetentionTimeTreeFactorGraph, RandomTreeFactorGraph
from msmsrt_scorer.lib.data_utils import sigmoid


def generate_random_example(random_seed=None, n_spec=None):
    """
    Generator for a random data for (MS, RT)-setting.

    :param random_seed: scalar, used as random seed
    :return:
    """
    rs = np.random.RandomState(random_seed)

    # SET UP DATA: Retention time and MSMS setup
    if n_spec is None:
        n_spec = rs.randint(4, 11)

    candidates = OrderedDict()
    for i in range(n_spec):
        candidates[i] = {}
        candidates[i]["n_cand"] = rs.randint(3, 6)

        # candidate MS2-scores
        candidates[i]["score"] = rs.rand(candidates[i]["n_cand"])
        candidates[i]["score"] /= np.max(candidates[i]["score"])
        candidates[i]["log_score"] = np.log(candidates[i]["score"])

    # Retention times
    candidates[0]["retention_time"] = rs.rand() / 5
    for i in range(1, n_spec):
        candidates[i]["retention_time"] = candidates[i - 1]["retention_time"] + (rs.rand() * 4)

    # Candidate RT-scores
    for i in range(n_spec):
        candidates[i]["pref_score"] = rs.randn(candidates[i]["n_cand"]) + i

    return candidates


class MacKayExample(object):
    """
    Example is taken from the book by [1].

    [1] MacKay, D. J., "Information theory, inference and learning algorithms", Cambridge university press (2003)
    """
    def __init__(self, f_0, f_1, f_2, f_01, f_12, D=0.5):
        self.f_0 = f_0
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_01 = f_01
        self.f_12 = f_12
        self.D = D

        self.lhs = self.likelihood_for_all_Zs()
        self.candidates, self.prefscores = self.prepare_cand_and_pref_dicts()

    def likelihood(self, Z, log=False):
        lh_ms = self.f_0[Z[0]] * self.f_1[Z[1]] * self.f_2[Z[2]]
        lh_rt = self.f_01[Z[0], Z[1]] * self.f_12[Z[1], Z[2]]

        lh = lh_ms ** (1 - self.D) * lh_rt ** self.D
        if log:
            lh = np.log(lh)

        return lh

    def MAP(self):
        return np.unravel_index(np.argmax(self.lhs), shape=self.lhs.shape), np.max(self.lhs)

    def likelihood_for_all_Zs(self, log=False):
        lhs = np.full((2, 2, 2), fill_value=np.nan)
        for z in it.product([0, 1], [0, 1], [0, 1]):
            lhs[z] = self.likelihood(z, log)
        return lhs

    def get_sum_marginals(self, normalize=True):
        marginals = OrderedDict()

        for i in range(3):
            marg = np.sum(self.lhs, axis=tuple(j for j in range(3) if j != i))

            if normalize:
                marg /= np.sum(marg)

            marginals[i] = marg

        return marginals

    def get_max_marginals(self, normalize=True):
        marginals = OrderedDict()

        for i in range(3):
            marg = np.max(self.lhs, axis=tuple(j for j in range(3) if j != i))

            if normalize:
                marg /= np.max(marg)

            marginals[i] = marg

        return marginals

    def prepare_cand_and_pref_dicts(self):
        f_10 = self.f_01
        f_21 = self.f_12

        candidates = OrderedDict()
        candidates[0] = {"n_cand": 2, "score": self.f_0, "log_score": np.log(self.f_0), "retention_time": 0.0}
        candidates[1] = {"n_cand": 2, "score": self.f_1, "log_score": np.log(self.f_1), "retention_time": 2.0}
        candidates[2] = {"n_cand": 2, "score": self.f_2, "log_score": np.log(self.f_2), "retention_time": 2.1}

        prefscores = OrderedDict()
        prefscores[0] = OrderedDict()
        prefscores[1] = OrderedDict()
        prefscores[2] = OrderedDict()
        prefscores[0][1] = {"score": self.f_01, "log_score": np.log(self.f_01)}
        prefscores[1][0] = {"score": f_10, "log_score": np.log(f_10)}
        prefscores[1][2] = {"score": self.f_12, "log_score": np.log(self.f_12)}
        prefscores[2][1] = {"score": f_21, "log_score": np.log(f_21)}

        return candidates, prefscores


class TestChainFactorGraph(unittest.TestCase):
    def test_sum_product__simple_data(self):
        candidates = generate_random_example(21, 3)

        CFG = ChainFactorGraph(candidates, use_log_space=False,
                               make_order_probs=lambda _x, loc: sigmoid(_x)).sum_product()
        R, Q = CFG.R_sum, CFG.Q_sum

        # == FORWARD ==
        D = CFG.D_rt
        # R-Messages passed from the leaf nodes
        np.testing.assert_equal(R[(0, 0)][0], candidates[0]["score"] ** (1.0 - D))
        np.testing.assert_equal(R[(1, 1)][1], candidates[1]["score"] ** (1.0 - D))
        np.testing.assert_equal(R[(2, 2)][2], candidates[2]["score"] ** (1.0 - D))

        # R-Messages
        for s in range(candidates[1]["n_cand"]):
            np.testing.assert_allclose(R[(0, 1)][1][s], (CFG.order_probs[0][1]["score"][:, s] ** D) @ Q[0][(0, 1)])
        for s in range(candidates[2]["n_cand"]):
            np.testing.assert_allclose(R[(1, 2)][2][s], (CFG.order_probs[1][2]["score"][:, s] ** D) @ Q[1][(1, 2)])

        # Q-Messages
        np.testing.assert_equal(Q[0][(0, 1)], R[(0, 0)][0])
        np.testing.assert_equal(Q[1][(1, 2)], R[(1, 1)][1] * R[(0, 1)][1])

        # == BACKWARD ==
        # R-Messages
        for s in range(candidates[1]["n_cand"]):
            np.testing.assert_allclose(R[(1, 2)][1][s], (CFG.order_probs[1][2]["score"][s, :] ** D) @ Q[2][(1, 2)])
        for s in range(candidates[0]["n_cand"]):
            np.testing.assert_allclose(R[(0, 1)][0][s], (CFG.order_probs[0][1]["score"][s, :] ** D) @ Q[1][(0, 1)])

        # Q-Messages
        np.testing.assert_equal(Q[2][(2, 2)], R[(1, 2)][2])
        np.testing.assert_equal(Q[2][(1, 2)], R[(2, 2)][2])
        np.testing.assert_equal(Q[1][(1, 1)], R[(1, 2)][1] * R[(0, 1)][1])
        np.testing.assert_equal(Q[1][(0, 1)], R[(1, 1)][1] * R[(1, 2)][1])
        np.testing.assert_equal(Q[0][(0, 0)], R[(0, 1)][0])

        # Marginalization
        def marginal(i=None, normalize=True):
            """ Marginals of all possible assignments of variable i """
            p = np.full(shape=(candidates[0]["n_cand"],
                               candidates[1]["n_cand"],
                               candidates[2]["n_cand"]), fill_value=np.nan)

            for x in it.product(range(candidates[0]["n_cand"]),
                                range(candidates[1]["n_cand"]),
                                range(candidates[2]["n_cand"])):

                p[x] = CFG.likelihood(x, log=False)

            marg = np.sum(p, axis=tuple(d for d in range(3) if d != i))
            # marg /= np.max(marg)

            if normalize:
                marg /= np.sum(marg)

            return marg

        marg_unnorm = CFG.get_sum_marginals(normalize=False)
        marg_norm = CFG.get_sum_marginals(normalize=True)
        for i in range(3):
            np.testing.assert_allclose(marg_unnorm[i], marginal(i, False))
            np.testing.assert_allclose(marg_norm[i], marginal(i, True))

    def test_marginals__mckay_example(self):
        # EXAMPLE: McKay textbook page 334
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.1, 0.9])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)
        f_12 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        marg_unrm = ChainFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                     order_probs=mckay.prefscores).sum_product().get_sum_marginals(False)
        marg_norm = ChainFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                     order_probs=mckay.prefscores).sum_product().get_sum_marginals(True)
        for i in range(3):
            np.testing.assert_allclose(marg_unrm[i], mckay.get_sum_marginals(False)[i])
            np.testing.assert_allclose(marg_norm[i], mckay.get_sum_marginals(True)[i])

        # -----------------------------------------------
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.9, 0.1])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)
        f_12 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        marg_unrm = ChainFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                     order_probs=mckay.prefscores).sum_product().get_sum_marginals(False)
        marg_norm = ChainFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                     order_probs=mckay.prefscores).sum_product().get_sum_marginals(True)
        for i in range(3):
            np.testing.assert_allclose(marg_unrm[i], mckay.get_sum_marginals(False)[i])
            np.testing.assert_allclose(marg_norm[i], mckay.get_sum_marginals(True)[i])

    def test_MAP__mckay_example(self):
        # EXAMPLE: McKay textbook page 334
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.1, 0.9])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)
        f_12 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        Z_max, p_max = ChainFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                        order_probs=mckay.prefscores).max_product().MAP()
        Z_max_ref, p_max_ref = mckay.MAP()
        np.testing.assert_equal(Z_max, Z_max_ref)
        np.testing.assert_equal(np.exp(p_max), p_max_ref)

        # -----------------------------------------------
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.9, 0.1])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)
        f_12 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        Z_max, p_max = ChainFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                        order_probs=mckay.prefscores).max_product().MAP()
        Z_max_ref, p_max_ref = mckay.MAP()
        np.testing.assert_equal(Z_max, Z_max_ref)
        np.testing.assert_equal(np.exp(p_max), p_max_ref)


class TestTreeFactorGraph(unittest.TestCase):
    def test_print_function(self):
        candidates = generate_random_example(2387, n_spec=100)

        # Based on retention times
        print("== Retention time ==")
        print(RetentionTimeTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), min_rt_diff=0.0))
        print(RetentionTimeTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), min_rt_diff=0.5))

        # Random Tree
        print("== Random ==")
        for rep in range(5):
            print(RandomTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep))


class TestRetentionTimeTreeFactorGraph(unittest.TestCase):
    def test_marginals__mckay_example(self):
        # EXAMPLE: MacKay textbook page 334
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.1, 0.9])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)
        f_12 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        marg_unrm = RetentionTimeTreeFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                                 order_probs=mckay.prefscores).sum_product().get_sum_marginals(False)
        marg_norm = RetentionTimeTreeFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                                 order_probs=mckay.prefscores).sum_product().get_sum_marginals(True)
        for i in range(3):
            np.testing.assert_allclose(marg_unrm[i], mckay.get_sum_marginals(False)[i])
            np.testing.assert_allclose(marg_norm[i], mckay.get_sum_marginals(True)[i])

        # -----------------------------------------------
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.9, 0.1])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)
        f_12 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        marg_unrm = RetentionTimeTreeFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                                 order_probs=mckay.prefscores).sum_product().get_sum_marginals(False)
        marg_norm = RetentionTimeTreeFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                                 order_probs=mckay.prefscores).sum_product().get_sum_marginals(True)
        for i in range(3):
            np.testing.assert_allclose(marg_unrm[i], mckay.get_sum_marginals(False)[i])
            np.testing.assert_allclose(marg_norm[i], mckay.get_sum_marginals(True)[i])

    def test_MAP__mckay_example(self):
        # EXAMPLE: McKay textbook page 334
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.1, 0.9])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)
        f_12 = np.array([[0.9999, 0.0001], [0.0001, 0.9999]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        Z_max, p_max = RetentionTimeTreeFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                                    order_probs=mckay.prefscores).max_product().MAP()
        Z_max_ref, p_max_ref = mckay.MAP()
        np.testing.assert_equal(Z_max, Z_max_ref)
        np.testing.assert_equal(np.exp(p_max), p_max_ref)

        # -----------------------------------------------
        f_0 = np.array([0.1, 0.9])
        f_1 = np.array([0.9, 0.1])
        f_2 = np.array([0.1, 0.9])
        f_01 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)
        f_12 = np.array([[0.0001, 0.9999], [0.9999, 0.0001]], dtype=float)

        # Sum-product
        mckay = MacKayExample(f_0, f_1, f_2, f_01, f_12)
        Z_max, p_max = RetentionTimeTreeFactorGraph(mckay.candidates, make_order_probs=sigmoid,
                                                    order_probs=mckay.prefscores).max_product().MAP()
        Z_max_ref, p_max_ref = mckay.MAP()
        np.testing.assert_equal(Z_max, Z_max_ref)
        np.testing.assert_equal(np.exp(p_max), p_max_ref)

    def test_MAP__random_data(self):
        for rep in range(25):
            candidates = generate_random_example(rep)

            for min_rt_diff in [0.0, 0.1, 1.0, 1.5]:
                TFG = RetentionTimeTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x),
                                                   min_rt_diff=min_rt_diff)

                # Set up probability for all possible Zs
                P = np.full([cnd["n_cand"] for cnd in candidates.values()], fill_value=np.nan)
                for Z in it.product(*[range(cnd["n_cand"]) for cnd in candidates.values()]):
                    P[Z] = TFG.likelihood(Z, TFG.use_log_space)

                Z_max, p_max = TFG.max_product().MAP()
                np.testing.assert_equal(np.unravel_index(np.argmax(P), P.shape), Z_max)
                np.testing.assert_allclose(np.max(P), p_max)

    def test_marginals__random_data(self):
        def marginal(P, i, normalize=True):
            marg = np.sum(P, axis=tuple(j for j in candidates if j != i))
            if normalize:
                marg /= np.sum(marg)
            return marg

        for rep in range(25):
            candidates = generate_random_example(rep)

            for min_rt_diff in [0.0, 0.1, 1.0, 1.5]:
                TFG = RetentionTimeTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x),
                                                   min_rt_diff=min_rt_diff)

                # Set up probability for all possible Zs
                P = np.full([cnd["n_cand"] for cnd in candidates.values()], fill_value=np.nan)
                for Z in it.product(*[range(cnd["n_cand"]) for cnd in candidates.values()]):
                    P[Z] = TFG.likelihood(Z, log=False)

                marg_norm = TFG.sum_product().get_sum_marginals(normalize=True)
                marg_unrm = TFG.sum_product().get_sum_marginals(normalize=False)

                for i in candidates:
                    np.testing.assert_allclose(marg_norm[i], marginal(P, i, normalize=True))
                    np.testing.assert_allclose(marg_unrm[i], marginal(P, i, normalize=False))

    def test_marginals__random_data__varying_D(self):
        def marginal(P, i, normalize=True):
            marg = np.sum(P, axis=tuple(j for j in candidates if j != i))
            if normalize:
                marg /= np.sum(marg)
            return marg

        for rep in range(10):
            candidates = generate_random_example(rep + 100)

            for D in [0.0, 0.25, 0.75, 1.0]:
                TFG = RetentionTimeTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), D=D,
                                                   min_rt_diff=1.0)

                # Set up probability for all possible Zs
                P = np.full([cnd["n_cand"] for cnd in candidates.values()], fill_value=np.nan)
                for Z in it.product(*[range(cnd["n_cand"]) for cnd in candidates.values()]):
                    P[Z] = TFG.likelihood(Z, log=False)

                marg_norm = TFG.sum_product().get_sum_marginals(normalize=True)
                marg_unrm = TFG.sum_product().get_sum_marginals(normalize=False)

                for i in candidates:
                    np.testing.assert_allclose(marg_norm[i], marginal(P, i, normalize=True))
                    np.testing.assert_allclose(marg_unrm[i], marginal(P, i, normalize=False))

    def test_max_marginals__random_data(self):
        def max_marginal(P, i, normalize=True):
            marg = np.max(P, axis=tuple(j for j in candidates if j != i))
            if normalize:
                marg /= np.max(marg)
            return marg

        for rep in range(25):
            candidates = generate_random_example(rep)

            for min_rt_diff in [0.0, 0.1, 1.0, 1.5]:
                TFG = RetentionTimeTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x),
                                                   min_rt_diff=min_rt_diff)

                # Set up probability for all possible Zs
                P = np.full([cnd["n_cand"] for cnd in candidates.values()], fill_value=np.nan)
                for Z in it.product(*[range(cnd["n_cand"]) for cnd in candidates.values()]):
                    P[Z] = TFG.likelihood(Z, log=False)

                marg_norm = TFG.max_product().get_max_marginals(normalize=True)
                marg_unrm = TFG.max_product().get_max_marginals(normalize=False)

                for i in candidates:
                    np.testing.assert_allclose(marg_norm[i], max_marginal(P, i, normalize=True))
                    np.testing.assert_allclose(marg_unrm[i], max_marginal(P, i, normalize=False))


class TestRandomTreeFactorGraph(unittest.TestCase):
    def test_normalization_constant_Z(self):
        candidates = generate_random_example(1999)

        # Sum-Margin
        TFG = RandomTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=1)
        marg_unrm = TFG.sum_product().get_sum_marginals(normalize=False)

        for i in candidates:
            np.testing.assert_allclose(np.sum(marg_unrm[0]), np.sum(marg_unrm[i]))

        # Max-Margin
        TFG = RandomTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=1)
        marg_unrm = TFG.max_product().get_max_marginals(normalize=False)

        for i in candidates:
            np.testing.assert_allclose(np.max(marg_unrm[0]), np.max(marg_unrm[i]))

    def test_sum_marginals__random_data(self):
        def sum_marginal(P, i, normalize=True):
            marg = np.sum(P, axis=tuple(j for j in candidates if j != i))
            if normalize:
                marg /= np.sum(marg)
            return marg

        for rep in range(25):
            candidates = generate_random_example(rep)

            TFG = RandomTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep)

            # Set up probability for all possible Zs
            P = np.full([cnd["n_cand"] for cnd in candidates.values()], fill_value=np.nan)
            for Z in it.product(*[range(cnd["n_cand"]) for cnd in candidates.values()]):
                P[Z] = TFG.likelihood(Z, log=False)

            marg_norm = TFG.sum_product().get_sum_marginals(normalize=True)
            marg_unrm = TFG.sum_product().get_sum_marginals(normalize=False)

            for i in candidates:
                np.testing.assert_allclose(marg_norm[i], sum_marginal(P, i, normalize=True))
                np.testing.assert_allclose(marg_unrm[i], sum_marginal(P, i, normalize=False))

    def test_max_marginals__random_data(self):
        def max_marginal(P, i, normalize=True):
            marg = np.max(P, axis=tuple(j for j in candidates if j != i))
            if normalize:
                marg /= np.max(marg)
            return marg

        for rep in range(25):
            candidates = generate_random_example(rep)

            TFG = RandomTreeFactorGraph(candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep)

            # Set up probability for all possible Zs
            P = np.full([cnd["n_cand"] for cnd in candidates.values()], fill_value=np.nan)
            for Z in it.product(*[range(cnd["n_cand"]) for cnd in candidates.values()]):
                P[Z] = TFG.likelihood(Z, log=False)

            marg_norm = TFG.max_product().get_max_marginals(normalize=True)
            marg_unrm = TFG.max_product().get_max_marginals(normalize=False)

            for i in candidates:
                np.testing.assert_allclose(marg_norm[i], max_marginal(P, i, normalize=True))
                np.testing.assert_allclose(marg_unrm[i], max_marginal(P, i, normalize=False))

    def test_MAP(self):
        for rep in range(25):
            candidates = generate_random_example(rep)

            TFG = RandomTreeFactorGraph(
                candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep, use_log_space=True
            )

            Z_max_dp, p_max_dp = TFG.MAP_only()
            Z_max_bf, p_max_bf = TFG.MAP_only__brute_force()

            self.assertEqual(tuple(Z_max_bf), tuple(Z_max_dp))
            np.testing.assert_allclose(p_max_bf, p_max_dp)

    def test_only_rt_different_from_only_ms(self):
        for rep in range(25):
            candidates = generate_random_example(rep + 1)

            TFG_rt = RandomTreeFactorGraph(
                candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep + 1, use_log_space=True, D=1
            )

            TFG_ms = RandomTreeFactorGraph(
                candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep + 1, use_log_space=True, D=0
            )

            Z_max_rt, p_max_rt = TFG_rt.MAP_only()
            Z_max_ms, p_max_ms = TFG_ms.MAP_only()

            self.assertNotEqual(tuple(Z_max_ms), tuple(Z_max_rt))
            self.assertNotEqual(p_max_ms, p_max_rt)

    def test_only_ms_inference(self):
        for rep in range(25):
            candidates = generate_random_example(rep + 2)

            TFG_ms = RandomTreeFactorGraph(
                candidates, make_order_probs=lambda _x, loc: sigmoid(_x), random_state=rep + 2, use_log_space=True, D=0
            )

            Z_max_ms, _ = TFG_ms.MAP_only()

            for i in candidates:
                z_i_max = np.argmax(candidates[i]["score"])
                self.assertEqual(z_i_max, Z_max_ms[i])


if __name__ == '__main__':
    unittest.main()
