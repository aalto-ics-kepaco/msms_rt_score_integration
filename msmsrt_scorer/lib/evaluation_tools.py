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

import numpy as np
import itertools as it
import pandas as pd

from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score

from msmsrt_scorer.lib.data_utils import sigmoid, hinge_sigmoid, step_fun
from msmsrt_scorer.lib.exact_solvers import RetentionTimeTreeFactorGraph, RandomTreeFactorGraph


def run_parameter_grid(candidates, h_param_grid, tree_method, n_trees, n_jobs, make_order_prob, norm_order_scores,
                       margin_type):
    """
    Run 'get_marginals' for a grid of (D, k)-tuples and multiple tree approximations in parallel

    :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

    :param h_param_grid: dictionary of lists, (D, k)-tuple grid. The output format is defined by the scikit-learn
        function 'model_selection.ParameterGrid'.

    :param tree_method: string, which tree approximation should be used: 'random' spanning tree or 'chain' graph. See
        Section 2.3.2

    :param n_trees: scalar, number of trees to run

    :param n_jobs: scalar, number of parallel jobs

    :param make_order_prob: string, which edge potential function should be used.

    :param norm_order_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one.

    :param margin_type: string, which margin type should be used: 'max' or 'sum'. See section 2.3

    :return: tuple (
        res: list of 'get_marginals' outputs
        candidates: pass through
        h_param_grid: pass through
        n_trees: pass through
    )
    """
    # Run Forward-Backward algorithm for each spanning tree and parameter tuple (in parallel)
    res = Parallel(n_jobs=n_jobs)(
        delayed(get_marginals)(candidates, D=params["D"], k=params["k"], tree_method=tree_method, rep=rep,
                               normalize=False, make_order_prob=make_order_prob, norm_order_scores=norm_order_scores,
                               margin_type=margin_type)
        for params, rep in it.product(h_param_grid, range(n_trees)))

    return res, candidates, h_param_grid, n_trees


def get_marginals(candidates, D, k, tree_method, rep, make_order_prob, normalize=True, norm_order_scores=False,
                  margin_type="max"):
    """
    Find the marginals for the given ms-feature set (in the 'candidates' dictionary).

    :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

    :param D: scalar, weight on the retention order information

    :param k: scalar, sigmoid parameter k

    :param tree_method: string, which tree approximation should be used: 'random' spanning tree or 'chain' graph. See
        Section 2.3.2

    :param rep: scalar, index of the tree approximation. For example, if a random tree ensemble is used, than this value
        indexes the different random trees.

    :param make_order_prob: string, which edge potential function should be used.

    :param normalize: boolean, indicating whether normalized marginals should be returned

    :param norm_order_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one.

    :param margin_type: string, which margin type should be used: 'max' or 'sum'. See section 2.3

    :return: tuple (
        (D, k): hyper parameters that where used, i.e. retention order weight and sigmoid parameter
        rep: index of the tree approximation
        marg: OrderedDict, containing the marginals (values) for all ms-features i (keys)
        Z_max: if margin_type is 'max' MAP candidate assignment, otherwise None
        p_max: if margin_type is 'max' likelihood value of the MAP candidate assignment
    )
    """

    # Set up order probability function, i.e. bind the sigmoid k parameter to the function call.
    if make_order_prob == "sigmoid":
        def _make_order_prob(x, loc):
            return sigmoid(x, k=k)

    elif make_order_prob == "stepfun":
        def _make_order_prob(x, loc):
            return step_fun(x)

    elif make_order_prob == "hinge_sigmoid":
        def _make_order_prob(x, loc):
            return hinge_sigmoid(x, k=k)

    else:
        raise ValueError("Invalid order probability function: '%s'" % make_order_prob)

    # Set up tree factor graph model
    if tree_method == "random":
        TFG = RandomTreeFactorGraph(candidates, D=D, make_order_probs=_make_order_prob,
                                    random_state=rep, norm_order_scores=norm_order_scores,
                                    remove_edges_with_zero_rt_diff=True)

    elif tree_method == "chain":
        TFG = RetentionTimeTreeFactorGraph(candidates, D=D, make_order_probs=_make_order_prob, min_rt_diff=0.0,
                                           norm_order_scores=norm_order_scores)

    else:
        raise ValueError("Invalid tree method: '%s'" % tree_method)

    # Find the marginals
    if margin_type == "max":
        TFG.max_product()  # Forward-backward algorithm
        marg = TFG.get_max_marginals(normalize=normalize)  # Recover the max-marginals
        Z_max, p_max = TFG.MAP()  # Recover the MAP-estimate
    elif margin_type == "sum":
        TFG.sum_product()  # Forward-backward algorithm
        marg = TFG.get_marginals(normalize=normalize)
        Z_max, p_max = None, -1
    else:
        raise ValueError("Invalid margin-type: '%s'" % margin_type)

    return (D, k), rep, marg, Z_max, p_max


def evaluate_parameter_grid(res, candidates, h_param_grid, n_random_trees):
    """
    Evaluate different performance measures for the (D, k) grid tuples given the output 'run_parameter_grid'.

    :return: pandas.DataFrame, shape[0]=n_param_tuples, containing all the performance measures for each hyper parameter
        combinations
    """
    def _unit_basis_vector(n_cand, r):
        e = np.zeros(n_cand)
        e[r] = 1.0
        return e

    # Collect and aggregate "ranking" measures from the spanning tree ensemble for each parameter tuple
    p_marg = []  # Sum of log(p_i), where p_i is the (average) marginal probabilities of the correct candidates
    un_p_marg = []  # Sum of log(p_i), where p_i is the (average) marginal probabilities of the correct candidates
    p_max = []  # Maximum a posteriori of the most likely candidate assignment
    marg = []  # Posterior max-marginals for all candidate sets
    n_marg = []  # Normalized posterior max-marginals for all candidate sets
    top1, top3, top5, top10, top20 = [], [], [], [], []  # top-k accuracies
    topk_auc = []  # Area-under-the-Top20 curve, Sum of top-1, top-2, ..., top-20 (counts) divided by (20 * n_ms2)
    un_topk_auc = []  # Area-under-the-Top20 curve, Sum of top-1, top-2, ..., top-20 (counts) divided by (20 * n_ms2)
    ndcg = []  # Normalized Discounted Cumulative Gain (NDCG) for k=20, a ranking measure
    l_param_D = []
    l_param_k = []

    # Process each parameter tuple
    for idx_param, params in enumerate(h_param_grid):
        D, k = params["D"], params["k"]
        l_param_D.append(D)
        l_param_k.append(k)

        p_marg.append(np.zeros(shape=(len(candidates),)))
        un_p_marg.append(np.zeros(shape=(len(candidates),)))
        p_max.append(0.0)
        marg.append({k: np.zeros(v["n_cand"]) for k, v in candidates.items()})
        n_marg.append({k: np.zeros(v["n_cand"]) for k, v in candidates.items()})
        ndcg.append(0.0)

        # Aggregate over the spanning-tree ensemble
        for rep in range(n_random_trees):
            _res = res[idx_param * n_random_trees + rep]  # access results for given parameter tuple and spanning tree
            assert (_res[0] == (D, k))
            assert (_res[1] == rep)

            # Sum up map-probabilities
            p_max[-1] += _res[4]

            for i in candidates:
                # Sum up marginal-probabilities of correct candidates (here we normalize to sum one)
                p_marg[-1][i] += (_res[2][i][candidates[i]["index_of_correct_structure"]] / np.sum(_res[2][i]))
                # Sum up marginal-probabilities of correct candidates
                un_p_marg[-1][i] += _res[2][i][candidates[i]["index_of_correct_structure"]]
                # Sum up marginal-probabilities
                marg[-1][i] += _res[2][i]
                # Sum up normalized marginal-probabilities
                n_marg[-1][i] += (_res[2][i] / np.sum(_res[2][i]))

        # Average map-probabilities
        p_max[-1] /= n_random_trees

        # Average marginals
        for i in candidates:
            marg[-1][i] /= n_random_trees
            n_marg[-1][i] /= n_random_trees

        # Sum up log-probabilities of correct candidate across all candidate sets
        p_marg[-1] = np.sum(np.log(p_marg[-1] / n_random_trees))
        un_p_marg[-1] = np.sum(np.log(un_p_marg[-1] / n_random_trees))

        # NDCG
        for i in candidates:
            if candidates[i]["n_cand"] == 1:
                ndcg[-1] += 1.0
            else:
                _true_relevance = _unit_basis_vector(candidates[i]["n_cand"],
                                                     candidates[i]["index_of_correct_structure"])
                _pred_relevance = marg[-1][i]
                _ndcg = ndcg_score(np.atleast_2d(_true_relevance), np.atleast_2d(_pred_relevance), k=20)
                ndcg[-1] += _ndcg
        ndcg[-1] /= len(candidates)

        # Calculate top-k accuracies
        _topk = get_topk_performance_from_scores(candidates, n_marg[-1], method="casmi2016")
        topk_auc.append(np.sum(_topk[0][:20]) / (20 * len(candidates)))
        top1.append(_topk[1][0])
        top3.append(_topk[1][2])
        top5.append(_topk[1][4])
        top10.append(_topk[1][9])
        top20.append(_topk[1][19])

        _un_topk = get_topk_performance_from_scores(candidates, marg[-1], method="casmi2016")
        un_topk_auc.append(np.sum(_un_topk[0][:20]) / (20 * len(candidates)))

    df = pd.DataFrame(data={
        "D": l_param_D, "k": l_param_k,
        "p_marg": p_marg, "un_p_marg": un_p_marg, "p_max": p_max,
        "topk_auc": topk_auc, "un_topk_auc": un_topk_auc,
        "top1": top1, "top3": top3, "top5": top5, "top10": top10, "top20": top20,
        "ndcg": ndcg})

    return df


def get_topk_performance_from_scores(candidates, scores=None, method="csifingerid"):
    """
    Given a sets candidates (and scores), calculate the ranks of the correct molecular structures within the candidate
    sets. Two methods for the rank calculate are implemented
     - "csifingerid" as used by [1]
     - "casmi2016" as used by [2] (and for the evaluation of our method, see Section 3.4)

    Lowest rank (corresponding to the largest score) is zero.

    [1] Dührkop et al., "Searching molecular structure databases with tandem mass spectra using CSI:FingerID",
        Proceedings of the National Academy of Sciences (PNAS) (2015)
    [2] Schymanski et al., "Critical Assessment of Small Molecule Identification 2016: automated methods",
        Journal of Cheminformatics (2017)

    :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
        candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

        See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
        expected here.

    :param scores: dict or None,
        if dictionary, the keys must correspond to the keys in candidates and values must be array-like score vectors.
        if None, the score vectors are taken from the candidates dictionary (key = 'score')

    :param method: string, indicating which rak calculation method should be used.

    :return: tuple (
        array-like, number of correct examples ranked at rank <= k (where k is the index in the array)
        array-like, percentage of correct examples, ranked at rank <= k
    )
    """
    if method == "csifingerid":
        topk_n, topk_acc = get_topk_performance_csifingerid(candidates, scores)
    elif method == "casmi2016":
        topk_n, topk_acc = get_topk_performance_casmi2016(candidates, scores)
    else:
        raise ValueError("Invalid ranking method: '%s'" % method)

    return topk_n, topk_acc


def get_topk_performance_csifingerid(candidates, scores=None):
    """
    Topk performance calculation after [1]. (see 'get_topk_performance_from_scores')
    """
    cscnt = np.zeros((np.max([cnd["n_cand"] for cnd in candidates.values()]) + 2,))

    for i in candidates:
        # If the correct candidate is not in the set
        if np.isnan(candidates[i]["index_of_correct_structure"]):
            continue

        if scores is None:
            # Use ranking based on the candidate scores
            _scores = - candidates[i]["score"]
        else:
            # Use ranking based on the marginal scores after MS and RT integration
            _scores = - scores[i]

        # Calculate ranks
        _ranks = rankdata(_scores, method="ordinal") - 1

        # Get the contribution of the correct candidate
        _s = _scores[candidates[i]["index_of_correct_structure"]]
        _c = (1. / np.sum(_scores == _s)).item()
        _r = _ranks[_scores == _s]

        # For all candidate with the same score, we update their corresponding ranks with their contribution
        cscnt[_r] += _c

    cscnt = np.cumsum(cscnt)

    return cscnt, cscnt / len(candidates) * 100


def get_topk_performance_casmi2016(candidates, scores=None):
    """
    Topk performance calculation after [2]. (see 'get_topk_performance_from_scores')
    """
    cscnt = np.zeros((np.max([cnd["n_cand"] for cnd in candidates.values()]) + 2,))

    for i in candidates:
        # If the correct candidate is not in the set
        if np.isnan(candidates[i]["index_of_correct_structure"]):
            continue

        if scores is None:
            # Use ranking based on the candidate scores
            _scores = - candidates[i]["score"]
        else:
            # Use ranking based on the marginal scores after MS and RT integration
            _scores = - scores[i]

        # Calculate ranks
        _ranks = np.ceil(rankdata(_scores, method="average") - 1).astype("int")

        # Get the contribution of the correct candidate
        _c = 1.0
        _r = _ranks[candidates[i]["index_of_correct_structure"]]

        # For all candidate with the same score, we update their corresponding ranks with their contribution
        cscnt[_r] += _c

    cscnt = np.cumsum(cscnt)

    return cscnt, cscnt / len(candidates) * 100


def _get_rank_and_contribution_of_correct_candidate(scores, index_of_correct_structure, method="csifingerid"):
    """
    Returns the rank of the correct molecular structure given a set of score and a rank calculation method.

    See 'get_topk_performance_from_scores'
    """
    # Account for order in which 'rankdata' ranks the elements
    _scores = - scores

    if method == "csifingerid":
        # Calculate ranks
        ranks = rankdata(_scores, method="ordinal") - 1

        # Get the contribution of the correct candidate
        s = _scores[index_of_correct_structure]
        c = (1. / np.sum(_scores == s)).item()
        r = ranks[_scores == s]

    elif method == "casmi2016":
        # Calculate ranks
        _ranks = np.ceil(rankdata(_scores, method="average") - 1).astype("int")

        # Get the contribution of the correct candidate
        c = 1.0
        r = _ranks[index_of_correct_structure]

    else:
        raise ValueError("Invalid ranking method: '%s'" % method)

    return r, c
