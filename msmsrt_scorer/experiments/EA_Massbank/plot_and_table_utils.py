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

import os
import numpy as np
import pandas as pd
import gzip
import pickle
import itertools as it

from scipy.stats import ttest_1samp, ttest_rel, ttest_ind, wilcoxon
from seaborn.utils import ci, ci_to_errsize
from seaborn.algorithms import bootstrap


def _get_sample_id_string(sample_id):
    if sample_id is None:
        _sample_id = "*"  # wildcard to select all samples
    else:
        _sample_id = "%03d" % sample_id
        
    return _sample_id


def IDIR(base_dir="results", mode="development", make_order_prob="sigmoid",
         D_value_method="fixed", ion_mode="positive", participant="MetFrag_2.4.5__8afe4a14",
         max_n_cand=np.inf, sort_candidates_by_ms2_score=False, tree_method="random", n_random_trees=16,
         min_min_rt_diff=0.0, max_min_rt_diff=0.0, param_selection_measure=None, norm_order_scores=None,
         training_dataset="MEOH_AND_CASMI_JOINT", keep_test_molecules=False, estimator="ranksvm", 
         molecular_representation="substructure_count", use_global_parameter_selection=None, margin_type=None,
         norm_scores=None):
    
    # Define output directory
    if tree_method == "random":
        _tree_description = "random__n_trees=%d" % n_random_trees
    elif tree_method == "retention_time":
        _tree_description = "retention_time__min=%.2f_max=%.2f" % (min_min_rt_diff, max_min_rt_diff)
    elif tree_method == "chain":
        _tree_description = tree_method
    elif tree_method is None:
        pass
    else:
        raise ValueError("Invalid tree method: '%s'" % tree_method)

    # Get first level directory name
    _tmp = []
    if tree_method is not None:
        _tmp.append("tree_method=%s" % _tree_description)
    if make_order_prob is not None:
        _tmp.append("make_order_prob=%s" % make_order_prob)
    if D_value_method is not None:
        _tmp.append("D_value_method=%s" % D_value_method)
    if param_selection_measure is not None:
        _tmp.append("param_selection_measure=%s" % param_selection_measure)
    if use_global_parameter_selection is not None:
        _tmp.append("globparam=%d" % use_global_parameter_selection)
    if norm_scores is not None:
        _tmp.append("norm_scores=%s" % norm_scores)
    if norm_order_scores is not None:
        _tmp.append("norm_order_scores=%d" % norm_order_scores)
    if margin_type is not None:
        _tmp.append("mtype=%s" % margin_type)
        
    idir = os.path.join(
        base_dir, mode, 
        "__".join(_tmp),
        "__".join(["ion_mode=%s" % ion_mode,
                   "participant=%s" % participant,
                   "max_n_cand=%.0f" % max_n_cand,
                   "sort_candidates_by_ms2_score=%d" % sort_candidates_by_ms2_score]),
        "__".join(["trainset=%s" % training_dataset,
                   "keep_test=%d" % keep_test_molecules,
                   "est=%s" % estimator,
                   "mol_rep=%s" % molecular_representation]))
    
    return idir


def IDIR_METFRAG(base_dir="results", mode="development", pref_model="cdk", ion_mode="positive",
                 participant="MetFrag_2.4.5__8afe4a14", max_n_cand=np.inf, sort_candidates_by_ms2_score=False,
                 param_selection_measure="topk_auc"):

    idir = os.path.join(
        base_dir, mode,
        "param_selection_measure=%s" % param_selection_measure,
        "__".join(["ion_mode=%s" % ion_mode,
                   "participant=%s" % participant,
                   "max_n_cand=%.0f" % max_n_cand,
                   "pref_model=%s" % pref_model,
                   "sort_candidates_by_ms2_score=%d" % sort_candidates_by_ms2_score]))

    return idir


def TOPK(order_prob_k=1.0, D_value=0.5, max_n_ms2=75, sample_id=None, method="casmi"):
    _sample_id = _get_sample_id_string(sample_id)
    
    if (order_prob_k is None) or (D_value is None): 
        fn = "__".join(["topk_%s" % method,
                        "max_n_ms2=%03d" % max_n_ms2,
                        "sample_id=%s" % _sample_id]) + ".pkl.gz"
    else:
        fn = "__".join(["topk_%s" % method, 
                        "order_prob_k=%.2f" % order_prob_k,
                        "D_value=%.4f" % D_value,
                        "max_n_ms2=%03d" % max_n_ms2,
                        "sample_id=%s" % _sample_id]) + ".pkl.gz"
    return fn


def TOPK_MISSING_MS2(n_ms2, max_n_ms2, perc_ms2, sample_id=None, method="casmi"):
    _sample_id = _get_sample_id_string(sample_id)

    fn = "__".join(["topk_%s" % method,
                    "perc_ms2=%.1f" % perc_ms2,
                    "n_ms2=%03d_%03d" % (n_ms2, max_n_ms2),
                    "sample_id=%s" % _sample_id]) + ".pkl.gz"

    return fn


def MARG(order_prob_k=1.0, D_value=0.5, max_n_ms2=75, sample_id=None):
    """ 
    Remember: Marginals are in the 'marginals/' folder 
    """
    _sample_id = _get_sample_id_string(sample_id)

    if (order_prob_k is None) or (D_value is None):
        fn = "__".join(["marginals",
                        "max_n_ms2=%03d" % max_n_ms2,
                        "sample_id=%s" % _sample_id]) + ".pkl.gz"
    else:
        fn = "__".join(["marginals",
                        "order_prob_k=%.2f" % order_prob_k,
                        "D_value=%.4f" % D_value,
                        "max_n_ms2=%03d" % max_n_ms2,
                        "sample_id=%s" % _sample_id]) + ".pkl.gz"
    
    return fn


def CAND(max_n_ms2=75, sample_id=None):
    """ 
    Remember: Candidate information are in the 'candidates/' folder 
    """
    _sample_id = _get_sample_id_string(sample_id)
    
    fn = "__".join(["candidates", 
                    "max_n_ms2=%03d" % max_n_ms2,
                    "sample_id=%s" % _sample_id]) + ".pkl.gz"
        
    return fn


def load_results(idir, label, max_n_ms2, k_values_to_consider=[1, 3, 5, 10, 20], method="casmi", return_percentage=True,
                 label_only_ms="Only MS", n_samples=50, load_baseline=True, load_ms_rt=True):
    # Load top-k performance
    results = []
    
    for s in range(n_samples):
        fn = os.path.join(idir, TOPK(D_value=None, method=method, max_n_ms2=max_n_ms2, order_prob_k=None, sample_id=s))
        with gzip.open(fn) as file: 
            _topk = pickle.load(file)
            
        # Load MS + RT if required
        if load_ms_rt:
            results.append([s, label] + [_topk["ms_rt"][return_percentage][k - 1] for k in k_values_to_consider])     
        
        # Load Only MS if required
        if load_baseline:
            results.append(
                [s, label_only_ms] + [_topk["baseline"][return_percentage][k - 1] for k in k_values_to_consider])

    results = pd.DataFrame(results, columns=["sample", "Method"] + ["Top-%d" % k for k in k_values_to_consider])
    
    # Load selected parameters
    opt_params = pd.read_csv(os.path.join(idir, "opt_params.csv"))
    # opt_params = opt_params.loc[opt_params["sample"] < n_samples]
    opt_params["Method"] = label
    
    # Load parameter goodness measures for all samples
    param_goodness_measure = pd.read_csv(os.path.join(idir, "measures.csv"))
    param_goodness_measure["Method"] = label
        
    return results, opt_params, param_goodness_measure


def load_results_missing_ms2(
        idir, max_n_ms2,
        n_samples=50, k_values_to_consider=[1, 3, 5, 10, 20], method="casmi", return_percentage=True,
        load_baseline=True, label_baseline="Only MS",
        load_ms_rt=True, label_ms_rt="MS + RT",
        load_random=True, label_random="Random"):
    # Load top-k performance
    results = []

    for s, (n_ms2, perc_ms2) in it.product(range(n_samples),
                                           zip(np.floor(np.linspace(0, max_n_ms2, 5)).astype("int"), [0, 25, 50, 75, 100])):

        fn = os.path.join(idir, TOPK_MISSING_MS2(n_ms2, max_n_ms2, perc_ms2, sample_id=s, method=method))
        with gzip.open(fn) as file:
            _topk = pickle.load(file)

        # Load MS + RT if required
        if load_ms_rt:
            results.append([s, label_ms_rt, perc_ms2] +
                           [_topk["ms_rt"][return_percentage][k - 1] for k in k_values_to_consider])

            # Load Only MS if required
        if load_baseline:
            results.append([s, label_baseline, perc_ms2] +
                           [_topk["baseline"][return_percentage][k - 1] for k in k_values_to_consider])

        # Load Random if required
        if load_random:
            results.append([s, label_random, perc_ms2] +
                           [_topk["random"][return_percentage][k - 1] for k in k_values_to_consider])

    results = pd.DataFrame(results, columns=["sample", "Method", "perc_ms2"] +
                                            ["Top-%d" % k for k in k_values_to_consider])

    # Load selected parameters
    opt_params = pd.read_csv(os.path.join(idir, "opt_params.csv"))
    opt_params["Method"] = label_ms_rt

    # Load parameter goodness measures for all samples
    param_goodness_measure = pd.read_csv(os.path.join(idir, "measures.csv"))
    param_goodness_measure["Method"] = label_ms_rt

    return results, opt_params, param_goodness_measure


def _label_p(x, _k, test, results, show_variance=False):
    _m = np.mean(x)
    if show_variance:
        _str = "%.1f (%.1f)" % (_m, np.var(x))
    else:
        _str = "%.1f" % _m
        
    if test is None:
        return _str
    
    _x0 = results.loc[results.Method == "Only MS", "Top-%d" % _k]
    _m0 = np.mean(_x0)
    
    if test == "ttest_1samp":
        _p = ttest_1samp(x, _m0)[1]
    elif test == "ttest_ind":
        _p = ttest_ind(x, _x0)[1]
    elif test == "ttest_rel":
        _p = ttest_rel(x, _x0)[1]
    elif test == "wilcoxon_oneside":
        if _m != _m0:
            _p = wilcoxon(x, _x0, alternative="greater")[1]
        else:
            _p = np.nan
    elif test == "wilcoxon_twoside":
        if _m != _m0:
            _p = wilcoxon(x, _x0, alternative="two-sided")[1]
        else:
            _p = np.nan
    else:
        raise ValueError("Invalid test: '%s'" % test)
    
    if _p < 0.001 and (_m > _m0):
        _str += " (***)"
    elif _p < 0.01 and (_m > _m0):
        _str += " (**)"
    elif _p < 0.05 and (_m > _m0): 
        _str += " (*)"

    return _str
