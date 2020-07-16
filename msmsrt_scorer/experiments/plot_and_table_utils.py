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

import pandas as pd
import os
import pickle
import gzip
import numpy as np
import itertools as it

from scipy.stats import wilcoxon
from typing import Optional, List


def _get_sample_id_string(sample_id: Optional[int]) -> str:
    if sample_id is None:
        _sample_id = "*"  # wildcard to select all samples
    else:
        _sample_id = "%03d" % sample_id

    return _sample_id


def TOPK(order_prob_k: Optional[float], D_value: Optional[float], max_n_ms2: int, sample_id: Optional[int] = None,
         method="casmi") -> str:
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


def load_results(idir: str, label_ms_rt: str, max_n_ms2: int, k_values_to_consider: Optional[List] = None,
                 method="casmi", return_percentage=True, label_only_ms="Only MS", n_samples=50, load_baseline=True,
                 load_ms_rt=True):

    if k_values_to_consider is None:
        k_values_to_consider = [1, 3, 5, 10, 20]

    # Load top-k performance
    results = []

    for s in range(n_samples):
        fn = os.path.join(idir, TOPK(D_value=None, method=method, max_n_ms2=max_n_ms2, order_prob_k=None, sample_id=s))
        with gzip.open(fn) as file:
            _topk = pickle.load(file)

        # Load MS + RT if required
        if load_ms_rt:
            results.append([s, label_ms_rt] + [_topk["ms_rt"][return_percentage][k - 1] for k in k_values_to_consider])

            # Load Only MS if required
        if load_baseline:
            results.append(
                [s, label_only_ms] + [_topk["baseline"][return_percentage][k - 1] for k in k_values_to_consider])

    results = pd.DataFrame(results, columns=["sample", "Method"] + ["Top-%d" % k for k in k_values_to_consider])

    # Load selected parameters
    opt_params = pd.read_csv(os.path.join(idir, "opt_params.csv"))
    # opt_params = opt_params.loc[opt_params["sample"] < n_samples]
    opt_params["Method"] = label_ms_rt

    # Load parameter goodness measures for all samples
    param_goodness_measure = pd.read_csv(os.path.join(idir, "measures.csv"))
    param_goodness_measure["Method"] = label_ms_rt

    return results, opt_params, param_goodness_measure


def load_results_missing_ms2(
        idir, max_n_ms2,
        n_samples=50, k_values_to_consider: Optional[List] = None, method="casmi", return_percentage=True,
        load_baseline=True, label_baseline="Only MS",
        load_ms_rt=True, label_ms_rt="MS + RT",
        load_random=True, label_random="Random"):

    if k_values_to_consider is None:
        k_values_to_consider = [1, 3, 5, 10, 20]

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


def _label_p(x: pd.Series, y: pd.Series, test: Optional[str] = "wilcoxon_twoside", print_variance=False,
             print_mean=True) -> str:
    """
    We use the Wilcoxon signed-rank test to evaluate the significance of metabolite identification performance of our
    score integration framework (MS + RT) compared to the baseline (Only MS).

    Let:
        - x be the list of top-k accuracies x_i for each sample i           (MS + RT)
        - y be the list of baseline top-k accuracies y_i for each sample i  (Only MS)

    The Wilcoxon signed-rank test tests, given the differences d_i = x_i - y_i, whether the null-hypothesis

        'wilcoxon-twoside' H0: median(d) = 0

        or

        'wilcoxon-oneside' H0: median(d) < 0

        can be rejected assuming the alternative hypothesis is

        'wilcoxon-twoside' H1: median(d) != 0

        or

        'wilcoxon-oneside' H1: median(d) > 0

    The test significance is added to the output string, if the average accuracy of MS + RT is larger then the one of
    Only MS. That means, the significance level indicates the significance of the (if observed) performance improvement.

    :param x: array-like, shape = (n_samples, ), top-k accuracies for the current setup

    :param y: array-like, shape = (n_samples, ), top-k accuracies for the baseline (Only MS)

    :param test: string, name of the test to use (None, "wilcoxon-twoside" and "wilcoxon-oneside")
    
    :param print_variance: boolean, if True the accuracy variance is added to the cell label.
    
    :param print_mean: boolean, if True the mean accuracy is added to the cell label.
    
    :return: string, cell label
    """
    x_mean = np.mean(x)  # mean performance for the current group and top-k

    cell_label = []  # output cell label
    if print_mean:
        cell_label.append("%.1f" % x_mean)
    if print_variance:
        cell_label.append("(%.1f)" % np.var(x))

    if np.all(x.values == y.values):
        # If all values are the same in x and y, the signed-rank test cannot be performed. Should only happen when
        # x and y belong to "Only MS".
        return " ".join(cell_label)

    if test == "wilcoxon_oneside":
        # Calculate D = {d_i = x_i - y_i}_i
        # H0: median(D) < 0
        # H1: median(D) > 0
        _p = wilcoxon(x=x, y=y, alternative="greater")[1]
    elif test == "wilcoxon_twoside":
        # Calculate D = {d_i = x_i - y_i}_i
        # H0: median(D) = 0
        # H1: median(D) != 0
        _p = wilcoxon(x=x, y=y, alternative="two-sided")[1]
    else:
        raise ValueError("Invalid test: '%s'" % test)

    y_mean = np.mean(y)
    if _p < 0.001 and (x_mean > y_mean):
        cell_label.append("(***)")
    elif _p < 0.01 and (x_mean > y_mean):
        cell_label.append("(**)")
    elif _p < 0.05 and (x_mean > y_mean):
        cell_label.append("(*)")

    return " ".join(cell_label)
