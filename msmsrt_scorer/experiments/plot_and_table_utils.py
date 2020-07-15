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
