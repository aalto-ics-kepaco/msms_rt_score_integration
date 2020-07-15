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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, List

from msmsrt_scorer.experiments.plot_and_table_utils import load_results
from msmsrt_scorer.experiments.EA_Massbank.plot_and_table_utils import IDIR as IDIR_EA
from msmsrt_scorer.experiments.CASMI_2016.plot_and_table_utils import IDIR as IDIR_CASMI


def figure__number_of_random_spanning_trees(base_dir: str, T_range: Optional[List] = None, for_paper=False):
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    eval_method = "casmi"
    mode = "application"
    make_order_prob = "sigmoid"
    if T_range is None:
        T_range = [1, 2, 4, 8, 16, 32, 64]

    # Load the results
    # ----------------
    results = {"Max": pd.DataFrame(), "Sum": pd.DataFrame()}
    for margin_type in results:
        for i, T in enumerate(T_range):
            #########################
            # LOAD CASMI 2016 RESULTS
            #########################
            for _ionm, _maxn, _nsamp in [("positive", 75, 50), ("negative", 50, 50)]:
                _idir = IDIR_CASMI(
                    tree_method="random", n_random_trees=T, ion_mode=_ionm, D_value_method=None,
                    base_dir=os.path.join(base_dir, "CASMI_2016/results__TFG__platt"), mode=mode,
                    param_selection_measure=param_selection_measure, make_order_prob=make_order_prob,
                    norm_order_scores=False, margin_type=margin_type.lower())

                _tmp = load_results(_idir, "MS + RT", _maxn, method=eval_method, label_only_ms="Only MS",
                                    n_samples=_nsamp, load_baseline=(i == 0))

                # Results
                _results = _tmp[0]
                _results["T"] = T
                _results["Ionization"] = _ionm
                _results["Dataset"] = "CASMI2016"
                _results["Margin"] = margin_type
                results[margin_type] = pd.concat([results[margin_type], _results], axis=0, sort=True)

            ############################
            # LOAD EA (MASSBANK) RESULTS
            ############################
            for _ionm, _maxn, _nsamp in [("positive", 100, 100), ("negative", 65, 50)]:
                _idir = IDIR_EA(
                    tree_method="random", n_random_trees=T, ion_mode=_ionm, D_value_method=None,
                    base_dir=os.path.join(base_dir, "EA_Massbank/results__TFG__platt"), mode=mode,
                    param_selection_measure=param_selection_measure, make_order_prob=make_order_prob,
                    norm_scores="none", margin_type=margin_type.lower())

                _tmp = load_results(_idir, "MS + RT", _maxn, method=eval_method,
                                    label_only_ms="Only MS", n_samples=_nsamp, load_baseline=(i == 0))
                # Results
                _results = _tmp[0]
                _results["T"] = T
                _results["Ionization"] = _ionm
                _results["Dataset"] = "EA"
                _results["Margin"] = margin_type
                results[margin_type] = pd.concat([results[margin_type], _results], axis=0, sort=True)

    # Prepare the results for plotting
    # --------------------------------
    _res_melt_max = results["Max"] \
        .drop(["sample", "Margin"], axis=1) \
        .melt(id_vars=["Ionization", "T", "Method", "Dataset"], var_name="Top-k", value_name="Top-k Accuracy (%)")
    _res_melt_sum = results["Sum"] \
        .drop(["sample", "Margin"], axis=1) \
        .melt(id_vars=["Ionization", "T", "Method", "Dataset"], var_name="Top-k", value_name="Top-k Accuracy (%)")

    # Plot the figure
    # ---------------
    if for_paper:
        n_rows = 1
        k_range = [1, 20]
        figsize = (6, 2)
    else:
        n_rows = 2
        k_range = [1, 5, 10, 20]
        figsize = (6, 4.5)

    fig, axrr = plt.subplots(n_rows, 2, figsize=figsize, sharex="all", squeeze=False)

    _xlabels = T_range
    _x = range(len(_xlabels))

    for i, topk in enumerate(["Top-%d" % k for k in k_range]):
        r, c = np.unravel_index(i, shape=axrr.shape)
        ax = axrr[r, c]

        # Plot average (RT + MS)
        _y = _res_melt_max \
            .loc[(_res_melt_max["Top-k"] == topk) & (_res_melt_max.Method != "Only MS")] \
            .groupby(["T", "Dataset", "Ionization"])["Top-k Accuracy (%)"].mean() \
            .groupby("T").mean()
        ax.plot(_x, _y, linestyle="--", label="MS + RT (Max)")
        ax.scatter(_x, _y)

        _y = _res_melt_sum \
            .loc[(_res_melt_sum["Top-k"] == topk) & (_res_melt_sum.Method != "Only MS")] \
            .groupby(["T", "Dataset", "Ionization"])["Top-k Accuracy (%)"].mean() \
            .groupby("T").mean()
        ax.plot(_x, _y, linestyle="--", label="MS + RT (Sum)")
        ax.scatter(_x, _y)

        # Plot average (Only MS)
        __y_max = _res_melt_max \
            .loc[(_res_melt_max["Top-k"] == topk) & (_res_melt_max.Method == "Only MS")] \
            .groupby(["Dataset", "Ionization"])["Top-k Accuracy (%)"].mean() \
            .mean()
        __y_sum = _res_melt_sum \
            .loc[(_res_melt_sum["Top-k"] == topk) & (_res_melt_sum.Method == "Only MS")] \
            .groupby(["Dataset", "Ionization"])["Top-k Accuracy (%)"].mean() \
            .mean()
        assert (__y_max == __y_sum)

        _y = __y_max
        ax.hlines(_y, -0.15, len(T_range) - 1 + 0.15, linestyle="-", color="black", label="Only MS")

        ax.set_title(topk)

        ax.grid(axis="y")

        if c == 0:
            ax.set_ylabel("Identification\nAccuracy (%)")

        if r < (n_rows - 1):
            ax.set_xticklabels([])
        else:
            ax.set_xticks(_x)
            ax.set_xticklabels(_xlabels)
            ax.set_xlabel("Number of Spanning-Trees")

        if i == 0:
            ax.legend(loc="lower right")

    return fig, axrr
