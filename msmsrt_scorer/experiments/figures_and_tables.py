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
import seaborn as sns
import itertools as it

from scipy.stats import wilcoxon
from typing import Optional, List

from msmsrt_scorer.experiments.plot_and_table_utils import load_results, load_results_missing_ms2, _label_p
from msmsrt_scorer.experiments.EA_Massbank.plot_and_table_utils import IDIR as IDIR_EA
from msmsrt_scorer.experiments.EA_Massbank.plot_and_table_utils import IDIR_METFRAG as IDIR_METFRAG_EA
from msmsrt_scorer.experiments.CASMI_2016.plot_and_table_utils import IDIR as IDIR_CASMI
from msmsrt_scorer.experiments.CASMI_2016.plot_and_table_utils import IDIR_METFRAG as IDIR_METFRAG_CASMI


def table__candidate_set_comparison(base_dir: str, to_latex=False, test="wilcoxon_twoside", ms2scorer="MetFrag"):
    """
    Table 6 in the papaer.
    """
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    eval_method = "casmi"
    margin_type = "max"
    n_random_trees = 32
    k_values_to_consider = [1, 5, 10, 20]
    make_order_prob = "sigmoid"

    if ms2scorer == "MetFrag":
        participant = "MetFrag_2.4.5__8afe4a14"
    elif ms2scorer == "IOKR":
        participant = "IOKR__696a17f3"
    else:
        raise ValueError("Invalid MS2-scorer '%s'. Choices are 'MetFrag' and 'IOKR'.")

    # Table parameters
    # ----------------
    escape = False
    index = False
    column_format = "lcccc"

    # Load the results
    # ----------------
    res_global = []

    for restrict_candidates_to_correct_mf, candidate_set in [(None, "All"), (True, "Correct MF")]:
        res = []
        for ion_mode, max_n_ms2, n_samples in [("positive", 75, 50), ("negative", 50, 50)]:
            _idir = IDIR_CASMI(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
                base_dir=os.path.join(base_dir, "CASMI_2016/results__TFG__platt"), mode="application",
                param_selection_measure=param_selection_measure, make_order_prob=make_order_prob,
                norm_order_scores=False, margin_type=margin_type, participant=participant,
                restrict_candidates_to_correct_mf=restrict_candidates_to_correct_mf)

            res.append(load_results(
                _idir, "MS + RT (our)", max_n_ms2=max_n_ms2, n_samples=n_samples, method=eval_method,
                k_values_to_consider=k_values_to_consider)[0])
            res[-1]["Ionization"] = ion_mode
            res[-1]["Candidate Set"] = candidate_set

        # Get table with aggregated scores
        # --------------------------------
        res = pd.concat(res)
        res_baseline = res[res.Method == "Only MS"]
        res = res \
            .drop("sample", axis=1) \
            .groupby(["Candidate Set", "Method"]) \
            .agg({"Top-1": lambda x: _label_p(x, res_baseline["Top-1"], test=test),
                  "Top-5": lambda x: _label_p(x, res_baseline["Top-5"], test=test),
                  "Top-10": lambda x: _label_p(x, res_baseline["Top-10"], test=test),
                  "Top-20": lambda x: _label_p(x, res_baseline["Top-20"], test=test)}) \
            .reset_index()

        res_global.append(res)

    res_global = pd.concat(res_global)

    if to_latex:
        return res_global.to_latex(escape=escape, index=index, column_format=column_format)
    else:
        return res_global.set_index(["Candidate Set", "Method"])


def table__alternative_methods_comparison(base_dir: str, to_latex=False, test="wilcoxon_twoside"):
    """
    Table 2 in the paper.
    """
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    eval_method = "casmi"
    margin_type = "max"
    n_random_trees = 32
    k_values_to_consider = [1, 5, 10, 20]
    make_order_prob = "sigmoid"

    # Table parameters
    # ----------------
    escape = False
    index = False
    column_format = "lllcccc"

    # Load the results
    # ----------------
    res_global_score = []
    res_sigf = []

    # EA Massbank
    for ion_mode, max_n_ms2, n_samples in [("positive", 100, 100), ("negative", 65, 50)]:
        res = []

        _idir_our = IDIR_EA(
            tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
            base_dir=os.path.join(base_dir, "EA_Massbank/results__TFG__platt"), mode="application",
            param_selection_measure=param_selection_measure, make_order_prob=make_order_prob,
            norm_scores="none", margin_type=margin_type)

        _idir_metfrag22 = IDIR_METFRAG_EA(
            ion_mode=ion_mode, base_dir=os.path.join(base_dir, "EA_Massbank/results__MetFrag22"), mode="application",
            param_selection_measure=param_selection_measure)

        _idir_chaingraph = IDIR_EA(
            tree_method="chain", ion_mode=ion_mode, D_value_method=None,
            base_dir=os.path.join(base_dir, "EA_Massbank/results__TFG__gridsearch"), mode="application",
            param_selection_measure=param_selection_measure, make_order_prob="hinge_sigmoid",
            norm_scores="none", margin_type=margin_type)

        res.append(load_results(
                _idir_our, "MS + RT (our)", max_n_ms2, method=eval_method, n_samples=n_samples,
                k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "EA (Massbank)"
        res[-1]["Ionization"] = ion_mode

        res.append(load_results(
                _idir_metfrag22, "MS + RT (MetFrag 2.2)", max_n_ms2, method=eval_method, n_samples=n_samples,
                k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "EA (Massbank)"
        res[-1]["Ionization"] = ion_mode

        res.append(load_results(
                _idir_chaingraph, "MS + RT (Chain-graph)", max_n_ms2, method=eval_method, n_samples=n_samples,
                k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "EA (Massbank)"
        res[-1]["Ionization"] = ion_mode

        # Get table with aggregated scores
        # --------------------------------
        res = pd.concat(res).drop_duplicates()
        res_sigf.append(res)

        res_baseline = res[res.Method == "Only MS"]

        assert (res_baseline.shape[0] == n_samples)
        assert (res[res.Method == "MS + RT (our)"].shape[0] == n_samples)
        assert (res[res.Method == "MS + RT (MetFrag 2.2)"].shape[0] == n_samples)
        assert (res[res.Method == "MS + RT (Chain-graph)"].shape[0] == n_samples)

        res_global_score.append(
            res.drop("sample", axis=1)
               .groupby(["Method", "Dataset", "Ionization"])
               .agg({"Top-1": lambda x: _label_p(x, res_baseline["Top-1"], test=test),
                     "Top-5": lambda x: _label_p(x, res_baseline["Top-5"], test=test),
                     "Top-10": lambda x: _label_p(x, res_baseline["Top-10"], test=test),
                     "Top-20": lambda x: _label_p(x, res_baseline["Top-20"], test=test)})
               .reset_index()
        )

    # CASMI 2016
    for ion_mode, max_n_ms2, n_samples in [("positive", 75, 50), ("negative", 50, 50)]:
        res = []

        _idir_our = IDIR_CASMI(
            tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
            base_dir=os.path.join(base_dir, "CASMI_2016/results__TFG__platt"), mode="application",
            param_selection_measure=param_selection_measure,
            make_order_prob=make_order_prob, norm_order_scores=False, margin_type=margin_type)

        _idir_metfrag22 = IDIR_METFRAG_CASMI(
            ion_mode=ion_mode, base_dir=os.path.join(base_dir, "CASMI_2016/results__MetFrag22"), mode="application",
            param_selection_measure=param_selection_measure)

        _idir_chaingraph = IDIR_CASMI(
            tree_method="chain", ion_mode=ion_mode, D_value_method=None,
            base_dir=os.path.join(base_dir, "CASMI_2016/results__TFG__gridsearch"), mode="application",
            param_selection_measure=param_selection_measure, make_order_prob="hinge_sigmoid",
            norm_order_scores=False, margin_type=margin_type)

        res.append(load_results(_idir_our, "MS + RT (our)", max_n_ms2, n_samples=n_samples, method=eval_method,
                                k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "CASMI 2016"
        res[-1]["Ionization"] = ion_mode

        res.append(load_results(_idir_metfrag22, "MS + RT (MetFrag 2.2)", max_n_ms2, n_samples=n_samples,
                                method=eval_method, k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "CASMI 2016"
        res[-1]["Ionization"] = ion_mode

        res.append(load_results(_idir_chaingraph, "MS + RT (Chain-graph)", max_n_ms2, n_samples=n_samples,
                                method=eval_method, k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "CASMI 2016"
        res[-1]["Ionization"] = ion_mode

        # Get table with aggregated scores
        # --------------------------------
        res = pd.concat(res).drop_duplicates()
        res_sigf.append(res)

        res_baseline = res[res.Method == "Only MS"]

        assert (res_baseline.shape[0] == n_samples)
        assert (res[res.Method == "MS + RT (our)"].shape[0] == n_samples)
        assert (res[res.Method == "MS + RT (MetFrag 2.2)"].shape[0] == n_samples)
        assert (res[res.Method == "MS + RT (Chain-graph)"].shape[0] == n_samples)

        res_global_score.append(
            res.drop("sample", axis=1)
               .groupby(["Method", "Dataset", "Ionization"])
               .agg({"Top-1": lambda x: _label_p(x, res_baseline["Top-1"], test=test),
                     "Top-5": lambda x: _label_p(x, res_baseline["Top-5"], test=test),
                     "Top-10": lambda x: _label_p(x, res_baseline["Top-10"], test=test),
                     "Top-20": lambda x: _label_p(x, res_baseline["Top-20"], test=test)})
               .reset_index()
        )

    # Get table comparing all methods pairwise
    # ----------------------------------------
    res_sigf = pd.concat(res_sigf)
    res_global_sigf = []

    for k in [1, 20]:
        res = []
        for m1, m2 in it.permutations(res_global_score[0].Method.unique(), 2):
            _smp_m1 = res_sigf[res_sigf.Method == m1]["sample"].values
            _smp_m2 = res_sigf[res_sigf.Method == m2]["sample"].values
            assert np.all(_smp_m1 == _smp_m2)

            _topk_m1 = res_sigf[res_sigf.Method == m1]["Top-%d" % k].values
            _topk_m2 = res_sigf[res_sigf.Method == m2]["Top-%d" % k].values

            p = wilcoxon(x=_topk_m1, y=_topk_m2, alternative="greater")[1]  # H0: med(x - y) < 0, H1: med(x - y) > 0

            res.append(pd.DataFrame({"Method-1": [m1], "Method-2": [m2], "p_value": [p], "k": [k]}))

        res_global_sigf.append(pd.concat(res))

    res_global_sigf = {_res["k"].unique().item(): _res.pivot(index="Method-1", columns="Method-2", values="p_value")
                                                      .applymap(lambda z: "%.1e" % z if z < 0.05 else "n.s")
                                                      .reset_index()
                       for _res in res_global_sigf}

    if to_latex:
        res_global_score = "\n---\n\n".join([_df.to_latex(escape=escape, index=index, column_format=column_format)
                                             for _df in res_global_score])
        res_global_sigf = {_k: _res.to_latex(escape=escape, index=index, column_format=column_format)
                           for _k, _res in res_global_sigf.items()}

        return res_global_score, res_global_sigf
    else:
        res_global_score = pd.concat(res_global_score).set_index(["Dataset", "Ionization", "Method"])

        return res_global_score, res_global_sigf


def table__edgepotential_function_comparison(base_dir: str, to_latex=False, test="wilcoxon_twoside"):
    """
    Table S1 in the paper.
    """
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    eval_method = "casmi"
    margin_type = "max"
    n_random_trees = 32
    k_values_to_consider = [1, 5, 10, 20]

    # Table parameters
    # ----------------
    escape = False
    index = False
    column_format = "lcccc"

    # Load the results
    # ----------------
    res = []

    # EA Dataset
    for i, (make_order_prob, sub_dir) in enumerate(zip(["sigmoid", "hinge_sigmoid", "stepfun"],
                                                       ["EA_Massbank/results__TFG__platt",
                                                        "EA_Massbank/results__TFG__gridsearch",
                                                        "EA_Massbank/results__TFG__gridsearch"])):
        for ion_mode, max_n_ms2, n_samples in [("positive", 100, 100), ("negative", 65, 50)]:
            _idir = IDIR_EA(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
                mode="application", base_dir=os.path.join(base_dir, sub_dir),
                param_selection_measure=param_selection_measure, make_order_prob=make_order_prob, norm_scores="none",
                margin_type=margin_type)

            res.append(load_results(
                _idir, "MS + RT (our, %s)" % make_order_prob, max_n_ms2, n_samples=n_samples, method=eval_method,
                k_values_to_consider=k_values_to_consider, load_baseline=(i == 0))[0])
            res[-1]["Dataset"] = "EA (Massbank)"
            res[-1]["Ionization"] = ion_mode

    # CASMI Dataset
    for i, (make_order_prob, sub_dir) in enumerate(zip(["sigmoid", "hinge_sigmoid", "stepfun"],
                                                       ["CASMI_2016/results__TFG__platt",
                                                        "CASMI_2016/results__TFG__gridsearch",
                                                        "CASMI_2016/results__TFG__gridsearch"])):
        for ion_mode, max_n_ms2, n_samples in [("positive", 75, 50), ("negative", 50, 50)]:
            _idir = IDIR_CASMI(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
                mode="application", base_dir=os.path.join(base_dir, sub_dir),
                param_selection_measure=param_selection_measure, make_order_prob=make_order_prob,
                norm_order_scores=False, margin_type=margin_type)

            res.append(load_results(
                _idir, "MS + RT (our, %s)" % make_order_prob, max_n_ms2, n_samples=n_samples, method=eval_method,
                k_values_to_consider=k_values_to_consider, load_baseline=(i == 0))[0])
            res[-1]["Dataset"] = "CASMI 2016"
            res[-1]["Ionization"] = ion_mode

    # Get table with aggregated scores
    # --------------------------------
    res = pd.concat(res)
    res_score = res \
        .drop("sample", axis=1) \
        .groupby(["Method", "Dataset", "Ionization"]).mean() \
        .groupby(["Method"]).mean() \
        .round(1) \
        .reset_index()

    # Get the table indicating the significance over the baseline
    # -----------------------------------------------------------
    res_baseline = res[res.Method == "Only MS"]

    for method in res.Method.unique():
        assert np.all(res_baseline["sample"].values == res[res.Method == method]["sample"].values)
        assert np.all(res_baseline["Dataset"].values == res[res.Method == method]["Dataset"].values)
        assert np.all(res_baseline["Ionization"].values == res[res.Method == method]["Ionization"].values)

    res_p = res[res.Method != "Only MS"] \
        .drop("sample", axis=1) \
        .groupby("Method") \
        .agg({"Top-1": lambda x: _label_p(x, res_baseline["Top-1"], test=test, print_mean=False),
              "Top-5": lambda x: _label_p(x, res_baseline["Top-5"], test=test, print_mean=False),
              "Top-10": lambda x: _label_p(x, res_baseline["Top-10"], test=test, print_mean=False),
              "Top-20": lambda x: _label_p(x, res_baseline["Top-20"], test=test, print_mean=False)}) \
        .reset_index()

    if to_latex:
        return "\n---\n\n".join([res_score.to_latex(escape=escape, index=index, column_format=column_format),
                          res_p.to_latex(escape=escape, index=index, column_format=column_format)])

    else:
        return res_score, res_p


def table__MetFrag_vs_IOKR_comparison(base_dir: str, to_latex=False):
    """
    Table 5 in the Paper.
    """
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    eval_method = "casmi"
    make_order_prob = "sigmoid"
    margin_type = "max"
    n_random_trees = 32
    k_values_to_consider = [1, 5, 10, 20]

    # Table parameters
    # ----------------
    ms2scorer_col_name = "\\msms{}-Scorers"
    escape = False
    index = False
    column_format = "@{}llcccc@{}"

    # Load the results
    # ----------------
    res = []

    for participant, ms2scorer in [("MetFrag_2.4.5__8afe4a14", "MetFrag"), ("IOKR__696a17f3", "IOKR")]:
        # EA Dataset
        for ion_mode, max_n_ms2, n_samples in [("positive", 100, 100), ("negative", 65, 50)]:
            _idir = IDIR_EA(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
                mode="application", base_dir=os.path.join(base_dir, "EA_Massbank/results__TFG__platt"),
                param_selection_measure=param_selection_measure, make_order_prob=make_order_prob, norm_scores="none",
                margin_type=margin_type, participant=participant)

            res.append(load_results(
                _idir, "MS + RT", max_n_ms2, n_samples=n_samples, method=eval_method,
                k_values_to_consider=k_values_to_consider)[0])
            res[-1]["Dataset"] = "EA (Massbank)"
            res[-1]["Ionization"] = ion_mode
            res[-1][ms2scorer_col_name] = ms2scorer

        # CASMI Dataset
        for ion_mode, max_n_ms2, n_samples in [("positive", 75, 50), ("negative", 50, 50)]:
            _idir = IDIR_CASMI(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
                mode="application", base_dir=os.path.join(base_dir, "CASMI_2016/results__TFG__platt"),
                param_selection_measure=param_selection_measure, make_order_prob=make_order_prob,
                norm_order_scores=False, margin_type=margin_type, participant=participant)

            res.append(load_results(
                _idir, "MS + RT", max_n_ms2, n_samples=n_samples, method=eval_method,
                k_values_to_consider=k_values_to_consider)[0])
            res[-1]["Dataset"] = "CASMI 2016"
            res[-1]["Ionization"] = ion_mode
            res[-1][ms2scorer_col_name] = ms2scorer

    res = pd.concat(res) \
        .drop("sample", axis=1) \
        .groupby([ms2scorer_col_name, "Method", "Dataset", "Ionization"]).mean() \
        .groupby([ms2scorer_col_name, "Method"]).mean() \
        .round(1) \
        .reset_index()  # type: pd.DataFrame

    if to_latex:
        return res.to_latex(escape=escape, index=index, column_format=column_format)
    else:
        return res


def figure__missing_ms2(base_dir: str):
    """
    Figure 4 in the paper.
    """
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    use_global_parameter_selection = True
    k_values_to_consider = [1, 5, 10, 20]
    eval_method = "casmi"
    n_random_trees = 32
    margin_type = "max"
    make_order_prob = "sigmoid"

    # Load the results
    # ----------------
    res = []

    # EA Dataset
    for ion_mode, max_n_ms2, n_samples in [("positive", 100, 100), ("negative", 65, 50)]:
        _idir = IDIR_EA(
            tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
            mode="missing_ms2", base_dir=os.path.join(base_dir, "EA_Massbank/results__TFG__platt"),
            param_selection_measure=param_selection_measure, make_order_prob=make_order_prob, norm_scores="none",
            use_global_parameter_selection=use_global_parameter_selection, margin_type=margin_type)

        res.append(load_results_missing_ms2(
            _idir, max_n_ms2, n_samples=n_samples, method=eval_method, load_random=False,
            k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "EA (Massbank)"
        res[-1]["Ionization"] = ion_mode
        res[-1]["Evaluation"] = eval_method

    # CASMI
    for ion_mode, max_n_ms2, n_samples in [("positive", 75, 50), ("negative", 50, 50)]:
        _idir = IDIR_CASMI(
            tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode, D_value_method=None,
            mode="missing_ms2", base_dir=os.path.join(base_dir, "CASMI_2016/results__TFG__platt"),
            param_selection_measure=param_selection_measure, make_order_prob=make_order_prob, norm_order_scores=False,
            margin_type=margin_type, use_global_parameter_selection=use_global_parameter_selection)

        res.append(load_results_missing_ms2(
            _idir, max_n_ms2, n_samples=n_samples, method=eval_method, load_random=False,
            k_values_to_consider=k_values_to_consider)[0])
        res[-1]["Dataset"] = "CASMI 2016"
        res[-1]["Ionization"] = ion_mode
        res[-1]["Evaluation"] = eval_method

    res = pd.concat(res)

    # Prepare data for plotting
    # -------------------------
    res_baseline = res[res.Method == "Only MS"] \
                       .reset_index() \
                       .drop(["index"], axis=1) \
                       .iloc[:, [0, 1, 2, 7, 8, 9, 3, 4, 5, 6]]
    res_msrt = res[res.Method == "MS + RT"] \
                   .reset_index() \
                   .drop(["index"], axis=1) \
                   .iloc[:, [0, 1, 2, 7, 8, 9, 3, 4, 5, 6]]

    assert all(res_baseline.Method == "Only MS")
    assert all(res_msrt.Method == "MS + RT")
    assert np.all(res_baseline.columns == np.array([
        "sample", "Method", "perc_ms2", "Dataset", "Ionization", "Evaluation", "Top-1", "Top-5", "Top-10", "Top-20"]))
    assert all(res_baseline["sample"] == res_msrt["sample"])

    res_imp = pd \
        .concat((res_msrt.loc[:, ["perc_ms2", "Dataset", "Ionization", "Evaluation"]],
                 res_msrt.iloc[:, 6:] - res_baseline.iloc[:, 6:]), axis=1) \
        .melt(id_vars=["perc_ms2", "Dataset", "Ionization", "Evaluation"], var_name="Top-k",
              value_name="Accuracy improvement (%-points)")

    # Plot results
    # ------------
    fig, axrr = plt.subplots(2, 2, squeeze=False, figsize=(6, 4.5), sharex="all")

    _xlabels = [0, 25, 50, 75, 100]
    _x = range(len(_xlabels))

    for i, topk in enumerate(["Top-%d" % k for k in [1, 20]]):
        # Get the current axis
        ax = axrr[1, i]

        # Plot basline, horizontal line at 0
        _h_bsl = ax.hlines(0, -0.15, 4.15, color="black", linestyle="-")

        # Plot average (RT + MS)
        #   (1) Calculate average for each (perc, dataset, ionization)
        #   (2) Calculate overall-average for each 'perc'
        _eval_method = "casmi"

        _y = res_imp.loc[(res_imp["Top-k"] == topk) & (res_imp["Evaluation"] == _eval_method)] \
            .groupby(["perc_ms2", "Dataset", "Ionization"]).mean() \
            .groupby("perc_ms2").mean()
        _h1_rtms, = ax.plot(_x, _y, linestyle="--", color="black")
        _h2_rtms = ax.scatter(_x, _y, color="black")

        ax.grid(axis="y", linestyle="--")

        if i == 0:
            ax.set_ylabel("Identification Accuracy\nImprovement (%-points)")

        ax.set_xlabel("Percentage of MS2 available")
        ax.set_xticks(_x)
        ax.set_xticklabels(_xlabels)

    for i, topk in enumerate(["Top-%d" % k for k in [1, 20]]):
        # Get the current axis
        ax = axrr[0, i]

        # Plot average (RT + MS)
        #   (1) Calculate average for each (perc, dataset, ionization)
        #   (2) Calculate overall-average for each 'perc'
        _eval_method = "casmi"

        # Plot Baseline (Only RT)
        _y = res_baseline \
            .loc[res_baseline.Evaluation == _eval_method, ["Dataset", "Ionization", "perc_ms2", topk]] \
            .groupby(["Dataset", "Ionization", "perc_ms2"]).mean() \
            .groupby(["perc_ms2"]).mean()
        ax.plot(_x, _y, linestyle="-", color="black")

        # Plot average (RT + MS)
        _y = res_msrt \
            .loc[res_msrt.Evaluation == _eval_method, ["Dataset", "Ionization", "perc_ms2", topk]] \
            .groupby(["Dataset", "Ionization", "perc_ms2"]).mean() \
            .groupby(["perc_ms2"]).mean()
        ax.plot(_x, _y, linestyle="--", color="black")
        ax.scatter(_x, _y, color="black")

        ax.grid(axis="y", linestyle="--")

        if i == 0:
            ax.set_ylabel("Identification\nAccuracy (%)")

        ax.set_title("%s" % topk)

    return fig, axrr


def figure__parameter_selection(base_dir: str):
    """
    Figure 4 in the paper.

    :param base_dir:
    :return:
    """

    # General parameters
    # ------------------
    D_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5]
    n_random_trees = 32
    param_selection_method = "topk_auc"
    margin_type = "max"

    # Prepare figure
    # --------------
    fig, axrr = plt.subplots(2, 2, figsize=(7.5, 5.25), sharey="all")

    for row, (make_order_prob, sub_dir) in enumerate([("sigmoid", "results__TFG__platt"),
                                                      ("stepfun", "results__TFG__gridsearch")]):
        # Load results
        # ------------
        msr = []

        # CASMI
        for ion_mode in ["Positive", "Negative"]:
            _idir = IDIR_CASMI(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode.lower(), D_value_method=None,
                base_dir=os.path.join(base_dir, "CASMI_2016", sub_dir), mode="development",
                param_selection_measure=None, make_order_prob=make_order_prob, margin_type=margin_type,
                norm_order_scores=False)

            msr.append(pd.read_csv(os.path.join(_idir, "measures.csv")))
            msr[-1]["Dataset"] = "CASMI 2016"
            msr[-1]["Ionization"] = ion_mode
            msr[-1] = msr[-1][msr[-1].D != 0]

        # EA Massbank
        for ion_mode in ["Positive", "Negative"]:
            _idir = IDIR_EA(
                tree_method="random", n_random_trees=n_random_trees, ion_mode=ion_mode.lower(), D_value_method=None,
                base_dir=os.path.join(base_dir, "EA_Massbank", sub_dir), mode="development",
                param_selection_measure=None, make_order_prob=make_order_prob, norm_scores="none",
                margin_type=margin_type)

            msr.append(pd.read_csv(os.path.join(_idir, "measures.csv")))
            msr[-1]["Dataset"] = "EA (Massbank)"
            msr[-1]["Ionization"] = ion_mode
            # 100 samples available for the positive ionization mode. Reduce to 50, so we have the same number for all
            # datasets.
            msr[-1] = msr[-1][(msr[-1].D != 0) & (msr[-1]["sample"] < 50)]

        # Prepare data for plotting
        # -------------------------
        msr = pd.concat(msr, axis=0, sort=True)
        msr_test = msr[(msr.set == "test")].reset_index()
        msr_train = msr[(msr.set == "train")].reset_index()
        opt = msr_train.iloc[msr_train.groupby(["sample", "Dataset", "Ionization"]).idxmax()[param_selection_method]]

        # Plot results
        # ------------
        cts = np.array([np.sum(opt.D == D) for D in D_range])
        _x = np.arange(len(D_range))

        for col, k in enumerate([1, 20]):
            ax_bar = axrr[row, col]
            ax_line = axrr[row, col].twinx()

            # Line-plot
            sns.pointplot(data=msr_test, x="D", y="top%d" % k, linestyles="--", errwidth=1.5, capsize=0.25,
                          scale=0.8, ax=ax_line, seed=2020)
            ax_line.set_ylabel("")

            # Bar-plot
            ax_bar.bar(x=_x, height=cts / np.sum(cts), width=0.7, alpha=0.25, edgecolor="black", color="grey")
            ax_bar.grid(axis="y")

            axrr[row, col].set_title("Top-%d" % k, fontweight="bold", fontsize=12)

            ax_line.tick_params(axis="y", colors=sns.color_palette()[0], labelsize=11)
            ax_bar.tick_params(axis="y", colors="gray", labelsize=11)

            if col == 0:
                ax_bar.set_ylabel("Selection frequency\nof parameter D\n(on Training sets)", fontsize=11,
                                  color="gray")
            if col == 1:
                ax_line.set_ylabel("Identification accuarcy (%)\n(on Test sets)", fontsize=11,
                                   color=sns.color_palette()[0])

            if row == 0:
                axrr[row, col].set_xlabel("")
                axrr[row, col].set_xticklabels([])
            else:
                axrr[row, col].set_xlabel("Weight on retention-order (D)", fontsize=11)
                axrr[row, col].set_xticklabels(ax_line.get_xticklabels(), rotation=45)

    return fig, axrr


def figure__number_of_random_spanning_trees(base_dir: str, L_range: Optional[List] = None, for_paper=False):
    """
    Figure 2 in the paper (for_paper=True).

    :param base_dir:
    :param L_range: list, range of the number of spanning trees to plot.
    :param for_paper: boolean, if true only the curves top-1 and top-20 accuracy are plotted (less space in the paper)
    :return:
    """
    # General parameters
    # ------------------
    param_selection_measure = "topk_auc"
    eval_method = "casmi"
    mode = "application"
    make_order_prob = "sigmoid"
    if L_range is None:
        L_range = [1, 2, 4, 8, 16, 32, 64]

    # Load the results
    # ----------------
    results = {"Max": pd.DataFrame(), "Sum": pd.DataFrame()}
    for margin_type in results:
        for i, T in enumerate(L_range):
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
    for margin_type in results:
        results[margin_type] = results[margin_type] \
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

    _xlabels = L_range
    _x = range(len(_xlabels))

    for i, topk in enumerate(["Top-%d" % k for k in k_range]):
        r, c = np.unravel_index(i, shape=axrr.shape)
        ax = axrr[r, c]

        # Plot average (RT + MS)
        # 1) aggregate across each sample for a specific dataset and ionization tuple
        # 2) globally average results
        for margin_type in results:
            _y = results[margin_type] \
                .loc[(results[margin_type]["Top-k"] == topk) & (results[margin_type].Method != "Only MS")] \
                .groupby(["T", "Dataset", "Ionization"])["Top-k Accuracy (%)"].mean() \
                .groupby("T").mean()
            ax.plot(_x, _y, linestyle="--", label="MS + RT (%s)" % margin_type)
            ax.scatter(_x, _y)

        # Plot average (Only MS)
        for margin_type in results:
            _y = results[margin_type] \
                .loc[(results[margin_type]["Top-k"] == topk) & (results[margin_type].Method == "Only MS")] \
                .groupby(["Dataset", "Ionization"])["Top-k Accuracy (%)"].mean() \
                .mean()

        ax.hlines(_y, -0.15, len(L_range) - 1 + 0.15, linestyle="-", color="black", label="Only MS")

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

