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

# Script to run the experiments presented in Section 4.2 of the paper for the CASMI 2016 datasets using the MetFrag
# 2.2 method (for comparison).

import argparse
import numpy as np
import os
import gzip
import pickle
import sys
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit, ParameterGrid

from msmsrt_scorer.lib.data_utils import prepare_candidate_set_MetFrag
from msmsrt_scorer.lib.evaluation_tools import get_topk_performance_from_scores, evaluate_parameter_grid, get_top20AUC

from msmsrt_scorer.experiments.CASMI_2016.eval__TFG import load_data

# Participant hash strings encoding the different D values (RT weight) used as parameters for MetFrag 2.2.
# NOTE: One could have done this similar to the way it was done for EA (Massbank) that we load the
#       fragmenter and retention time scores (from MetFrag) separately and combine them here, instead of
#       taking the combined value from the MetFrag output. However, this is all iteratively grown code and
#       so did my understanding.
SETTINGS = {"baseline": "8afe4a14",
            0.5: "b01d6cd2", 0.3: "eb4e6fd9", 0.2: "eb34ba9d", 0.1: "cdb99681",
            0.05: "af718875", 0.01: "59cce739", 0.005: "4800dbf0", 0.001: "bdfe8b28"}


class Arguments(object):
    """
    Class simulating the output of ArgumentParser()
    """
    def __init__(self, participant, pref_model, ion_mode, max_n_cand, sort_candidates_by_ms2_score, database_fn):
        self.database_fn = database_fn
        self.participant = participant
        self.pref_model = pref_model
        self.ion_mode = ion_mode
        self.max_n_cand = max_n_cand
        self.sort_candidates_by_ms2_score = sort_candidates_by_ms2_score


def _run_for_grid_item(args_baseline, D, sub_set):
    """
    Calculate the marginals for the ms-features in the 'subset'.

    :param args_baseline: ArgumentParser() for the baseline (Only MS) setting.
    :param D: scalar, retention order weight
    :param sub_set: list, of indices selecting the training respectively test set ms-features
    :return: tuple (
        (D, None): hyper parameters that where used, i.e. retention order weight, no sigmoid parameter neded
        rep: 0
        marg: OrderedDict, containing the marginals (values) for all ms-features i (keys)
        Z_max: None
        p_max: -1
    )
    """
    # Load data for specified D value
    args_D = Arguments(participant="MetFrag_2.4.5__%s" % SETTINGS[D],
                       database_fn=args_baseline.database_fn,
                       pref_model=args_baseline.pref_model,
                       ion_mode=args_baseline.ion_mode,
                       max_n_cand=args_baseline.max_n_cand,
                       sort_candidates_by_ms2_score=args_baseline.sort_candidates_by_ms2_score)
    challenges, candidates = load_data(args_D)  # loads the full dataset

    # Prepare and subset candidate sets
    cnds = prepare_candidate_set_MetFrag(challenges, candidates, sub_set, ms2_idc=range(len(sub_set)))

    # Extract marginals, which are just the (combined) MetFrag-scores
    marg = {i: cnd["score"] for i, cnd in cnds.items()}

    return (D, None), 0, marg, None, -1.0


def run_parameter_grid(args, sub_set, h_param_grid, n_jobs):
    """
    Get marginals for the different hyper parameter grid values of D for the 'sub_set' if ms-features.

    :param args: argparse.ArgumentParser() object, holding the program parameters
    :param sub_set: list, indices that should be used for the grid-search, e.g. the training set
    :param h_param_grid: dictionary of lists, D value grid. The output format is defined by the scikit-learn
        function 'model_selection.ParameterGrid'.
    :param n_jobs: scalar, number of grid values to run in parallel

    :return: tuple (
        res: results for each D value grid element
        candidates: candidates loaded from the CSV
        h_param_grid: pass through
        1: number of "trees", just for compatibility here
    )
    """
    # Run Forward-Backward algorithm for each spanning tree and parameter tuple (in parallel)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_run_for_grid_item)(args, params["D"], sub_set) for params in h_param_grid)

    # FIXME: This is kind of nasty here.
    challenges, candidates = load_data(args)  # loads the full dataset

    # Prepare and subset candidate sets
    cnds = prepare_candidate_set_MetFrag(challenges, candidates, sub_set, ms2_idc=range(len(sub_set)))

    return res, cnds, h_param_grid, 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--param_selection_measure", type=str, default="topk_auc",
                            choices=["topk_auc", "ndcg", "p_marg", "p_max", "un_topk_auc", "un_p_marg"],
                            help="Criteria for the selection of the best (D, k)-tuple (hyper-parameters, see Section "
                                 "3.4 and 4.2.2). In the paper 'topk_auc' (top20AUC) was used.")

    arg_parser.add_argument("--D_value_grid", nargs="+", type=float,
                            help="Grid-values for the retention order weight. (1 - D) llh(MS) + D * llh(RT)")

    # Optional parameters optimization
    arg_parser.add_argument("--max_n_ms2", type=int, default=75,
                            help="Number of MS2 spectra used for the evaluation of the score integration framework. "
                                 "For each sub-sample (see '--n_samples') we use this parameter to define the test set "
                                 "size. The remaining spectra are used to determine the best (D, k)-tuple. See Section "
                                 "3.1 for details on the training and test split sizes.")

    arg_parser.add_argument("--n_jobs", type=int, default=4,
                            help="Number of jobs used to parallelize the score-integration on the spanning-tree"
                                 "ensembles.")

    # Optional parameters dataset
    arg_parser.add_argument("--ion_mode", type=str, default="positive",
                            help="Load spectra only of the specified ionization mode. In the paper we always analyzed "
                                 "negative and positive mode spectra separately.")

    arg_parser.add_argument("--participant", type=str, default="MetFrag_2.4.5__8afe4a14",
                            choices=["MetFrag_2.4.5__8afe4a14"], help="MS2 scoring approach to be used.")

    arg_parser.add_argument("--pref_model", type=str, default="c6d6f521",
                            help="Hash string identifying the RankSVM model used to predict the preference scores for "
                                 "the CASMI spectra. Take a look into the SQLite DB to see the actual configurations.")

    arg_parser.add_argument("--max_n_cand", type=float, default=np.inf,
                            help="Limit the maximum number of candidates per ms-feature. We use all candidates in the"
                                 "paper. This option is mainly used for development.")

    arg_parser.add_argument("--sort_candidates_by_ms2_score", action="store_true",
                            help="Should the molecular candidates for each ms-feature be sorted in descending order "
                                 "(largest MS2 scores first) while loading the data. This option was mainly added to "
                                 "remove leaked information in the case where we simulate missing MS2 information. The "
                                 "ordering would be that leaked information. For the paper we always load the "
                                 "candidates in a random order.")

    # Optional parameters data
    arg_parser.add_argument("--database_fn", type=str, help="Path to the score SQLite DB.")

    arg_parser.add_argument("--mode", type=str, default="debug", choices=["debug", "application"])

    arg_parser.add_argument("--base_odir", type=str, default="results__MetFrag22",
                            help="Base directory to store the results and output files.")

    # Optional parameters evaluation
    arg_parser.add_argument("--n_samples", type=int, default=50,
                            help="Number of random dataset sub-samples. Those have been already pre-sampled and are "
                                 "stored in the DB (see Section 3.1)")

    args = arg_parser.parse_args()
    print(str(args))

    # Define output directory
    odir = os.path.join(args.base_odir, args.mode,
                        "__".join(["param_selection_measure=%s" % args.param_selection_measure]),
                        "__".join(["ion_mode=%s" % args.ion_mode,
                                   "participant=%s" % args.participant,
                                   "max_n_cand=%.0f" % args.max_n_cand,
                                   "pref_model=%s" % args.pref_model,
                                   "sort_candidates_by_ms2_score=%d" % args.sort_candidates_by_ms2_score]))
    os.makedirs(odir, exist_ok=True)
    os.makedirs(os.path.join(odir, "candidates"), exist_ok=True)

    # Load data to determine the number of candidate sets
    challenges, candidates = load_data(args)

    # Set up hyper-parameter grid
    h_param_grid = ParameterGrid({"D": args.D_value_grid, "k": [None]})
    print("Number of grid-pairs: %d" % len(h_param_grid))

    measure_df = pd.DataFrame()
    opt_param_df = pd.DataFrame(columns=["sample", "D"])

    for s, (train_set, test_set) in enumerate(ShuffleSplit(n_splits=args.n_samples, test_size=args.max_n_ms2,
                                                           random_state=29181).split(candidates)):
        print("rep=%d/%d" % (s + 1, args.n_samples))
        print("\tn_train=%d, n_test=%d" % (len(train_set), len(test_set)))

        # Perform grid-search on training set
        df_train = evaluate_parameter_grid(*run_parameter_grid(args, train_set, h_param_grid, args.n_jobs))
        df_train["sample"] = s
        df_train["set"] = "train"
        measure_df = pd.concat([measure_df, df_train], sort=True, axis=0)

        # Get top-k baseline performance of the test set
        # Prepare the candidate set, i.e. sub-setting, sampling the MS2 (if needed), ...
        cnds_test = prepare_candidate_set_MetFrag(challenges, candidates, test_set, ms2_idc=range(len(test_set)))

        topk_bsl_test_casmi = get_topk_performance_from_scores(cnds_test, None, method="casmi2016")
        measure_df = pd.concat([
            measure_df,
            pd.DataFrame({
                "sample": [s],
                "set": ["test"],
                "D": [0.0],
                "k": [None],
                "topk_auc": [get_top20AUC(topk_bsl_test_casmi, len(candidates))],
                "top1": [topk_bsl_test_casmi[1][0]],
                "top3": [topk_bsl_test_casmi[1][2]],
                "top5": [topk_bsl_test_casmi[1][4]],
                "top10": [topk_bsl_test_casmi[1][9]],
                "top20": [topk_bsl_test_casmi[1][19]]
            })], sort=True, axis=0)

        print("\tTop-5 parameters:")
        print(df_train[["D", "k", args.param_selection_measure, "top1", "top5"]]
              .nlargest(5, columns=args.param_selection_measure))
        print("\tFlop-5 parameters:")
        print(df_train[["D", "k", args.param_selection_measure, "top1", "top5"]]
              .nsmallest(5, columns=args.param_selection_measure))

        # Get the optimal (D, k) parameter tuple
        _idxmax = df_train[args.param_selection_measure].idxmax()
        D_opt = df_train.loc[_idxmax, "D"]
        k_opt = df_train.loc[_idxmax, "k"]
        print("\tOptimal parameters: D=%f, k=%f" % (D_opt, k_opt))
        opt_param_df.loc[s] = [s, D_opt]

        # Run Forward-Backward algorithm on the test set
        res = _run_for_grid_item(args, D_opt, test_set)

        # Average the marginals
        marg_test = {i: res[2][i] for i in cnds_test}

        # Calculate the top-k performance on the test set using RT+MS
        topk_test_casmi = {"ms_rt": get_topk_performance_from_scores(cnds_test, marg_test, method="casmi2016"),
                           "baseline": get_topk_performance_from_scores(cnds_test, None, method="casmi2016")}

        topk_test_csi = {"ms_rt": get_topk_performance_from_scores(cnds_test, marg_test, method="csifingerid"),
                         "baseline": get_topk_performance_from_scores(cnds_test, None, method="csifingerid")}

        with gzip.open(os.path.join(odir, "__".join(["topk_csi",
                                                     "max_n_ms2=%03d" % args.max_n_ms2,
                                                     "sample_id=%03d" % s]) + ".pkl.gz"), "wb+") as of:
            pickle.dump(topk_test_csi, of, protocol=pickle.HIGHEST_PROTOCOL)

        with gzip.open(os.path.join(odir, "__".join(["topk_casmi",
                                                     "max_n_ms2=%03d" % args.max_n_ms2,
                                                     "sample_id=%03d" % s]) + ".pkl.gz"), "wb+") as of:
            pickle.dump(topk_test_casmi, of, protocol=pickle.HIGHEST_PROTOCOL)

        # Write out information to reload candidates
        ofn = os.path.join(odir, "candidates", "__".join(["candidates",
                                                          "max_n_ms2=%03d" % args.max_n_ms2,
                                                          "sample_id=%03d" % s]) + ".pkl.gz")
        if not os.path.exists(ofn):
            with gzip.open(ofn, "wb+") as of:
                pickle.dump({"test_set": test_set, "random_state": s, "max_n_ms2": args.max_n_ms2}, of,
                            protocol=pickle.HIGHEST_PROTOCOL)

    # Write out stuff for the ranking measurements and parameter selection evaluation
    measure_df.to_csv(os.path.join(odir, "measures.csv"), index=False)

    if args.mode in ["application", "debug"]:
        opt_param_df.to_csv(os.path.join(odir, "opt_params.csv"), index=False)

    sys.exit(0)
