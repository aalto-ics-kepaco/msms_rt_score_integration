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

# Script to run the experiments presented in Section 4.2 of the paper for the CASMI 2016 datasets.
# The description of the experiment can be found in Section 3.5.1.

import argparse
import sqlite3
import numpy as np
import os
import gzip
import pickle
import sys
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit, ParameterGrid

from msmsrt_scorer.lib.data_utils import load_dataset_CASMI, prepare_candidate_set_MetFrag
from msmsrt_scorer.lib.data_utils import prepare_candidate_set_IOKR
from msmsrt_scorer.lib.evaluation_tools import get_topk_performance_from_scores, evaluate_parameter_grid, get_marginals
from msmsrt_scorer.lib.evaluation_tools import run_parameter_grid, get_top20AUC


def load_data(args):
    """
    Load data from CASMI database

    :param args: argparse.ArgumentParser() object, holding the program parameters

    :return: challenge and candidates, dictionaries
    """
    with sqlite3.connect("file:" + args.database_fn + "?mode=ro", uri=True) as db:
        challenges, candidates = load_dataset_CASMI(
            db, ion_mode=args.ion_mode, participant=args.participant, prefmodel=args.pref_model,
            max_n_cand=args.max_n_cand, sort_candidates_by_ms2_score=args.sort_candidates_by_ms2_score,
            restrict_candidates_to_correct_mf=args.restrict_candidates_to_correct_mf)

    return challenges, candidates


def load_platt_k(args):
    """
    Load the sigmoid parameter k determined using Platt's method during the RankSVM model training.

    See Section 4.2.2

    :param args: argparse.ArgumentParser() object, holding the script parameters
    :return: scalar, sigmoid k parameter
    """
    with sqlite3.connect("file:" + args.database_fn + "?mode=ro", uri=True) as db:
        res = db.execute("SELECT platt_parameters FROM preference_scores_meta"
                         "   WHERE name IS ?", (args.pref_model, )).fetchall()
        k = - eval(res[0][0])["A"]

    return k


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--param_selection_measure", type=str, default="topk_auc",
                            choices=["topk_auc", "ndcg", "p_marg", "p_max", "un_topk_auc", "un_p_marg"],
                            help="Criteria for the selection of the best (D, k)-tuple (hyper-parameters, see Section "
                                 "3.4 and 4.2.2). In the paper 'topk_auc' (top20AUC) was used.")

    arg_parser.add_argument("--D_value_grid", nargs="+", type=float,
                            help="Grid-values for the retention order weight. (1 - D) * llh(MS) + D * llh(RT)")

    arg_parser.add_argument("--order_prob_k_grid", nargs="+", type=str, default="platt",
                            help="K-parameter grid for the sigmoid used as edge potential function (see Section 2.2.3).")

    arg_parser.add_argument("--margin_type", type=str, default="max", choices=["max", "sum"],
                            help="Which marginal should be used: max-marginal or sum-marginal. See Section 2.3.")

    arg_parser.add_argument("--make_order_prob", type=str, choices=["sigmoid", "stepfun", "hinge_sigmoid"],
                            default="sigmoid", help="Which function to use as edge potential function. (see Section "
                                                    "2.2.3)")

    arg_parser.add_argument("--norm_order_scores", action="store_true",
                            help="Should the transition matrix between the candidate sets (order scores) be normalized,"
                                 "i.e. the probabilities to go from r in M(i) to any s in M(j) are summing up to one.")

    arg_parser.add_argument("--tree_method", type=str, choices=["random", "chain"], default="random",
                            help="Which tree approximation to use for the MRF (see Section 2.3.2).")

    arg_parser.add_argument("--n_random_trees", type=int, default=32,
                            help="Number of random spanning-trees to average the marginal distribution.")

    arg_parser.add_argument("--n_jobs", type=int, default=4,
                            help="Number of jobs used to parallelize the score-integration on the spanning-tree"
                                 "ensembles.")

    arg_parser.add_argument("--max_n_ms2", type=int, default=75,
                            help="Limit the maximum number of candidates per ms-feature. We use all candidates in the"
                                 "paper. This option is mainly used for development.")

    # Optional parameters dataset
    arg_parser.add_argument("--ion_mode", type=str, default="positive",
                            help="Load spectra only of the specified ionization mode. In the paper we always analyzed "
                                 "negative and positive mode spectra separately.")

    arg_parser.add_argument("--participant", type=str, default="MetFrag_2.4.5__8afe4a14",
                            help="MS2 scoring approach to be used. For the paper we used: 'MetFrag_2.4.5__8afe4a14' "
                                 "and 'IOKR__696a17f3'. See Section 3.3 for details.")

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

    arg_parser.add_argument("--restrict_candidates_to_correct_mf", action="store_true",
                            help="Should the candidate set be restricted to molecules with the same molecular formula "
                                 "as the correct candidate only. This option is used in the experiments presented in "
                                 "Section 4.3.2.")

    # Optional parameters data
    arg_parser.add_argument("--database_fn", type=str, help="Path to the score SQLite DB.")

    arg_parser.add_argument("--mode", type=str, default="debug_application",
                            choices=["debug_development", "development", "application", "debug_application"],
                            help="Mode to run this script. The parameter adds a sub-directory to the output with the "
                                 "same name, so that results can be analysed separately. 'debug' and 'development' "
                                 "are both running the (D, k) grid-search over training _and_ test set. They are "
                                 "mainly intended for development purposes. For example, the results for the parameter "
                                 "selection analysis in Section 4.2.2 are created using 'development'. 'application' "
                                 "and 'debug_application' are used to run our framework in the application scenario, "
                                 "i.e. finding the best (D, k)-tuple based on the training set and subsequently "
                                 "evaluating the score integration on the test set.")

    arg_parser.add_argument("--base_odir", type=str, default="results__TFG",
                            help="Base directory to store the results and output files.")

    # Optional parameters evaluation
    arg_parser.add_argument("--n_samples", type=int, default=50,
                            help="Number of random dataset sub-samples. Those have been already pre-sampled and are "
                                 "stored in the DB (see Section 3.1)")

    args = arg_parser.parse_args()
    print(str(args))

    # Define output directory
    if args.tree_method == "random":
        _tree_description = "random__n_trees=%d" % args.n_random_trees
        n_trees = args.n_random_trees
    elif args.tree_method == "chain":
        _tree_description = args.tree_method
        n_trees = 1
    else:
        raise ValueError("Invalid tree method: '%s'" % args.tree_method)

    _tmp = ["tree_method=%s" % _tree_description,
            "make_order_prob=%s" % args.make_order_prob]

    if args.mode in ["application", "debug_application"]:
        _tmp.append("param_selection_measure=%s" % args.param_selection_measure)

    _tmp.append("norm_order_scores=%d" % args.norm_order_scores)
    _tmp.append("mtype=%s" % args.margin_type)

    if args.restrict_candidates_to_correct_mf:
        _tmp.append("crcmf=%d" % args.restrict_candidates_to_correct_mf)

    odir = os.path.join(args.base_odir, args.mode,
                        "__".join(_tmp),
                        "__".join(["ion_mode=%s" % args.ion_mode,
                                   "participant=%s" % args.participant,
                                   "max_n_cand=%.0f" % args.max_n_cand,
                                   "pref_model=%s" % args.pref_model,
                                   "sort_candidates_by_ms2_score=%d" % args.sort_candidates_by_ms2_score]))
    os.makedirs(odir, exist_ok=True)
    os.makedirs(os.path.join(odir, "candidates"), exist_ok=True)
    os.makedirs(os.path.join(odir, "marginals"), exist_ok=True)

    # Load data
    challenges, candidates = load_data(args)

    # Set up hyper-parameter grid
    if args.order_prob_k_grid[0] == "platt":
        _k = [load_platt_k(args)]
        print("Platt k: %.5f" % _k[0])
    else:
        _k = list(map(float, args.order_prob_k_grid))
    h_param_grid = ParameterGrid({"D": args.D_value_grid, "k": _k})
    print("Number of grid-pairs: %d" % len(h_param_grid))

    measure_df = pd.DataFrame()
    opt_param_df = pd.DataFrame(columns=["sample", "D", "k"])

    for s, (train_set, test_set) in enumerate(ShuffleSplit(n_splits=args.n_samples, test_size=args.max_n_ms2,
                                                           random_state=29181).split(candidates)):
        print("rep=%d/%d" % (s + 1, args.n_samples))

        # Prepare the candidate set, i.e. sub-setting, sampling the MS2 (if needed), ...
        if args.participant == "MetFrag_2.4.5__8afe4a14":
            cnds_test = prepare_candidate_set_MetFrag(challenges, candidates, test_set, ms2_idc=range(len(test_set)),
                                                      verbose=True, normalize=False)
            cnds_train = prepare_candidate_set_MetFrag(challenges, candidates, train_set, ms2_idc=range(len(train_set)),
                                                       verbose=True, normalize=False)
        elif args.participant == "IOKR__696a17f3":
            cnds_test = prepare_candidate_set_IOKR(challenges, candidates, test_set, ms2_idc=range(len(test_set)),
                                                   verbose=True, normalize=False)
            cnds_train = prepare_candidate_set_IOKR(challenges, candidates, train_set, ms2_idc=range(len(train_set)),
                                                    verbose=True, normalize=False)
        else:
            raise ValueError("Invalid participant: '%s'" % args.participant)

        _names_test = set([v["name"] for v in cnds_test.values()])
        _names_train = set([v["name"] for v in cnds_train.values()])
        assert (not (_names_test & _names_train))
        print("\tn_train=%d, n_test=%d" % (len(train_set), len(test_set)))

        # Perform grid-search on training set
        df_train = evaluate_parameter_grid(
            *run_parameter_grid(cnds_train, h_param_grid, args.tree_method, n_trees, args.n_jobs, args.make_order_prob,
                                args.norm_order_scores, args.margin_type))
        df_train["set"] = "train"
        df_train["sample"] = s
        measure_df = pd.concat([measure_df, df_train], sort=True, axis=0)

        # Get top-k baseline performance of the test set
        topk_bsl_test_casmi = get_topk_performance_from_scores(cnds_test, None, method="casmi2016")
        measure_df = pd.concat([
            measure_df,
            pd.DataFrame({"sample": [s],
                          "set": ["test"],
                          "top1": [topk_bsl_test_casmi[1][0]],
                          "top3": [topk_bsl_test_casmi[1][2]],
                          "top5": [topk_bsl_test_casmi[1][4]],
                          "top10": [topk_bsl_test_casmi[1][9]],
                          "top20": [topk_bsl_test_casmi[1][19]],
                          "D": [0.0], "k": [None],
                          "topk_auc": [get_top20AUC(topk_bsl_test_casmi, len(cnds_test))]})],
            sort=True, axis=0)

        if args.mode in ["development", "debug_development"]:
            # Perform grid-search on test set for debugging and development purposes
            df_test = evaluate_parameter_grid(
                *run_parameter_grid(cnds_test, h_param_grid, args.tree_method, n_trees, args.n_jobs,
                                    args.make_order_prob, args.norm_order_scores, args.margin_type))
            df_test["set"] = "test"
            df_test["sample"] = s
            measure_df = pd.concat([measure_df, df_test], sort=True, axis=0)

            # Get top-k baseline performance of the training set for development purposes
            topk_bsl_train_casmi = get_topk_performance_from_scores(cnds_train, None, method="casmi2016")
            measure_df = pd.concat([
                measure_df,
                pd.DataFrame({"sample": [s],
                              "set": ["train"],
                              "top1": [topk_bsl_train_casmi[1][0]],
                              "top3": [topk_bsl_train_casmi[1][2]],
                              "top5": [topk_bsl_train_casmi[1][4]],
                              "top10": [topk_bsl_train_casmi[1][9]],
                              "top20": [topk_bsl_train_casmi[1][19]],
                              "D": [0.0], "k": [None],
                              "topk_auc": [get_top20AUC(topk_bsl_train_casmi, len(cnds_train))]})],
                sort=True, axis=0)

        elif args.mode in ["application", "debug_application"]:
            print("\tTop-5 parameters:")
            print(df_train[["D", "k", args.param_selection_measure, "top1", "top5"]]
                  .nlargest(5, columns=args.param_selection_measure))
            print("\tFlop-5 parameters:")
            print(df_train[["D", "k", args.param_selection_measure, "top1", "top5"]]
                  .nsmallest(5, columns=args.param_selection_measure).iloc[::-1])

            # Get the optimal (D, k) parameter tuple
            _idx = df_train[args.param_selection_measure].idxmax()

            D_opt = df_train.loc[_idx, "D"]
            k_opt = df_train.loc[_idx, "k"]
            print("\tOptimal parameters: D=%f, k=%f" % (D_opt, k_opt))
            opt_param_df.loc[s] = [s, D_opt, k_opt]

            # Run Forward-Backward algorithm on the test set
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(get_marginals)(cnds_test, D=D_opt, k=k_opt, tree_method=args.tree_method, rep=rep,
                                       make_order_prob=args.make_order_prob, norm_order_scores=args.norm_order_scores,
                                       margin_type=args.margin_type)
                for rep in range(n_trees))

            # Average the marginals across trees
            marg_test = {k: np.zeros(v["n_cand"]) for k, v in cnds_test.items()}
            for i in marg_test:
                for r in res:
                    marg_test[i] += r[2][i]
                marg_test[i] /= n_trees

            # Calculate the top-k performance on the test set using RT+MS
            topk_test_casmi = {"ms_rt": get_topk_performance_from_scores(cnds_test, marg_test, method="casmi2016"),
                               "baseline": get_topk_performance_from_scores(cnds_test, None, method="casmi2016")}
            print("\tMS only: top1=%d (%.2f%%), top5=%d (%.2f%%), top10=%d (%.2f%%), top20=%d (%.2f%%)\n"
                  "\tMS + RT: top1=%d (%.2f%%), top5=%d (%.2f%%), top10=%d (%.2f%%), top20=%d (%.2f%%)\n"
                  % (topk_test_casmi["baseline"][0][0], topk_test_casmi["baseline"][1][0],
                     topk_test_casmi["baseline"][0][4], topk_test_casmi["baseline"][1][4],
                     topk_test_casmi["baseline"][0][9], topk_test_casmi["baseline"][1][9],
                     topk_test_casmi["baseline"][0][19], topk_test_casmi["baseline"][1][19],
                     topk_test_casmi["ms_rt"][0][0], topk_test_casmi["ms_rt"][1][0],
                     topk_test_casmi["ms_rt"][0][4], topk_test_casmi["ms_rt"][1][4],
                     topk_test_casmi["ms_rt"][0][9], topk_test_casmi["ms_rt"][1][9],
                     topk_test_casmi["ms_rt"][0][19], topk_test_casmi["ms_rt"][1][19]))

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

            # Write out marginal
            with gzip.open(os.path.join(odir, "marginals",
                                        "__".join(["marginals",
                                                   "max_n_ms2=%03d" % args.max_n_ms2,
                                                   "sample_id=%03d" % s]) + ".pkl.gz"), "wb+") as of:
                pickle.dump(marg_test, of, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("Invalid mode: '%s'" % args.mode)

    # Write out stuff for the ranking measurements and parameter selection evaluation
    measure_df.to_csv(os.path.join(odir, "measures.csv"), index=False)

    if args.mode in ["application", "debug_application"]:
        opt_param_df.to_csv(os.path.join(odir, "opt_params.csv"), index=False)

    sys.exit(0)
